// This binary requires GStreamer 1.x development libraries on the system:
// `libgstreamer1.0-dev`, `libgstreamer-plugins-base1.0-dev`, and
// `gstreamer1.0-plugins-good` for the `v4l2src` and `jpegenc` elements.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result, anyhow};
use gst::prelude::*;
use gstreamer as gst;
use gstreamer_app as gst_app;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

const CAMERA_DEVICE: &str = "/dev/video0";
const PIPELINE_STR: &str = concat!(
    "v4l2src device=/dev/video0 ! ",
    "video/x-raw,width=1280,height=720,framerate=30/1 ! ",
    "videoconvert ! ",
    "jpegenc quality=75 ! ",
    "appsink name=sink emit-signals=true sync=false drop=true max-buffers=2"
);
const VLM_INTERVAL: Duration = Duration::from_secs(1);

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(true)
        .init();

    run().await
}

async fn run() -> Result<()> {
    let pipeline_str = std::env::var("ARGUS_PIPELINE").unwrap_or_else(|_| PIPELINE_STR.to_string());
    let uses_v4l2 = pipeline_str.contains("v4l2src");

    if uses_v4l2 && !Path::new(CAMERA_DEVICE).exists() {
        error!(
            "Camera device {CAMERA_DEVICE} was not found. Check the CSI camera, overlay, and V4L2 setup."
        );
        return Err(anyhow!("missing camera device at {CAMERA_DEVICE}"));
    }

    // GStreamer must be initialized before creating any elements or pipelines.
    gst::init().context("failed to initialize GStreamer")?;

    // Build the camera pipeline from a string so it is easy to mirror on-device
    // when debugging with `gst-launch-1.0`.
    let pipeline = build_pipeline(&pipeline_str).context("failed to build camera pipeline")?;
    let appsink = pipeline_appsink(&pipeline).context("failed to access appsink named `sink`")?;

    // Two fan-out channels:
    // - live_stream: latest JPEG frames for WebSocket streaming
    // - vlm: one JPEG roughly every second for object detection
    let (live_stream_tx, mut live_stream_rx) = mpsc::channel::<Vec<u8>>(2);
    let (vlm_tx, mut vlm_rx) = mpsc::channel::<Vec<u8>>(1);

    let live_stream_callback_tx = live_stream_tx.clone();
    let vlm_callback_tx = vlm_tx.clone();
    let last_vlm_send = Arc::new(Mutex::new(Instant::now() - VLM_INTERVAL));

    install_appsink_callbacks(
        &appsink,
        live_stream_callback_tx,
        vlm_callback_tx,
        Arc::clone(&last_vlm_send),
    );

    // Drop the main-task senders so channel lifetime is owned by the callback.
    drop(live_stream_tx);
    drop(vlm_tx);

    // Placeholder consumer for the live desktop stream path.
    let live_task = tokio::spawn(async move {
        while let Some(frame_bytes) = live_stream_rx.recv().await {
            handle_live_frame(frame_bytes);
        }
    });

    // Placeholder consumer for the 1 FPS VLM path.
    let vlm_task = tokio::spawn(async move {
        while let Some(frame_bytes) = vlm_rx.recv().await {
            handle_vlm_frame(frame_bytes);
        }
    });

    // Watch the GStreamer bus on a dedicated thread so async camera and pipeline
    // errors can trigger a clean shutdown from `tokio::select!`.
    let bus = pipeline.bus().context("pipeline did not expose a bus")?;
    let (pipeline_error_tx, mut pipeline_error_rx) = mpsc::channel::<String>(1);
    thread::spawn(move || {
        for message in bus.iter_timed(gst::ClockTime::NONE) {
            if let gst::MessageView::Error(err) = message.view() {
                let source = err
                    .src()
                    .map(|src| src.path_string())
                    .unwrap_or_else(|| "unknown".into());
                let details = err
                    .debug()
                    .map(|debug| debug.to_string())
                    .unwrap_or_else(|| "no debug details".to_string());
                let formatted = format!(
                    "GStreamer pipeline error from {source}: {} ({details})",
                    err.error()
                );

                let _ = pipeline_error_tx.blocking_send(formatted);
                break;
            }
        }
    });

    pipeline
        .set_state(gst::State::Playing)
        .context("failed to set the camera pipeline to Playing")?;
    info!("Camera pipeline started: {pipeline_str}");

    tokio::select! {
        result = tokio::signal::ctrl_c() => {
            result.context("failed to listen for Ctrl+C")?;
            info!("Ctrl+C received, shutting down camera pipeline");
        }
        maybe_error = pipeline_error_rx.recv() => {
            match maybe_error {
                Some(message) => {
                    error!("{message}");
                    shutdown_pipeline(pipeline, appsink, live_task, vlm_task).await?;
                    return Err(anyhow!("camera pipeline stopped due to a GStreamer error"));
                }
                None => {
                    warn!("Pipeline error watcher ended unexpectedly");
                }
            }
        }
    }

    shutdown_pipeline(pipeline, appsink, live_task, vlm_task).await
}

fn build_pipeline(pipeline_str: &str) -> Result<gst::Pipeline> {
    let element = gst::parse::launch(pipeline_str).context("unable to parse pipeline string")?;
    element
        .downcast::<gst::Pipeline>()
        .map_err(|_| anyhow!("parsed pipeline was not a gst::Pipeline"))
}

fn pipeline_appsink(pipeline: &gst::Pipeline) -> Result<gst_app::AppSink> {
    pipeline
        .by_name("sink")
        .context("pipeline is missing an element named `sink`")?
        .downcast::<gst_app::AppSink>()
        .map_err(|_| anyhow!("element named `sink` is not an appsink"))
}

fn install_appsink_callbacks(
    appsink: &gst_app::AppSink,
    live_stream_tx: mpsc::Sender<Vec<u8>>,
    vlm_tx: mpsc::Sender<Vec<u8>>,
    last_vlm_send: Arc<Mutex<Instant>>,
) {
    // The appsink callback runs on a GStreamer streaming thread, so it must stay
    // synchronous and non-blocking. We only clone frame bytes and `try_send`.
    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |sink| {
                let sample = sink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                let buffer = sample.buffer().ok_or(gst::FlowError::Error)?;
                let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;
                let frame_bytes = map.as_slice().to_vec();

                if live_stream_tx.try_send(frame_bytes.clone()).is_err() {
                    debug!("Dropping live stream frame because the channel is full");
                }

                let should_send_to_vlm = {
                    let mut last_send = last_vlm_send.lock().map_err(|_| gst::FlowError::Error)?;
                    if last_send.elapsed() >= VLM_INTERVAL {
                        *last_send = Instant::now();
                        true
                    } else {
                        false
                    }
                };

                if should_send_to_vlm && vlm_tx.try_send(frame_bytes).is_err() {
                    debug!("Dropping VLM frame because the channel is full");
                }

                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );
}

async fn shutdown_pipeline(
    pipeline: gst::Pipeline,
    appsink: gst_app::AppSink,
    live_task: tokio::task::JoinHandle<()>,
    vlm_task: tokio::task::JoinHandle<()>,
) -> Result<()> {
    pipeline
        .set_state(gst::State::Null)
        .context("failed to set the camera pipeline to Null")?;
    info!("Camera pipeline stopped");

    // Dropping the pipeline releases the appsink callback and closes the channels,
    // which lets the placeholder consumer tasks exit naturally.
    drop(appsink);
    drop(pipeline);

    if let Err(join_error) = live_task.await {
        warn!("Live stream task ended unexpectedly: {join_error}");
    }
    if let Err(join_error) = vlm_task.await {
        warn!("VLM task ended unexpectedly: {join_error}");
    }

    Ok(())
}

fn handle_live_frame(frame_bytes: Vec<u8>) {
    info!("Live frame: {} bytes", frame_bytes.len());
    // TODO: send over WebSocket to desktop app
}

fn handle_vlm_frame(frame_bytes: Vec<u8>) {
    info!(
        "VLM frame selected at {:?}: {} bytes",
        SystemTime::now(),
        frame_bytes.len()
    );
    // TODO: send to VLM API for object detection, update memory graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    const TEST_PIPELINE_STR: &str = concat!(
        "videotestsrc is-live=true pattern=ball ! ",
        "video/x-raw,width=1280,height=720,framerate=30/1 ! ",
        "videoconvert ! ",
        "jpegenc quality=75 ! ",
        "appsink name=sink emit-signals=true sync=false drop=true max-buffers=2"
    );

    fn init_gstreamer() {
        static INIT: OnceLock<()> = OnceLock::new();
        INIT.get_or_init(|| gst::init().expect("failed to initialize GStreamer for tests"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn videotestsrc_produces_live_and_vlm_frames() {
        init_gstreamer();

        let pipeline = build_pipeline(TEST_PIPELINE_STR).expect("failed to build test pipeline");
        let appsink = pipeline_appsink(&pipeline).expect("missing test appsink");

        let (live_stream_tx, mut live_stream_rx) = mpsc::channel::<Vec<u8>>(2);
        let (vlm_tx, mut vlm_rx) = mpsc::channel::<Vec<u8>>(1);

        install_appsink_callbacks(
            &appsink,
            live_stream_tx,
            vlm_tx,
            Arc::new(Mutex::new(Instant::now() - VLM_INTERVAL)),
        );

        pipeline
            .set_state(gst::State::Playing)
            .expect("failed to start test pipeline");

        let live_frame = tokio::time::timeout(Duration::from_secs(2), live_stream_rx.recv())
            .await
            .expect("timed out waiting for a live frame")
            .expect("live frame channel closed unexpectedly");
        assert!(
            !live_frame.is_empty(),
            "live frame should contain JPEG bytes"
        );

        let vlm_frame = tokio::time::timeout(Duration::from_secs(2), vlm_rx.recv())
            .await
            .expect("timed out waiting for a VLM frame")
            .expect("VLM frame channel closed unexpectedly");
        assert!(!vlm_frame.is_empty(), "VLM frame should contain JPEG bytes");

        pipeline
            .set_state(gst::State::Null)
            .expect("failed to stop test pipeline");
    }
}
