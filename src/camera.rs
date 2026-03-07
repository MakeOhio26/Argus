// This binary requires GStreamer 1.x development libraries on the system:
// `libgstreamer1.0-dev`, `libgstreamer-plugins-base1.0-dev`, and
// `gstreamer1.0-plugins-good` for the `v4l2src` and `jpegenc` elements.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use gstreamer as gst;
use gstreamer_app as gst_app;
use gst::prelude::*;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

pub const CAMERA_DEVICE: &str = "/dev/video0";
pub const PIPELINE_STR: &str = concat!(
    "v4l2src device=/dev/video0 ! ",
    "video/x-raw,width=1280,height=720,framerate=30/1 ! ",
    "videoconvert ! ",
    "jpegenc quality=75 ! ",
    "appsink name=sink emit-signals=true sync=false drop=true max-buffers=2"
);
pub const VLM_INTERVAL: Duration = Duration::from_secs(1);

pub fn build_pipeline(pipeline_str: &str) -> Result<gst::Pipeline> {
    let element = gst::parse::launch(pipeline_str).context("unable to parse pipeline string")?;
    element
        .downcast::<gst::Pipeline>()
        .map_err(|_| anyhow!("parsed pipeline was not a gst::Pipeline"))
}

pub fn pipeline_appsink(pipeline: &gst::Pipeline) -> Result<gst_app::AppSink> {
    pipeline
        .by_name("sink")
        .context("pipeline is missing an element named `sink`")?
        .downcast::<gst_app::AppSink>()
        .map_err(|_| anyhow!("element named `sink` is not an appsink"))
}

pub fn install_appsink_callbacks(
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

pub async fn shutdown_pipeline(
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
