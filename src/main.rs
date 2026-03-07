// This binary requires GStreamer 1.x development libraries on the system:
// `libgstreamer1.0-dev`, `libgstreamer-plugins-base1.0-dev`, and
// `gstreamer1.0-plugins-good` for the `v4l2src` and `jpegenc` elements.

// GStreamer camera pipeline — binary-only, not part of the library.
mod camera;

// Memory modules come from the library crate (src/lib.rs).
use argus::frame::Frame;
use argus::frame_store::FrameStore;
use argus::gemini::ReqwestGeminiClient;
use argus::memory::MemorySystem;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use gstreamer as gst;
use gstreamer::prelude::*;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use camera::{
    CAMERA_DEVICE, PIPELINE_STR, VLM_INTERVAL, build_pipeline, install_appsink_callbacks,
    pipeline_appsink, shutdown_pipeline,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file if present (silently ignored if missing).
    // This lets you put GEMINI_API_KEY=... in a local .env instead of
    // exporting it every shell session.
    dotenvy::dotenv().ok();

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

    let pipeline = build_pipeline(&pipeline_str).context("failed to build camera pipeline")?;
    let appsink = pipeline_appsink(&pipeline).context("failed to access appsink named `sink`")?;

    // Two fan-out channels:
    // - live_stream: latest JPEG frames for WebSocket streaming (TODO)
    // - vlm: one JPEG roughly every second for Gemini analysis
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

    drop(live_stream_tx);
    drop(vlm_tx);

    // Build the memory system.
    //
    // GEMINI_API_KEY can be set in a .env file or exported in the shell.
    // The FrameStore keeps up to 10 novel (perceptually distinct) frames
    // per observed entity on disk under ./argus_store/.
    let api_key = std::env::var("GEMINI_API_KEY")
        .context("GEMINI_API_KEY not set — add it to .env or export it")?;
    let gemini_client = Box::new(ReqwestGeminiClient::new(api_key));
    let frame_store =
        FrameStore::new("./argus_store", 10).context("failed to create frame store")?;
    let mut memory = MemorySystem::new(gemini_client, frame_store);

    // Live stream placeholder — TODO: forward to WebSocket.
    let live_task = tokio::spawn(async move {
        while let Some(frame_bytes) = live_stream_rx.recv().await {
            info!("Live frame: {} bytes", frame_bytes.len());
            // TODO: send over WebSocket to desktop app
        }
    });

    // VLM task — sends one frame per second to Gemini, updates memory graph.
    //
    // `MemorySystem` is owned exclusively by this task.  No Arc/Mutex needed
    // because only one frame is processed at a time — if Gemini is slow the
    // VLM channel fills up (capacity=1) and GStreamer drops the next frame,
    // which is intentional for a 1 FPS subsampled stream.
    let vlm_task = tokio::spawn(async move {
        let mut counter = 0u64;
        while let Some(frame_bytes) = vlm_rx.recv().await {
            counter += 1;
            match Frame::new(format!("frame_{counter}"), frame_bytes) {
                Ok(frame) => {
                    if let Err(e) = memory.process_frame(frame).await {
                        warn!("Memory update failed: {e}");
                    } else {
                        info!(
                            "Graph updated — entities: {}, relations: {}",
                            memory.query(|g| g.entity_count()),
                            memory.query(|g| g.relation_count()),
                        );
                    }
                }
                Err(e) => warn!("Invalid frame bytes: {e}"),
            }
        }
    });

    // Watch the GStreamer bus on a dedicated thread.
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
                    .map(|d| d.to_string())
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
