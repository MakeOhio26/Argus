// This binary requires GStreamer 1.x development libraries on the system:
// `libgstreamer1.0-dev`, `libgstreamer-plugins-base1.0-dev`, and
// `gstreamer1.0-plugins-good` for the `v4l2src` and `jpegenc` elements.

// GStreamer camera pipeline — binary-only, not part of the library.
mod camera;

// Memory modules come from the library crate (src/lib.rs).
use argus::frame::Frame;
use argus::frame_store::FrameStore;
use argus::gemini::{ReqwestGeminiClient, StubGeminiClient};
use argus::memory::MemorySystem;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use futures_util::SinkExt;
use gstreamer as gst;
use gstreamer::prelude::*;
use tokio::sync::{broadcast, mpsc};
use tokio_tungstenite::tungstenite::Message;
use tracing::{error, info, warn};

use camera::{
    VLM_INTERVAL, build_pipeline, format_pipeline_error, install_appsink_callbacks,
    pipeline_appsink, resolve_camera_config, shutdown_pipeline,
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

async fn serve_ws_client(
    stream: tokio::net::TcpStream,
    mut rx: broadcast::Receiver<Vec<u8>>,
) -> Result<()> {
    let ws = tokio_tungstenite::accept_async(stream).await?;
    let (mut sink, _source) = futures_util::StreamExt::split(ws);
    loop {
        match rx.recv().await {
            Ok(frame_bytes) => sink.send(Message::Binary(frame_bytes.into())).await?,
            Err(broadcast::error::RecvError::Lagged(n)) => {
                warn!("WebSocket client lagged, dropped {n} frames");
            }
            Err(broadcast::error::RecvError::Closed) => break,
        }
    }
    Ok(())
}

async fn run() -> Result<()> {
    let stub_mode = std::env::args().any(|a| a == "--stub");
    let camera_config = resolve_camera_config()?;

    // GStreamer must be initialized before creating any elements or pipelines.
    gst::init().context("failed to initialize GStreamer")?;

    let pipeline =
        build_pipeline(&camera_config.pipeline_str).context("failed to build camera pipeline")?;
    let appsink = pipeline_appsink(&pipeline).context("failed to access appsink named `sink`")?;

    // Two fan-out channels:
    // - live_stream: broadcast of JPEG frames to all WebSocket clients
    // - vlm: one JPEG roughly every second for Gemini analysis
    let (live_stream_tx, _) = broadcast::channel::<Vec<u8>>(16);
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

    drop(vlm_tx);

    // Build the memory system.
    //
    // GEMINI_API_KEY can be set in a .env file or exported in the shell.
    // The FrameStore keeps up to 10 novel (perceptually distinct) frames
    // per observed entity on disk under ./argus_store/.
    let gemini_client: Box<dyn argus::gemini::GeminiClient> = if stub_mode {
        info!("--stub flag set: using StubGeminiClient (no real Gemini calls)");
        Box::new(StubGeminiClient)
    } else {
        let api_key = std::env::var("GEMINI_API_KEY")
            .context("GEMINI_API_KEY not set — add it to .env or export it")?;
        Box::new(ReqwestGeminiClient::new(api_key))
    };
    let frame_store =
        FrameStore::new("./argus_store", 10).context("failed to create frame store")?;
    let mut memory = MemorySystem::new(gemini_client, frame_store, Some(PathBuf::from("./argus_graph.json")));

    // WebSocket server — broadcasts live JPEG frames to connected clients.
    let ws_bind_addr = std::env::var("ARGUS_WS_BIND_ADDR")
        .unwrap_or_else(|_| "0.0.0.0".to_string());
    let ws_port: u16 = std::env::var("ARGUS_WS_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8765);
    let listener = tokio::net::TcpListener::bind((ws_bind_addr.as_str(), ws_port))
        .await
        .context("failed to bind WebSocket listener")?;
    log_ws_listener(&ws_bind_addr, ws_port);

    let live_task = tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, addr)) => {
                    info!("WebSocket client connected: {addr}");
                    let rx = live_stream_tx.subscribe();
                    tokio::spawn(async move {
                        if let Err(e) = serve_ws_client(stream, rx).await {
                            warn!("WebSocket client {addr} disconnected: {e}");
                        }
                    });
                }
                Err(e) => {
                    error!("WebSocket accept error: {e}");
                    break;
                }
            }
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
    let camera_backend = camera_config.backend;
    thread::spawn(move || {
        for message in bus.iter_timed(gst::ClockTime::NONE) {
            if let gst::MessageView::Error(err) = message.view() {
                let formatted = format_pipeline_error(
                    camera_backend,
                    err.src().map(|src| src.path_string().to_string()),
                    err.error().to_string(),
                    err.debug().map(|debug| debug.to_string()),
                );
                let _ = pipeline_error_tx.blocking_send(formatted);
                break;
            }
        }
    });

    pipeline
        .set_state(gst::State::Playing)
        .context("failed to set the camera pipeline to Playing")?;
    info!("Camera backend selected: {}", camera_config.backend.as_str());
    info!("Camera pipeline started: {}", camera_config.pipeline_str);

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

fn log_ws_listener(bind_addr: &str, ws_port: u16) {
    info!("WebSocket live feed listening on {bind_addr}:{ws_port}");

    if bind_addr == "0.0.0.0" {
        info!("Connect locally via ws://127.0.0.1:{ws_port}/");
        info!("Connect remotely via ws://<device-ip>:{ws_port}/");
    } else {
        info!("Connect via ws://{bind_addr}:{ws_port}/");
    }
}
