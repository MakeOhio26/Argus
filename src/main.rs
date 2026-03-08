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
use argus::rover::{DefaultRoverClient, RoverError};

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use futures_util::SinkExt;
use gstreamer as gst;
use gstreamer::prelude::*;
use tokio::sync::{RwLock, broadcast, mpsc, watch};
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

/// Run a blocking rover command on the thread pool and return a JSON response value.
async fn invoke_rover<F>(
    rover: &Arc<std::sync::Mutex<Option<DefaultRoverClient>>>,
    command: &'static str,
    f: F,
) -> serde_json::Value
where
    F: FnOnce(&mut DefaultRoverClient) -> Result<String, RoverError> + Send + 'static,
{
    let rover_clone = Arc::clone(rover);
    let result = tokio::task::spawn_blocking(move || {
        let mut guard = rover_clone.lock().unwrap();
        match guard.as_mut() {
            None => Err("rover not connected".to_string()),
            Some(r) => f(r).map_err(|e| e.to_string()),
        }
    })
    .await;

    match result {
        Ok(Ok(response)) => serde_json::json!({
            "type": "rover_result",
            "command": command,
            "response": response,
        }),
        Ok(Err(msg)) => serde_json::json!({ "type": "error", "message": msg }),
        Err(_) => serde_json::json!({ "type": "error", "message": "internal task error" }),
    }
}

async fn serve_ws_client(
    stream: tokio::net::TcpStream,
    mut frame_rx: broadcast::Receiver<Vec<u8>>,
    mut graph_rx: broadcast::Receiver<String>,
    initial_graph: Arc<RwLock<String>>,
    frame_store: Arc<FrameStore>,
    rover: Arc<std::sync::Mutex<Option<DefaultRoverClient>>>,
) -> Result<()> {
    use futures_util::StreamExt;

    let ws = tokio_tungstenite::accept_async(stream).await?;
    let (mut sink, mut source) = ws.split();

    // Send current graph snapshot immediately on connect (if one exists).
    {
        let cell = initial_graph.read().await;
        if !cell.is_empty() {
            sink.send(Message::Text(cell.clone().into())).await?;
        }
    }

    loop {
        tokio::select! {
            result = frame_rx.recv() => {
                match result {
                    Ok(bytes) => sink.send(Message::Binary(bytes.into())).await?,
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!("WebSocket client lagged on video, dropped {n} frames");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            result = graph_rx.recv() => {
                match result {
                    Ok(json) => sink.send(Message::Text(json.into())).await?,
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!("WebSocket client lagged on graph, dropped {n} updates");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            maybe_msg = source.next() => {
                match maybe_msg {
                    None => break,
                    Some(Err(e)) => { warn!("WebSocket receive error: {e}"); break; }
                    Some(Ok(Message::Text(text))) => {
                        match serde_json::from_str::<serde_json::Value>(&text) {
                            Ok(v) if v["type"] == "get_graph" => {
                                let cell = initial_graph.read().await;
                                if !cell.is_empty() {
                                    sink.send(Message::Text(cell.clone().into())).await?;
                                }
                            }
                            Ok(v) if v["type"] == "get_entity_images" => {
                                let entity = v["entity"].as_str().unwrap_or("").to_string();
                                if entity.is_empty() {
                                    let _ = sink.send(Message::Text(
                                        r#"{"type":"error","message":"missing entity field"}"#.into()
                                    )).await;
                                } else {
                                    let store = Arc::clone(&frame_store);
                                    let entity_key = entity.clone();
                                    let result = tokio::task::spawn_blocking(move || {
                                        store.load_all(&entity_key)
                                    }).await;
                                    match result {
                                        Ok(Ok(images)) => {
                                            use base64::{Engine, engine::general_purpose::STANDARD};
                                            let encoded: Vec<String> = images.iter()
                                                .map(|b| STANDARD.encode(b))
                                                .collect();
                                            let response = serde_json::json!({
                                                "type": "entity_images",
                                                "entity": entity,
                                                "images": encoded,
                                            });
                                            if let Ok(json) = serde_json::to_string(&response) {
                                                let _ = sink.send(Message::Text(json.into())).await;
                                            }
                                        }
                                        _ => {
                                            let _ = sink.send(Message::Text(
                                                r#"{"type":"error","message":"failed to load entity images"}"#.into()
                                            )).await;
                                        }
                                    }
                                }
                            }
                            Ok(v) if v["type"] == "rover_ping" => {
                                let json = invoke_rover(&rover, "ping", |r| r.ping()).await;
                                if let Ok(s) = serde_json::to_string(&json) {
                                    let _ = sink.send(Message::Text(s.into())).await;
                                }
                            }
                            Ok(v) if v["type"] == "rover_read_air" => {
                                let json = invoke_rover(&rover, "read_air", |r| r.read_air()).await;
                                if let Ok(s) = serde_json::to_string(&json) {
                                    let _ = sink.send(Message::Text(s.into())).await;
                                }
                            }
                            Ok(v) if v["type"] == "rover_read_motion" => {
                                let json = invoke_rover(&rover, "read_motion", |r| r.read_motion()).await;
                                if let Ok(s) = serde_json::to_string(&json) {
                                    let _ = sink.send(Message::Text(s.into())).await;
                                }
                            }
                            Ok(v) if v["type"] == "rover_zero" => {
                                let json = invoke_rover(&rover, "zero", |r| r.zero()).await;
                                if let Ok(s) = serde_json::to_string(&json) {
                                    let _ = sink.send(Message::Text(s.into())).await;
                                }
                            }
                            Ok(v) if v["type"] == "rover_stop" => {
                                let json = invoke_rover(&rover, "stop", |r| r.stop()).await;
                                if let Ok(s) = serde_json::to_string(&json) {
                                    let _ = sink.send(Message::Text(s.into())).await;
                                }
                            }
                            Ok(v) if v["type"] == "rover_rotate" => {
                                match v["degrees"].as_f64() {
                                    None => {
                                        let _ = sink.send(Message::Text(
                                            r#"{"type":"error","message":"missing or invalid degrees field"}"#.into()
                                        )).await;
                                    }
                                    Some(deg) => {
                                        let json = invoke_rover(&rover, "rotate", move |r| r.rotate(deg)).await;
                                        if let Ok(s) = serde_json::to_string(&json) {
                                            let _ = sink.send(Message::Text(s.into())).await;
                                        }
                                    }
                                }
                            }
                            Ok(v) if v["type"] == "rover_distance" => {
                                match v["centimeters"].as_f64() {
                                    None => {
                                        let _ = sink.send(Message::Text(
                                            r#"{"type":"error","message":"missing or invalid centimeters field"}"#.into()
                                        )).await;
                                    }
                                    Some(cm) => {
                                        let json = invoke_rover(&rover, "distance", move |r| r.distance(cm)).await;
                                        if let Ok(s) = serde_json::to_string(&json) {
                                            let _ = sink.send(Message::Text(s.into())).await;
                                        }
                                    }
                                }
                            }
                            Ok(_) => {
                                let _ = sink.send(Message::Text(
                                    r#"{"type":"error","message":"unrecognized message"}"#.into()
                                )).await;
                            }
                            Err(_) => warn!("Malformed WebSocket message"),
                        }
                    }
                    Some(Ok(Message::Close(_))) => break,
                    Some(Ok(_)) => {} // ignore binary/ping/pong from client
                }
            }
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

    // Graph update broadcast: pushes JSON text to all connected WS clients after each VLM frame.
    let (graph_tx, _) = broadcast::channel::<String>(8);
    let graph_tx_vlm = graph_tx.clone();
    // Shared latest graph envelope — new connections read this for an immediate snapshot.
    let current_graph: Arc<RwLock<String>> = Arc::new(RwLock::new(String::new()));

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
        Arc::new(FrameStore::new("./argus_store", 10).context("failed to create frame store")?);
    let mut memory = MemorySystem::new(gemini_client, Arc::clone(&frame_store), Some(PathBuf::from("./argus_graph.json")));

    let (shutdown_tx, _) = watch::channel(false);
    let mut shutdown_signal_rx = shutdown_tx.subscribe();
    let ctrlc_shutdown_tx = shutdown_tx.clone();
    ctrlc::set_handler(move || {
        let _ = ctrlc_shutdown_tx.send(true);
    })
    .context("failed to install Ctrl+C handler")?;

    let (shutdown_tx, _) = watch::channel(false);
    let mut shutdown_signal_rx = shutdown_tx.subscribe();
    let ctrlc_shutdown_tx = shutdown_tx.clone();
    ctrlc::set_handler(move || {
        let _ = ctrlc_shutdown_tx.send(true);
    })
    .context("failed to install Ctrl+C handler")?;

    // Rover serial client — optional, gated on ARGUS_ROVER_PORT env var.
    let rover_client: Arc<std::sync::Mutex<Option<DefaultRoverClient>>> =
        match std::env::var("ARGUS_ROVER_PORT") {
            Err(_) => {
                info!("ARGUS_ROVER_PORT not set — rover commands will return 'not connected'");
                Arc::new(std::sync::Mutex::new(None))
            }
            Ok(port) => match argus::rover::open(&port) {
                Ok(client) => {
                    info!("Rover connected on {port}");
                    Arc::new(std::sync::Mutex::new(Some(client)))
                }
                Err(e) => {
                    warn!("Failed to open rover on {port}: {e} — continuing without rover");
                    Arc::new(std::sync::Mutex::new(None))
                }
            },
        };

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

    let current_graph_listener = Arc::clone(&current_graph);
    let frame_store_listener = Arc::clone(&frame_store);
    let rover_listener = Arc::clone(&rover_client);
    let mut live_shutdown_rx = shutdown_tx.subscribe();
    let live_task = tokio::spawn(async move {
        loop {
            tokio::select! {
                changed = live_shutdown_rx.changed() => {
                    if changed.is_ok() && *live_shutdown_rx.borrow() {
                        info!("WebSocket server shutting down");
                        break;
                    }
                }
                result = listener.accept() => {
                    match result {
                        Ok((stream, addr)) => {
                            info!("WebSocket client connected: {addr}");
                            let frame_rx = live_stream_tx.subscribe();
                            let graph_rx = graph_tx.subscribe();
                            let initial = Arc::clone(&current_graph_listener);
                            let store = Arc::clone(&frame_store_listener);
                    let rover = Arc::clone(&rover_listener);
                            tokio::spawn(async move {
                                if let Err(e) = serve_ws_client(stream, frame_rx, graph_rx, initial, store, rover).await {
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
            }
        }
    });

    // VLM task — sends one frame per second to Gemini, updates memory graph.
    //
    // `MemorySystem` is owned exclusively by this task.  No Arc/Mutex needed
    // because only one frame is processed at a time — if Gemini is slow the
    // VLM channel fills up (capacity=1) and GStreamer drops the next frame,
    // which is intentional for a 1 FPS subsampled stream.
    let current_graph_vlm = Arc::clone(&current_graph);
    let vlm_task = tokio::spawn(async move {
        let mut counter = 0u64;
        let mut graph_seq = 0u64;
        while let Some(frame_bytes) = vlm_rx.recv().await {
            counter += 1;
            match Frame::new(format!("frame_{counter}"), frame_bytes) {
                Ok(frame) => {
                    if let Err(e) = memory.process_frame_with_voc_trend(frame, false).await {
                        warn!("Memory update failed: {e}");
                    } else {
                        info!(
                            "Graph updated — entities: {}, relations: {}",
                            memory.query(|g| g.entity_count()),
                            memory.query(|g| g.relation_count()),
                        );
                        // Serialize graph and push to all connected WS clients.
                        match memory.graph_json() {
                            Ok(inner) => {
                                // Wrap in envelope: inject "type" and "seq" into the JSON object.
                                match serde_json::from_str::<serde_json::Value>(&inner) {
                                    Ok(mut val) => {
                                        val["type"] = serde_json::json!("graph_update");
                                        val["seq"]  = serde_json::json!(graph_seq);
                                        if let Ok(envelope) = serde_json::to_string(&val) {
                                            *current_graph_vlm.write().await = envelope.clone();
                                            let _ = graph_tx_vlm.send(envelope);
                                            graph_seq += 1;
                                        }
                                    }
                                    Err(e) => warn!("Failed to wrap graph envelope: {e}"),
                                }
                            }
                            Err(e) => warn!("Failed to serialize graph: {e}"),
                        }
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
        changed = shutdown_signal_rx.changed() => {
            if changed.is_err() || !*shutdown_signal_rx.borrow() {
                return Err(anyhow!("Ctrl+C handler ended unexpectedly"));
            }
            info!("Ctrl+C received, shutting down camera pipeline");
        }
        maybe_error = pipeline_error_rx.recv() => {
            match maybe_error {
                Some(message) => {
                    error!("{message}");
                    let _ = shutdown_tx.send(true);
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
