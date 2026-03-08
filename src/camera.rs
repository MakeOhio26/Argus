// This binary requires GStreamer 1.x development libraries on the system:
// `libgstreamer1.0-dev`, `libgstreamer-plugins-base1.0-dev`, and
// `gstreamer1.0-plugins-good` for the `v4l2src` and `jpegenc` elements.
// On Raspberry Pi 5 systems using libcamera from `/usr/local`, the GStreamer
// plugin path must include `/usr/local/lib/aarch64-linux-gnu/gstreamer-1.0`.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use gstreamer as gst;
use gstreamer_app as gst_app;
use gst::prelude::*;
use tokio::sync::{broadcast, mpsc};
use tracing::{debug, error, info, warn};

const DEFAULT_V4L2_DEVICE: &str = "/dev/video0";
const DEFAULT_GST_PLUGIN_PATH: &str = "/usr/local/lib/aarch64-linux-gnu/gstreamer-1.0";
const DEFAULT_CAMERA_ORIENTATION: &str = "rotate-180";
pub const VLM_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CameraBackend {
    Libcamera,
    V4l2,
}

impl CameraBackend {
    fn from_env() -> Result<Self> {
        match std::env::var("ARGUS_CAMERA_BACKEND")
            .unwrap_or_else(|_| "libcamera".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "libcamera" => Ok(Self::Libcamera),
            "v4l2" => Ok(Self::V4l2),
            other => Err(anyhow!(
                "unsupported ARGUS_CAMERA_BACKEND `{other}`; expected `libcamera` or `v4l2`"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Libcamera => "libcamera",
            Self::V4l2 => "v4l2",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CameraOrientation {
    None,
    Rotate90,
    Rotate180,
    Rotate270,
    HorizontalFlip,
    VerticalFlip,
}

impl CameraOrientation {
    fn from_env() -> Result<Self> {
        match std::env::var("ARGUS_CAMERA_ORIENTATION")
            .unwrap_or_else(|_| DEFAULT_CAMERA_ORIENTATION.to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "none" => Ok(Self::None),
            "rotate-90" => Ok(Self::Rotate90),
            "rotate-180" => Ok(Self::Rotate180),
            "rotate-270" => Ok(Self::Rotate270),
            "hflip" => Ok(Self::HorizontalFlip),
            "vflip" => Ok(Self::VerticalFlip),
            other => Err(anyhow!(
                "unsupported ARGUS_CAMERA_ORIENTATION `{other}`; expected none, rotate-90, rotate-180, rotate-270, hflip, or vflip"
            )),
        }
    }

    fn as_gst_method(self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Rotate90 => Some("clockwise"),
            Self::Rotate180 => Some("rotate-180"),
            Self::Rotate270 => Some("counterclockwise"),
            Self::HorizontalFlip => Some("horizontal-flip"),
            Self::VerticalFlip => Some("vertical-flip"),
        }
    }
}

#[derive(Debug)]
pub struct CameraConfig {
    pub backend: CameraBackend,
    pub pipeline_str: String,
}

pub fn resolve_camera_config() -> Result<CameraConfig> {
    configure_gstreamer_plugin_path();

    if let Ok(pipeline_str) = std::env::var("ARGUS_PIPELINE") {
        return Ok(CameraConfig {
            backend: inferred_backend(&pipeline_str),
            pipeline_str,
        });
    }

    let backend = CameraBackend::from_env()?;
    let orientation = CameraOrientation::from_env()?;
    let pipeline_str = match backend {
        CameraBackend::Libcamera => build_libcamera_pipeline(orientation),
        CameraBackend::V4l2 => {
            let device = std::env::var("ARGUS_V4L2_DEVICE")
                .unwrap_or_else(|_| DEFAULT_V4L2_DEVICE.to_string());
            validate_v4l2_device(&device)?;
            build_v4l2_pipeline(&device, orientation)
        }
    };

    Ok(CameraConfig {
        backend,
        pipeline_str,
    })
}

pub fn format_pipeline_error(
    backend: CameraBackend,
    source: Option<String>,
    error: String,
    debug_details: Option<String>,
) -> String {
    let source = source.unwrap_or_else(|| "unknown".into());
    let details = debug_details.unwrap_or_else(|| "no debug details".to_string());
    let formatted = format!("GStreamer pipeline error from {source}: {error} ({details})");

    if backend == CameraBackend::V4l2
        && source.contains("v4l2src")
        && (formatted.contains("Failed to allocate required memory")
            || formatted.contains("reason not-negotiated")
            || formatted.contains("Buffer pool activation failed"))
    {
        return format!(
            "{formatted}. On Raspberry Pi 5, /dev/video0 is often an rp1-cfe raw frontend node. \
Use ARGUS_CAMERA_BACKEND=libcamera or provide ARGUS_PIPELINE with libcamerasrc instead of raw v4l2src."
        );
    }

    formatted
}

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
    live_stream_tx: broadcast::Sender<Vec<u8>>,
    vlm_tx: mpsc::Sender<Vec<u8>>,
    last_vlm_send: Arc<Mutex<Instant>>,
) {
    // The appsink callback runs on a GStreamer streaming thread, so it must stay
    // synchronous and non-blocking. We only clone frame bytes and send.
    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |sink| {
                let sample = sink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                let buffer = sample.buffer().ok_or(gst::FlowError::Error)?;
                let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;
                let frame_bytes = map.as_slice().to_vec();

                // Errors only if there are no receivers — fine to ignore.
                let _ = live_stream_tx.send(frame_bytes.clone());

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

fn configure_gstreamer_plugin_path() {
    let plugin_dir = Path::new(DEFAULT_GST_PLUGIN_PATH);
    if !plugin_dir.exists() {
        return;
    }

    let mut plugin_paths = std::env::var_os("GST_PLUGIN_PATH").unwrap_or_default();
    let plugin_path_str = plugin_dir.as_os_str();

    if plugin_paths.is_empty() {
        plugin_paths.push(plugin_path_str);
    } else {
        let existing = std::env::split_paths(&plugin_paths).collect::<Vec<_>>();
        if existing.iter().any(|path| path == plugin_dir) {
            return;
        }

        let mut merged = existing;
        merged.push(plugin_dir.to_path_buf());
        if let Ok(joined) = std::env::join_paths(merged) {
            plugin_paths = joined;
        } else {
            return;
        }
    }

    // This happens during single-threaded startup before GStreamer is initialized.
    unsafe {
        std::env::set_var("GST_PLUGIN_PATH", plugin_paths);
    }
}

fn inferred_backend(pipeline_str: &str) -> CameraBackend {
    if pipeline_str.contains("v4l2src") {
        CameraBackend::V4l2
    } else {
        CameraBackend::Libcamera
    }
}

fn validate_v4l2_device(device: &str) -> Result<()> {
    if !Path::new(device).exists() {
        error!("Camera device {device} was not found. Check the CSI camera, overlay, and V4L2 setup.");
        return Err(anyhow!("missing camera device at {device}"));
    }

    Ok(())
}

fn build_libcamera_pipeline(orientation: CameraOrientation) -> String {
    format!(
        concat!(
            "libcamerasrc ! ",
            "video/x-raw,colorimetry=bt709,format=NV12,width=1280,height=720,framerate=30/1 ! ",
            "queue leaky=downstream max-size-buffers=2 ! ",
            "videoconvert ! ",
            "{orientation}",
            "jpegenc quality=75 ! ",
            "appsink name=sink emit-signals=true sync=false drop=true max-buffers=2"
        ),
        orientation = videoflip_segment(orientation)
    )
}

fn build_v4l2_pipeline(device: &str, orientation: CameraOrientation) -> String {
    format!(
        concat!(
            "v4l2src device={device} ! ",
            "video/x-raw,width=1280,height=720,framerate=30/1 ! ",
            "videoconvert ! ",
            "{orientation}",
            "jpegenc quality=75 ! ",
            "appsink name=sink emit-signals=true sync=false drop=true max-buffers=2"
        ),
        device = device,
        orientation = videoflip_segment(orientation)
    )
}

fn videoflip_segment(orientation: CameraOrientation) -> &'static str {
    match orientation.as_gst_method() {
        Some(method) => match method {
            "clockwise" => "videoflip method=clockwise ! ",
            "rotate-180" => "videoflip method=rotate-180 ! ",
            "counterclockwise" => "videoflip method=counterclockwise ! ",
            "horizontal-flip" => "videoflip method=horizontal-flip ! ",
            "vertical-flip" => "videoflip method=vertical-flip ! ",
            _ => unreachable!("unsupported videoflip method"),
        },
        None => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex as StdMutex, OnceLock};

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

    fn env_lock() -> &'static StdMutex<()> {
        static LOCK: OnceLock<StdMutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| StdMutex::new(()))
    }

    fn with_env<T>(vars: &[(&str, Option<&str>)], f: impl FnOnce() -> T) -> T {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let previous: Vec<_> = vars
            .iter()
            .map(|(key, _)| ((*key).to_string(), std::env::var_os(key)))
            .collect();
        for (key, value) in vars {
            match value {
                Some(value) => unsafe {
                    std::env::set_var(key, value);
                },
                None => unsafe {
                    std::env::remove_var(key);
                },
            }
        }
        let result = f();
        for (key, value) in previous {
            match value {
                Some(value) => unsafe {
                    std::env::set_var(&key, value);
                },
                None => unsafe {
                    std::env::remove_var(&key);
                },
            }
        }
        result
    }

    #[test]
    fn default_backend_is_libcamera() {
        with_env(&[("ARGUS_CAMERA_BACKEND", None)], || {
            assert_eq!(CameraBackend::from_env().unwrap(), CameraBackend::Libcamera);
        });
    }

    #[test]
    fn default_orientation_is_rotate_180() {
        with_env(&[("ARGUS_CAMERA_ORIENTATION", None)], || {
            assert_eq!(
                CameraOrientation::from_env().unwrap(),
                CameraOrientation::Rotate180
            );
        });
    }

    #[test]
    fn inferred_backend_detects_v4l2_pipeline() {
        assert_eq!(
            inferred_backend("v4l2src device=/dev/video0 ! appsink name=sink"),
            CameraBackend::V4l2
        );
    }

    #[test]
    fn v4l2_error_is_rewritten_for_pi5_guidance() {
        let message = format_pipeline_error(
            CameraBackend::V4l2,
            Some("/GstPipeline:pipeline0/GstV4l2Src:v4l2src0".into()),
            "Failed to allocate required memory".into(),
            Some("Buffer pool activation failed".into()),
        );
        assert!(message.contains("ARGUS_CAMERA_BACKEND=libcamera"));
    }

    #[test]
    fn libcamera_pipeline_includes_default_rotate_180() {
        let pipeline = build_libcamera_pipeline(CameraOrientation::Rotate180);
        assert!(pipeline.contains("videoflip method=rotate-180"));
    }

    #[test]
    fn v4l2_pipeline_omits_videoflip_when_orientation_is_none() {
        let pipeline = build_v4l2_pipeline("/dev/video0", CameraOrientation::None);
        assert!(!pipeline.contains("videoflip method="));
    }

    #[test]
    fn resolve_camera_config_uses_orientation_in_generated_pipeline() {
        with_env(
            &[
                ("ARGUS_PIPELINE", None),
                ("ARGUS_CAMERA_BACKEND", Some("libcamera")),
                ("ARGUS_CAMERA_ORIENTATION", Some("rotate-270")),
            ],
            || {
                let config = resolve_camera_config().expect("camera config should resolve");
                assert_eq!(config.backend, CameraBackend::Libcamera);
                assert!(config.pipeline_str.contains("videoflip method=counterclockwise"));
            },
        );
    }

    #[test]
    fn resolve_camera_config_applies_default_orientation() {
        with_env(
            &[
                ("ARGUS_PIPELINE", None),
                ("ARGUS_CAMERA_BACKEND", Some("libcamera")),
                ("ARGUS_CAMERA_ORIENTATION", None),
            ],
            || {
                let config = resolve_camera_config().expect("camera config should resolve");
                assert!(config.pipeline_str.contains("videoflip method=rotate-180"));
            },
        );
    }

    #[test]
    fn orientation_env_accepts_hflip() {
        with_env(&[("ARGUS_CAMERA_ORIENTATION", Some("hflip"))], || {
            assert_eq!(
                CameraOrientation::from_env().expect("orientation should parse"),
                CameraOrientation::HorizontalFlip
            );
        });
    }

    #[test]
    fn resolve_camera_config_preserves_custom_pipeline_override() {
        with_env(
            &[
                (
                    "ARGUS_PIPELINE",
                    Some("videotestsrc ! jpegenc ! appsink name=sink"),
                ),
                ("ARGUS_CAMERA_ORIENTATION", Some("rotate-180")),
            ],
            || {
                let config = resolve_camera_config().expect("camera config should resolve");
                assert_eq!(config.pipeline_str, "videotestsrc ! jpegenc ! appsink name=sink");
            },
        )
    }

    #[tokio::test(flavor = "current_thread")]
    async fn videotestsrc_produces_live_and_vlm_frames() {
        init_gstreamer();

        let pipeline = build_pipeline(TEST_PIPELINE_STR).expect("failed to build test pipeline");
        let appsink = pipeline_appsink(&pipeline).expect("missing test appsink");

        let (live_stream_tx, mut live_stream_rx) = broadcast::channel::<Vec<u8>>(2);
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
