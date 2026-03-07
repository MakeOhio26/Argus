use chrono::{DateTime, Utc};

use crate::error::{ArgusError, Result};

/// A single captured frame from the vision pipeline.
///
/// Python analogy:
/// ```python
/// @dataclass
/// class Frame:
///     id: str
///     jpeg_bytes: bytes
///     captured_at: datetime
/// ```
///
/// Frames always arrive as raw JPEG bytes from GStreamer — base64 encoding
/// is an internal detail handled inside `gemini.rs` before sending to the API.
#[derive(Debug)]
pub struct Frame {
    pub id: String,
    pub jpeg_bytes: Vec<u8>,
    pub captured_at: DateTime<Utc>,
}

impl Frame {
    /// Create a new Frame, validating that `jpeg_bytes` is non-empty.
    ///
    /// Returns `Err(ArgusError::InvalidFrame)` for empty byte slices.
    ///
    /// Python analogy: `__post_init__` validation in a dataclass.
    pub fn new(id: impl Into<String>, jpeg_bytes: Vec<u8>) -> Result<Self> {
        if jpeg_bytes.is_empty() {
            return Err(ArgusError::InvalidFrame(
                "jpeg_bytes cannot be empty".into(),
            ));
        }
        Ok(Self {
            id: id.into(),
            jpeg_bytes,
            captured_at: Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_frame_creation() {
        let frame = Frame::new("frame_001", vec![0xFF, 0xD8, 0xFF]).unwrap();
        assert_eq!(frame.id, "frame_001");
        assert_eq!(frame.jpeg_bytes.len(), 3);
    }

    #[test]
    fn empty_bytes_returns_error() {
        let result = Frame::new("bad", vec![]);
        assert!(result.is_err());
        match result.unwrap_err() {
            ArgusError::InvalidFrame(msg) => assert!(msg.contains("empty")),
            other => panic!("expected InvalidFrame, got {other:?}"),
        }
    }
}
