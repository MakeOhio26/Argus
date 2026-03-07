use thiserror::Error;

#[derive(Debug, Error)]
pub enum ArgusError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Gemini API error (status {status}): {body}")]
    GeminiApi { status: u16, body: String },

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid frame: {0}")]
    InvalidFrame(String),

    #[error("Malformed response: {0}")]
    MalformedResponse(String),

    #[error("Graph error: {0}")]
    Graph(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),
}

/// Type alias so all modules can write `Result<T>` instead of `Result<T, ArgusError>`.
///
/// Python analogy: there's no direct equivalent, but think of it as a typed
/// return annotation shorthand that every function in this crate uses.
pub type Result<T> = std::result::Result<T, ArgusError>;
