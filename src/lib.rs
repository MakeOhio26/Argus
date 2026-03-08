//! Argus — AI spatial memory module.
//!
//! This library crate exposes the memory pipeline for use by integration tests
//! and future callers.  The GStreamer camera pipeline lives only in the binary
//! (`src/main.rs`) and is not exported here.
//!
//! # Quick start
//! ```rust,no_run
//! use argus::frame::Frame;
//! use argus::frame_store::FrameStore;
//! use argus::gemini::ReqwestGeminiClient;
//! use argus::memory::MemorySystem;
//!
//! # async fn example() -> argus::error::Result<()> {
//! let client = Box::new(ReqwestGeminiClient::new("your-api-key"));
//! let store  = std::sync::Arc::new(FrameStore::new("./argus_store", 10)?);
//! let mut memory = MemorySystem::new(client, store, Some(std::path::PathBuf::from("./argus_graph.json")));
//!
//! let frame = Frame::new("frame_001", vec![0xFF, 0xD8, 0xFF])?;
//! memory.process_frame(frame).await?;
//! # Ok(())
//! # }
//! ```

pub mod error;
pub mod frame;
pub mod frame_store;
pub mod gemini;
pub mod graph;
pub mod memory;
pub mod voc;
