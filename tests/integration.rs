//! Integration test: full pipeline from Frame → Gemini (wiremock) → Graph + FrameStore.
//!
//! Uses `wiremock` to stand up a real local HTTP server and `tempfile` for an
//! isolated frame store directory — no real API key or camera hardware needed.

use tempfile::TempDir;
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

use argus::frame::Frame;
use argus::frame_store::FrameStore;
use argus::gemini::ReqwestGeminiClient;
use argus::memory::MemorySystem;

/// Minimal valid JPEG bytes (1×1 pixel).
fn tiny_jpeg() -> Vec<u8> {
    let img = image::RgbImage::from_pixel(1, 1, image::Rgb([200u8, 100, 50]));
    let mut buf = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut buf),
        image::ImageFormat::Jpeg,
    )
    .unwrap();
    buf
}

/// A canned Gemini response containing two entities and one relation.
fn canned_gemini_response() -> serde_json::Value {
    serde_json::json!({
        "candidates": [{
            "content": {
                "parts": [{
                    "text": r#"{"entities":[{"label":"laptop","category":"object"},{"label":"person_1","category":"person"}],"relations":[{"subject":"person_1","relation":"uses","object":"laptop"}]}"#
                }]
            },
            "finishReason": "STOP"
        }]
    })
}

#[tokio::test]
async fn full_pipeline_entities_and_relations_in_graph() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex(r".*generateContent.*"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_gemini_response()))
        .mount(&mock_server)
        .await;

    let dir = TempDir::new().unwrap();
    let client = Box::new(
        ReqwestGeminiClient::new("fake-key").with_base_url(mock_server.uri()),
    );
    let store = std::sync::Arc::new(FrameStore::new(dir.path(), 10).unwrap());
    let mut memory = MemorySystem::new(client, store, None);

    let frame = Frame::new("frame_1", tiny_jpeg()).unwrap();
    memory.process_frame(frame).await.unwrap();

    assert_eq!(memory.query(|g| g.entity_count()), 2);
    assert_eq!(memory.query(|g| g.relation_count()), 1);
}

#[tokio::test]
async fn second_frame_with_same_entities_updates_confidence() {
    let mock_server = MockServer::start().await;
    // Serve the same response for both calls.
    Mock::given(method("POST"))
        .and(path_regex(r".*generateContent.*"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_gemini_response()))
        .expect(2)
        .mount(&mock_server)
        .await;

    let dir = TempDir::new().unwrap();
    let client = Box::new(
        ReqwestGeminiClient::new("fake-key").with_base_url(mock_server.uri()),
    );
    let store = std::sync::Arc::new(FrameStore::new(dir.path(), 10).unwrap());
    let mut memory = MemorySystem::new(client, store, None);

    memory.process_frame(Frame::new("frame_1", tiny_jpeg()).unwrap()).await.unwrap();
    memory.process_frame(Frame::new("frame_2", tiny_jpeg()).unwrap()).await.unwrap();

    // Two frames, same entities → still 2 nodes (not 4).
    assert_eq!(memory.query(|g| g.entity_count()), 2);
    assert_eq!(memory.query(|g| g.relation_count()), 1);
}

#[tokio::test]
async fn frame_stored_and_retrievable_for_each_entity() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex(r".*generateContent.*"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_gemini_response()))
        .mount(&mock_server)
        .await;

    let dir = TempDir::new().unwrap();
    let client = Box::new(
        ReqwestGeminiClient::new("fake-key").with_base_url(mock_server.uri()),
    );
    let store = std::sync::Arc::new(FrameStore::new(dir.path(), 10).unwrap());
    let mut memory = MemorySystem::new(client, store, None);

    memory.process_frame(Frame::new("frame_1", tiny_jpeg()).unwrap()).await.unwrap();

    let laptop_frame = memory.get_entity_latest_frame("laptop").unwrap();
    assert!(laptop_frame.is_some(), "frame should be stored for 'laptop'");
    assert!(!laptop_frame.unwrap().is_empty());

    let person_frame = memory.get_entity_latest_frame("person_1").unwrap();
    assert!(person_frame.is_some(), "frame should be stored for 'person_1'");
}

#[tokio::test]
async fn duplicate_frames_not_stored_twice_for_entity() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex(r".*generateContent.*"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_gemini_response()))
        .expect(2)
        .mount(&mock_server)
        .await;

    let dir = TempDir::new().unwrap();
    let client = Box::new(
        ReqwestGeminiClient::new("fake-key").with_base_url(mock_server.uri()),
    );
    // max_frames=1 — only one novel frame per entity.
    let store = std::sync::Arc::new(FrameStore::new(dir.path(), 1).unwrap());
    let mut memory = MemorySystem::new(client, store, None);

    let jpeg = tiny_jpeg();
    memory.process_frame(Frame::new("frame_1", jpeg.clone()).unwrap()).await.unwrap();
    memory.process_frame(Frame::new("frame_2", jpeg).unwrap()).await.unwrap();

    // Identical JPEG → second frame rejected as not novel.
    let frames = memory.get_entity_frames("laptop").unwrap();
    assert_eq!(frames.len(), 1, "identical frame should not be stored twice");
}
