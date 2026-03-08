//! End-to-end test: real Gemini API call with images from tests/test_images/.
//!
//! Requires GEMINI_API_KEY (loaded from .env). Run with:
//!   cargo test --test process_images

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use argus::frame::Frame;
use argus::gemini::{GeminiClient, ReqwestGeminiClient};

fn load_api_key() -> Option<String> {
    dotenvy::dotenv().ok();
    std::env::var("GEMINI_API_KEY").ok()
}

fn load_test_images() -> Vec<(String, Vec<u8>)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/test_images");
    let mut images = vec![];
    if !dir.exists() {
        return images;
    }
    for entry in fs::read_dir(&dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if ext == "jpg" || ext == "jpeg" {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let bytes = fs::read(&path).unwrap();
            images.push((name, bytes));
        }
    }
    images.sort_by(|a, b| a.0.cmp(&b.0));
    images
}

#[tokio::test]
async fn process_images_end_to_end() {
    let api_key = match load_api_key() {
        Some(key) => key,
        None => {
            eprintln!("GEMINI_API_KEY not set — skipping process_images test");
            return;
        }
    };

    let images = load_test_images();
    assert!(!images.is_empty(), "no images found in tests/test_images/");

    let client = ReqwestGeminiClient::new(api_key);

    for (name, bytes) in images {
        println!("Testing image: {name}");
        let frame = Frame::new(&name, bytes)
            .unwrap_or_else(|e| panic!("Frame::new failed for {name}: {e}"));
        let analysis = client
            .analyze_frame(&frame, "Entities: (none)\nRelations: (none)")
            .await
            .unwrap_or_else(|e| panic!("analyze_frame failed for {name}: {e}"));

        assert!(!analysis.raw_text.is_empty(), "{name}: raw_text is empty");
        assert!(!analysis.entities.is_empty(), "{name}: no entities returned");

        for entity in &analysis.entities {
            assert!(!entity.label.is_empty(), "{name}: entity has empty label");
            assert!(
                ["object", "person", "furniture", "vehicle", "animal"]
                    .contains(&entity.category.as_str()),
                "{name}: unknown category '{}'",
                entity.category
            );
        }

        let labels: HashSet<&str> =
            analysis.entities.iter().map(|e| e.label.as_str()).collect();
        for rel in &analysis.relations {
            assert!(
                labels.contains(rel.subject.as_str()),
                "{name}: relation subject '{}' not in entities",
                rel.subject
            );
            assert!(
                labels.contains(rel.object.as_str()),
                "{name}: relation object '{}' not in entities",
                rel.object
            );
        }

        println!(
            "  entities: {:?}",
            analysis.entities.iter().map(|e| &e.label).collect::<Vec<_>>()
        );
    }
}
