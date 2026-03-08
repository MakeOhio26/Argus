use std::path::PathBuf;

use tracing::warn;

use crate::error::Result;
use crate::frame::Frame;
use crate::frame_store::FrameStore;
use crate::gemini::GeminiClient;
use crate::graph::SpatialGraph;

/// The AI memory system.
///
/// Owns the Gemini client, the spatial graph, and the frame store.
/// The VLM task in `main.rs` takes exclusive ownership of this struct and
/// calls `process_frame` once per VLM-rate frame (~1 FPS) — so no locking
/// is needed.
///
/// Python analogy:
/// ```python
/// class MemorySystem:
///     def __init__(self, client, frame_store):
///         self.client = client
///         self.graph = SpatialGraph()
///         self.frame_store = frame_store
///
///     async def process_frame(self, frame): ...
/// ```
pub struct MemorySystem {
    client: Box<dyn GeminiClient>,
    graph: SpatialGraph,
    frame_store: FrameStore,
    graph_path: Option<PathBuf>,
}

impl MemorySystem {
    pub fn new(
        client: Box<dyn GeminiClient>,
        frame_store: FrameStore,
        graph_path: Option<PathBuf>,
    ) -> Self {
        let graph = graph_path
            .as_deref()
            .filter(|p| p.exists())
            .and_then(|p| match SpatialGraph::load(p) {
                Ok(g) => {
                    tracing::info!(
                        "Loaded graph from {} ({} entities, {} relations)",
                        p.display(),
                        g.entity_count(),
                        g.relation_count()
                    );
                    Some(g)
                }
                Err(e) => {
                    warn!("Could not load graph from {}: {e}", p.display());
                    None
                }
            })
            .unwrap_or_else(SpatialGraph::new);

        MemorySystem {
            client,
            graph,
            frame_store,
            graph_path,
        }
    }

    /// Process one frame through the full pipeline:
    ///
    /// 1. Serialize the current graph state into a text context string.
    /// 2. Send the frame + context to Gemini.
    /// 3. Upsert all returned entities into the graph.
    /// 4. Attempt to store the frame for each entity (novel-perspective only).
    /// 5. Upsert all returned relations (invalid ones are warned and skipped).
    ///
    /// Rust concept: `&mut self` means this method has exclusive mutable access
    /// to the struct — equivalent to Python's `self` but the borrow checker
    /// ensures nothing else can read or write `MemorySystem` at the same time.
    pub async fn process_frame(&mut self, frame: Frame) -> Result<()> {
        // Step 1 — snapshot current graph as context for the prompt.
        let graph_context = self.graph.to_context_string();

        // Step 2 — call Gemini (real or mock).
        let analysis = self.client.analyze_frame(&frame, &graph_context).await?;

        // Step 3 — upsert entities.
        for entity in &analysis.entities {
            self.graph
                .upsert_entity(&entity.label, &entity.category, entity.confidence);
        }

        // Step 4 — save frame for each entity (skips if not novel).
        for entity in &analysis.entities {
            if let Err(e) =
                self.frame_store
                    .save_if_novel(&entity.label, &frame.id, &frame.jpeg_bytes)
            {
                warn!("Failed to store frame for entity '{}': {e}", entity.label);
            }
        }

        // Step 5 — upsert relations. Warn + skip rather than hard-failing so a
        // single hallucinated entity label doesn't abort the whole update.
        for relation in &analysis.relations {
            if let Err(e) =
                self.graph
                    .upsert_relation(&relation.subject, &relation.relation, &relation.object)
            {
                warn!(
                    "Skipping invalid relation {:?} --{}--> {:?}: {e}",
                    relation.subject, relation.relation, relation.object
                );
            }
        }

        // Step 6 — persist updated graph to disk (if a path was configured).
        if let Some(path) = &self.graph_path {
            if let Err(e) = self.graph.save(path) {
                warn!("Failed to save graph to {}: {e}", path.display());
            }
        }

        Ok(())
    }

    /// Query the graph using an arbitrary closure.
    ///
    /// Rust concept: the closure `f` receives a shared (`&`) reference to the
    /// graph and can return any value `T`.  The caller doesn't get raw access
    /// to the graph — all mutations go through `process_frame`.
    ///
    /// Python analogy:
    /// ```python
    /// def query(self, f):
    ///     return f(self.graph)
    /// ```
    pub fn query<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&SpatialGraph) -> T,
    {
        f(&self.graph)
    }

    /// Serialize the current graph state to a compact JSON string for WebSocket transmission.
    pub fn graph_json(&self) -> crate::error::Result<String> {
        self.graph.to_json()
    }

    /// Return all stored novel frames for `entity_label`, oldest-first.
    pub fn get_entity_frames(&self, entity_label: &str) -> Result<Vec<Vec<u8>>> {
        self.frame_store.load_all(entity_label)
    }

    /// Return the most recently stored frame for `entity_label`, or `None`.
    pub fn get_entity_latest_frame(&self, entity_label: &str) -> Result<Option<Vec<u8>>> {
        self.frame_store.load_latest(entity_label)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ArgusError;
    use crate::frame::Frame;
    use crate::frame_store::FrameStore;
    use crate::gemini::{Entity, MockGeminiClient, SceneAnalysis, SpatialRelation};
    use tempfile::TempDir;

    fn make_frame(id: &str) -> Frame {
        // Minimal valid JPEG bytes for test frames.
        let img = image::RgbImage::from_pixel(1, 1, image::Rgb([200u8, 100, 50]));
        let mut buf = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut buf),
            image::ImageFormat::Jpeg,
        )
        .unwrap();
        Frame::new(id, buf).unwrap()
    }

    fn make_scene_analysis() -> SceneAnalysis {
        SceneAnalysis {
            entities: vec![
                Entity {
                    label: "coffee_cup".into(),
                    category: "object".into(),
                    confidence: 0.9,
                },
                Entity {
                    label: "desk".into(),
                    category: "furniture".into(),
                    confidence: 0.95,
                },
            ],
            relations: vec![SpatialRelation {
                subject: "coffee_cup".into(),
                relation: "on".into(),
                object: "desk".into(),
            }],
            raw_text: "{}".into(),
        }
    }

    #[tokio::test]
    async fn process_frame_updates_entity_and_relation_counts() {
        let dir = TempDir::new().unwrap();
        let mut mock = MockGeminiClient::new();
        mock.expect_analyze_frame()
            .times(1)
            .returning(|_, _| Ok(make_scene_analysis()));

        let store = FrameStore::new(dir.path(), 10).unwrap();
        let mut memory = MemorySystem::new(Box::new(mock), store, None);
        memory.process_frame(make_frame("f1")).await.unwrap();

        assert_eq!(memory.query(|g| g.entity_count()), 2);
        assert_eq!(memory.query(|g| g.relation_count()), 1);
    }

    #[tokio::test]
    async fn two_frames_confidence_updated() {
        let dir = TempDir::new().unwrap();
        let mut mock = MockGeminiClient::new();
        mock.expect_analyze_frame()
            .times(2)
            .returning(|_, _| Ok(make_scene_analysis()));

        let store = FrameStore::new(dir.path(), 10).unwrap();
        let mut memory = MemorySystem::new(Box::new(mock), store, None);
        memory.process_frame(make_frame("f1")).await.unwrap();
        memory.process_frame(make_frame("f2")).await.unwrap();

        // Two frames, same entities → still 2 nodes (not 4).
        assert_eq!(memory.query(|g| g.entity_count()), 2);
    }

    #[tokio::test]
    async fn frame_stored_after_process_frame() {
        let dir = TempDir::new().unwrap();
        let mut mock = MockGeminiClient::new();
        mock.expect_analyze_frame()
            .times(1)
            .returning(|_, _| Ok(make_scene_analysis()));

        let store = FrameStore::new(dir.path(), 10).unwrap();
        let mut memory = MemorySystem::new(Box::new(mock), store, None);
        memory.process_frame(make_frame("f1")).await.unwrap();

        let latest = memory.get_entity_latest_frame("coffee_cup").unwrap();
        assert!(latest.is_some(), "frame should have been stored");
        assert!(!latest.unwrap().is_empty());
    }

    #[tokio::test]
    async fn gemini_error_propagates() {
        let dir = TempDir::new().unwrap();
        let mut mock = MockGeminiClient::new();
        mock.expect_analyze_frame()
            .returning(|_, _| Err(ArgusError::GeminiApi { status: 429, body: "rate limited".into() }));

        let store = FrameStore::new(dir.path(), 10).unwrap();
        let mut memory = MemorySystem::new(Box::new(mock), store, None);
        let result = memory.process_frame(make_frame("f1")).await;
        assert!(matches!(result, Err(ArgusError::GeminiApi { status: 429, .. })));
    }

    #[tokio::test]
    async fn invalid_relation_is_skipped_not_hard_failure() {
        let dir = TempDir::new().unwrap();
        let mut mock = MockGeminiClient::new();
        mock.expect_analyze_frame()
            .times(1)
            .returning(|_, _| {
                Ok(SceneAnalysis {
                    entities: vec![Entity {
                        label: "cup".into(),
                        category: "object".into(),
                        confidence: 0.9,
                    }],
                    relations: vec![SpatialRelation {
                        subject: "cup".into(),
                        relation: "on".into(),
                        // "ghost" does not exist in the graph — should be skipped
                        object: "ghost".into(),
                    }],
                    raw_text: "{}".into(),
                })
            });

        let store = FrameStore::new(dir.path(), 10).unwrap();
        let mut memory = MemorySystem::new(Box::new(mock), store, None);
        // Should succeed despite the invalid relation.
        memory.process_frame(make_frame("f1")).await.unwrap();
        assert_eq!(memory.query(|g| g.entity_count()), 1);
        assert_eq!(memory.query(|g| g.relation_count()), 0);
    }
}
