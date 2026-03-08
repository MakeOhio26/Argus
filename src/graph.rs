use std::collections::HashMap;
use std::path::Path;

use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use serde::{Deserialize, Serialize};

use crate::error::{ArgusError, Result};

/// An entity observed in the scene — one node in the spatial graph.
///
/// Python analogy: a simple `@dataclass` with three fields.
/// No temporal tracking here — the `FrameStore` handles all frame history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityNode {
    pub label: String,    // unique identifier, e.g. "coffee_cup"
    pub category: String, // "object" | "person" | "furniture" | "vehicle" | "animal"
    pub confidence: f32,  // latest value from Gemini (0.0 – 1.0)
    #[serde(default)]
    pub crossed_out: bool, // true = dismissed from the suspicions list
}

/// A directed spatial relationship between two entities — one edge in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationEdge {
    pub relation: String, // "on" | "near" | "above" | "below" | "holds" | "faces"
}

#[derive(Serialize, Deserialize)]
struct GraphSnapshot {
    entities: Vec<EntityNode>,
    relations: Vec<SnapshotRelation>,
}

#[derive(Serialize, Deserialize)]
struct SnapshotRelation {
    subject: String,
    relation: String,
    object: String,
}

/// Entity shape sent over WebSocket — includes derived `rank`.
#[derive(Serialize)]
struct WsEntitySnapshot {
    rank: u32,
    label: String,
    category: String,
    confidence: f32,
    crossed_out: bool,
}

#[derive(Serialize)]
struct WsGraphSnapshot {
    entities: Vec<WsEntitySnapshot>,
    relations: Vec<SnapshotRelation>,
}

/// The spatial semantic graph.
///
/// Python analogy: `networkx.DiGraph` with attribute dicts on nodes and edges.
///
/// Uses `StableDiGraph` (not `DiGraph`) so that node indices remain valid even
/// after other nodes are removed — this keeps `label_to_node` consistent.
pub struct SpatialGraph {
    graph: StableDiGraph<EntityNode, RelationEdge>,
    /// Maps entity label → NodeIndex for O(1) lookup.
    /// Python analogy: a `dict[str, node_id]`.
    label_to_node: HashMap<String, NodeIndex>,
}

impl SpatialGraph {
    pub fn new() -> Self {
        SpatialGraph {
            graph: StableDiGraph::new(),
            label_to_node: HashMap::new(),
        }
    }

    /// Insert or update an entity node.
    ///
    /// If an entity with this label already exists, its confidence is updated
    /// to the latest value. If it's new, a node is inserted.
    ///
    /// Python analogy: `graph.nodes[label].update({"confidence": ...})`
    /// or `graph.add_node(label, ...)` if new.
    pub fn upsert_entity(&mut self, label: &str, category: &str, confidence: f32) {
        if let Some(&idx) = self.label_to_node.get(label) {
            self.graph[idx].confidence = confidence;
        } else {
            let idx = self.graph.add_node(EntityNode {
                label: label.to_string(),
                category: category.to_string(),
                confidence,
                crossed_out: false,
            });
            self.label_to_node.insert(label.to_string(), idx);
        }
    }

    /// Insert or update a directed edge: subject --[relation]--> object.
    ///
    /// Returns `Err(Graph)` if either entity label is not yet in the graph.
    /// (The caller is responsible for upserting entities before relations.)
    ///
    /// Python analogy:
    /// ```python
    /// if graph.has_edge(subject, object):
    ///     graph[subject][object]["relation"] = relation
    /// else:
    ///     graph.add_edge(subject, object, relation=relation)
    /// ```
    pub fn upsert_relation(&mut self, subject: &str, relation: &str, object: &str) -> Result<()> {
        let subject_idx = self.label_to_node.get(subject).copied().ok_or_else(|| {
            ArgusError::Graph(format!("unknown entity: {subject}"))
        })?;
        let object_idx = self.label_to_node.get(object).copied().ok_or_else(|| {
            ArgusError::Graph(format!("unknown entity: {object}"))
        })?;

        if let Some(edge_idx) = self.graph.find_edge(subject_idx, object_idx) {
            self.graph[edge_idx].relation = relation.to_string();
        } else {
            self.graph.add_edge(
                subject_idx,
                object_idx,
                RelationEdge {
                    relation: relation.to_string(),
                },
            );
        }
        Ok(())
    }

    /// Serialize the current graph state into human-readable text for the Gemini prompt.
    ///
    /// Example output:
    /// ```text
    /// Entities in memory: coffee_cup (object, 0.95), desk (furniture, 0.98)
    /// Relations: coffee_cup --on--> desk
    /// ```
    ///
    /// An empty graph produces:
    /// ```text
    /// Entities in memory: (none)
    /// Relations: (none)
    /// ```
    pub fn to_context_string(&self) -> String {
        let entity_list: Vec<String> = self
            .graph
            .node_indices()
            .filter(|&idx| !self.graph[idx].crossed_out)
            .map(|idx| {
                let n = &self.graph[idx];
                format!("{} ({}, {:.2})", n.label, n.category, n.confidence)
            })
            .collect();

        let relation_list: Vec<String> = self
            .graph
            .edge_indices()
            .filter(|&eidx| {
                let (src, dst) = self.graph.edge_endpoints(eidx).unwrap();
                !self.graph[src].crossed_out && !self.graph[dst].crossed_out
            })
            .map(|eidx| {
                let (src, dst) = self.graph.edge_endpoints(eidx).unwrap();
                let src_label = &self.graph[src].label;
                let dst_label = &self.graph[dst].label;
                let rel = &self.graph[eidx].relation;
                format!("{src_label} --{rel}--> {dst_label}")
            })
            .collect();

        let entities_str = if entity_list.is_empty() {
            "(none)".to_string()
        } else {
            entity_list.join(", ")
        };

        let relations_str = if relation_list.is_empty() {
            "(none)".to_string()
        } else {
            relation_list.join(", ")
        };

        format!("Entities in memory: {entities_str}\nRelations: {relations_str}")
    }

    pub fn entity_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn relation_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Returns true if an entity with the given label exists in the graph.
    pub fn contains_entity(&self, label: &str) -> bool {
        self.label_to_node.contains_key(label)
    }

    /// Persist the graph to a JSON file at `path`.
    pub fn save(&self, path: &Path) -> Result<()> {
        let entities: Vec<EntityNode> = self
            .graph
            .node_indices()
            .map(|idx| self.graph[idx].clone())
            .collect();

        let relations: Vec<SnapshotRelation> = self
            .graph
            .edge_indices()
            .map(|eidx| {
                let (src, dst) = self.graph.edge_endpoints(eidx).unwrap();
                SnapshotRelation {
                    subject: self.graph[src].label.clone(),
                    relation: self.graph[eidx].relation.clone(),
                    object: self.graph[dst].label.clone(),
                }
            })
            .collect();

        let json = serde_json::to_string_pretty(&GraphSnapshot { entities, relations })?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Serialize the graph to a compact JSON string for WebSocket transmission.
    ///
    /// Entities are sorted descending by confidence; `rank` is 1-based (1 = highest confidence).
    pub fn to_json(&self) -> Result<String> {
        let mut entities: Vec<WsEntitySnapshot> = self
            .graph
            .node_indices()
            .map(|idx| {
                let n = &self.graph[idx];
                WsEntitySnapshot {
                    rank: 0,
                    label: n.label.clone(),
                    category: n.category.clone(),
                    confidence: n.confidence,
                    crossed_out: n.crossed_out,
                }
            })
            .collect();

        entities.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, e) in entities.iter_mut().enumerate() {
            e.rank = (i + 1) as u32;
        }

        let relations: Vec<SnapshotRelation> = self
            .graph
            .edge_indices()
            .map(|eidx| {
                let (src, dst) = self.graph.edge_endpoints(eidx).unwrap();
                SnapshotRelation {
                    subject: self.graph[src].label.clone(),
                    relation: self.graph[eidx].relation.clone(),
                    object: self.graph[dst].label.clone(),
                }
            })
            .collect();

        Ok(serde_json::to_string(&WsGraphSnapshot { entities, relations })?)
    }

    /// Load a graph from a JSON file previously written by [`SpatialGraph::save`].
    pub fn load(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let snapshot: GraphSnapshot = serde_json::from_str(&json)?;
        let mut graph = SpatialGraph::new();
        for entity in snapshot.entities {
            graph.upsert_entity(&entity.label, &entity.category, entity.confidence);
        }
        for rel in snapshot.relations {
            if let Err(e) = graph.upsert_relation(&rel.subject, &rel.relation, &rel.object) {
                tracing::warn!("Skipping relation on load: {e}");
            }
        }
        Ok(graph)
    }
}

impl Default for SpatialGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_new_entity() {
        let mut g = SpatialGraph::new();
        g.upsert_entity("cup", "object", 0.9);
        assert_eq!(g.entity_count(), 1);
        assert!(g.contains_entity("cup"));
    }

    #[test]
    fn upsert_same_entity_updates_confidence() {
        let mut g = SpatialGraph::new();
        g.upsert_entity("cup", "object", 0.8);
        g.upsert_entity("cup", "object", 0.95);
        // Still 1 node, confidence updated to latest
        assert_eq!(g.entity_count(), 1);
        let idx = g.label_to_node["cup"];
        assert!((g.graph[idx].confidence - 0.95).abs() < 1e-5);
    }

    #[test]
    fn upsert_relation_between_known_entities() {
        let mut g = SpatialGraph::new();
        g.upsert_entity("cup", "object", 0.9);
        g.upsert_entity("desk", "furniture", 0.98);
        g.upsert_relation("cup", "on", "desk").unwrap();
        assert_eq!(g.relation_count(), 1);
    }

    #[test]
    fn upsert_relation_unknown_entity_returns_error() {
        let mut g = SpatialGraph::new();
        g.upsert_entity("cup", "object", 0.9);
        let result = g.upsert_relation("cup", "on", "ghost");
        assert!(result.is_err());
        match result.unwrap_err() {
            ArgusError::Graph(msg) => assert!(msg.contains("ghost")),
            other => panic!("expected Graph error, got {other:?}"),
        }
    }

    #[test]
    fn to_context_string_empty_graph() {
        let g = SpatialGraph::new();
        let ctx = g.to_context_string();
        assert!(ctx.contains("(none)"));
    }

    #[test]
    fn to_context_string_with_entities_and_relations() {
        let mut g = SpatialGraph::new();
        g.upsert_entity("cup", "object", 0.9);
        g.upsert_entity("desk", "furniture", 0.95);
        g.upsert_relation("cup", "on", "desk").unwrap();
        let ctx = g.to_context_string();
        assert!(ctx.contains("cup"));
        assert!(ctx.contains("desk"));
        assert!(ctx.contains("--on-->"));
    }
}
