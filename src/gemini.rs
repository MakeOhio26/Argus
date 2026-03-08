use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};

use crate::error::{ArgusError, Result};
use crate::frame::Frame;

// ── Gemini REST API request structs ──────────────────────────────────────────
//
// These mirror the JSON shape expected by:
//   POST /v1beta/models/{model}:generateContent?key={api_key}
//
// Python analogy: TypedDict / dataclasses used with requests.post(json=...).

#[derive(Debug, Serialize)]
pub struct GeminiRequest {
    pub contents: Vec<GeminiContent>,
}

#[derive(Debug, Serialize)]
pub struct GeminiContent {
    pub role: String,
    pub parts: Vec<GeminiPart>,
}

/// A part can be either plain text or an inline base64-encoded image.
///
/// `#[serde(untagged)]` means Serde serializes each variant as its inner
/// fields directly — no wrapper key like `"type": "Text"`.  This matches
/// the Gemini API's actual wire format.
///
/// Python analogy: a Union[TextPart, ImagePart] where the JSON discriminator
/// is implicit (presence of "text" vs "inline_data" key).
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum GeminiPart {
    Text { text: String },
    InlineData { inline_data: InlineData },
}

#[derive(Debug, Serialize)]
pub struct InlineData {
    pub mime_type: String,
    pub data: String, // base64-encoded image bytes
}

// ── Gemini REST API response structs ─────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct GeminiResponse {
    pub candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiCandidate {
    pub content: GeminiResponseContent,
    #[serde(rename = "finishReason")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiResponseContent {
    pub parts: Vec<GeminiResponsePart>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiResponsePart {
    pub text: String,
}

// ── Domain types returned to callers ─────────────────────────────────────────

/// An entity extracted from the Gemini response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub label: String,    // snake_case unique identifier, e.g. "coffee_cup"
    pub category: String, // "object" | "person" | "furniture" | "vehicle" | "animal"
}

/// A spatial relationship between two entities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialRelation {
    pub subject: String, // entity label
    pub relation: String, // "near" | "on" | "above" | "below" | "holds" | "faces"
    pub object: String,  // entity label
}

/// Parsed output of one Gemini `analyze_frame` call.
#[derive(Debug, Clone)]
pub struct SceneAnalysis {
    pub entities: Vec<Entity>,
    pub relations: Vec<SpatialRelation>,
    /// Raw response text, kept for debugging.
    pub raw_text: String,
}

// ── GeminiClient trait (mockable interface) ───────────────────────────────────
//
// Separating the trait from the implementation lets tests swap in a mock
// without making any real HTTP requests.
//
// Python analogy: Protocol class.  Any type that implements this trait can be
// used wherever `dyn GeminiClient` is expected — compile-time duck typing.
//
// `#[cfg_attr(test, mockall::automock)]` generates `MockGeminiClient`
// automatically in test builds — like Python's `unittest.mock.MagicMock`.

#[cfg_attr(test, mockall::automock)]
#[async_trait]
pub trait GeminiClient: Send + Sync {
    /// Analyse one frame given the current graph context.
    ///
    /// `graph_context` is the output of `SpatialGraph::to_context_string()` and
    /// is embedded in the prompt so Gemini knows what's already in memory.
    async fn analyze_frame(&self, frame: &Frame, graph_context: &str) -> Result<SceneAnalysis>;
}

// ── Stub implementation (for local testing without a real API key) ────────────

/// A no-op [`GeminiClient`] that always returns an empty [`SceneAnalysis`].
///
/// Useful for testing the vision feed pipeline end-to-end without hitting the
/// real Gemini API.  Swap this in for [`ReqwestGeminiClient`] in `main.rs`.
pub struct StubGeminiClient;

#[async_trait]
impl GeminiClient for StubGeminiClient {
    async fn analyze_frame(&self, _frame: &Frame, _graph_context: &str) -> Result<SceneAnalysis> {
        Ok(SceneAnalysis {
            entities: vec![],
            relations: vec![],
            raw_text: "(stub)".into(),
        })
    }
}

// ── Real reqwest-based implementation ────────────────────────────────────────

pub struct ReqwestGeminiClient {
    api_key: String,
    /// Base URL — overridable to point at a wiremock server during tests.
    base_url: String,
    model: String,
    http: reqwest::Client,
}

impl ReqwestGeminiClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://generativelanguage.googleapis.com".into(),
            model: "gemini-2.5-flash".into(),
            http: reqwest::Client::new(),
        }
    }

    /// Override the base URL — used in integration tests to target wiremock.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    fn build_prompt(graph_context: &str) -> String {
        format!(
            r#"You are maintaining a spatial memory graph of the scene.

Current graph state:
{graph_context}

Analyze the new frame. Return ONLY a JSON object for entities/relations to ADD or UPDATE:
{{
  "entities": [{{"label": "<snake_case_name>", "category": "<object|person|furniture|vehicle|animal>"}}],
  "relations": [{{"subject": "<entity_label>", "relation": "<near|on|above|below|holds|faces>", "object": "<entity_label>"}}]
}}
No markdown fences, no explanation. Only valid JSON."#
        )
    }
}

#[async_trait]
impl GeminiClient for ReqwestGeminiClient {
    async fn analyze_frame(&self, frame: &Frame, graph_context: &str) -> Result<SceneAnalysis> {
        let url = format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.base_url, self.model, self.api_key
        );

        let body = GeminiRequest {
            contents: vec![GeminiContent {
                role: "user".into(),
                parts: vec![
                    GeminiPart::Text {
                        text: Self::build_prompt(graph_context),
                    },
                    GeminiPart::InlineData {
                        inline_data: InlineData {
                            mime_type: "image/jpeg".into(),
                            data: STANDARD.encode(&frame.jpeg_bytes),
                        },
                    },
                ],
            }],
        };

        let response = self.http.post(&url).json(&body).send().await?;
        let status = response.status().as_u16();

        if status != 200 {
            let body_text = response.text().await.unwrap_or_default();
            return Err(ArgusError::GeminiApi {
                status,
                body: body_text,
            });
        }

        let gemini_resp: GeminiResponse = response.json().await?;
        parse_scene_analysis(gemini_resp)
    }
}

// ── Response parsing ──────────────────────────────────────────────────────────

/// Intermediate deserialization target — matches the JSON schema Gemini is
/// prompted to return.
#[derive(Debug, Deserialize)]
struct GeminiSceneJson {
    entities: Vec<Entity>,
    relations: Vec<SpatialRelation>,
}

/// Extract JSON from a response that may be wrapped in markdown fences.
///
/// Gemini sometimes wraps its output in ` ```json ... ``` ` even when told not
/// to.  This strips those fences so `serde_json` can parse the raw JSON.
fn extract_json(text: &str) -> &str {
    if let Some(start) = text.find("```json") {
        let after = &text[start + 7..];
        if let Some(end) = after.find("```") {
            return after[..end].trim();
        }
    }
    // Also handle plain ``` fences without the "json" tag.
    if let Some(start) = text.find("```") {
        let after = &text[start + 3..];
        if let Some(end) = after.find("```") {
            return after[..end].trim();
        }
    }
    text.trim()
}

fn parse_scene_analysis(response: GeminiResponse) -> Result<SceneAnalysis> {
    let raw_text = response
        .candidates
        .into_iter()
        .next()
        .and_then(|c| c.content.parts.into_iter().next())
        .map(|p| p.text)
        .ok_or_else(|| ArgusError::MalformedResponse("no candidates in response".into()))?;

    let json_str = extract_json(&raw_text);
    let parsed: GeminiSceneJson = serde_json::from_str(json_str)
        .map_err(|e| ArgusError::MalformedResponse(format!("JSON parse error: {e}")))?;

    Ok(SceneAnalysis {
        entities: parsed.entities,
        relations: parsed.relations,
        raw_text,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_response(text: &str) -> GeminiResponse {
        GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: GeminiResponseContent {
                    parts: vec![GeminiResponsePart {
                        text: text.to_string(),
                    }],
                },
                finish_reason: Some("STOP".into()),
            }],
        }
    }

    #[test]
    fn parse_clean_json_response() {
        let json = r#"{"entities":[{"label":"coffee_cup","category":"object"}],"relations":[]}"#;
        let analysis = parse_scene_analysis(make_response(json)).unwrap();
        assert_eq!(analysis.entities.len(), 1);
        assert_eq!(analysis.entities[0].label, "coffee_cup");
        assert_eq!(analysis.relations.len(), 0);
    }

    #[test]
    fn parse_markdown_fenced_response() {
        let fenced = "```json\n{\"entities\":[{\"label\":\"desk\",\"category\":\"furniture\"}],\"relations\":[]}\n```";
        let analysis = parse_scene_analysis(make_response(fenced)).unwrap();
        assert_eq!(analysis.entities[0].label, "desk");
    }

    #[test]
    fn parse_plain_fence_without_json_tag() {
        let fenced = "```\n{\"entities\":[],\"relations\":[]}\n```";
        let analysis = parse_scene_analysis(make_response(fenced)).unwrap();
        assert_eq!(analysis.entities.len(), 0);
    }

    #[test]
    fn malformed_response_returns_error() {
        let result = parse_scene_analysis(make_response("I cannot analyse this image."));
        assert!(result.is_err());
    }

    #[test]
    fn extract_json_strips_fences() {
        let s = "```json\n{\"foo\":1}\n```";
        assert_eq!(extract_json(s), "{\"foo\":1}");
    }

    #[test]
    fn extract_json_passthrough_for_clean_json() {
        let s = r#"{"foo":1}"#;
        assert_eq!(extract_json(s), s);
    }

    #[tokio::test]
    async fn reqwest_client_calls_correct_endpoint_and_parses_response() {
        use wiremock::matchers::{method, path_regex};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        let fake_body = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": r#"{"entities":[{"label":"laptop","category":"object"}],"relations":[]}"#}]
                },
                "finishReason": "STOP"
            }]
        });

        Mock::given(method("POST"))
            .and(path_regex(r".*generateContent.*"))
            .respond_with(ResponseTemplate::new(200).set_body_json(fake_body))
            .mount(&mock_server)
            .await;

        let client = ReqwestGeminiClient::new("fake-key")
            .with_base_url(mock_server.uri());

        let frame = crate::frame::Frame::new("f1", vec![0xFF, 0xD8, 0xFF]).unwrap();
        let analysis = client.analyze_frame(&frame, "Entities: (none)\nRelations: (none)").await.unwrap();

        assert_eq!(analysis.entities.len(), 1);
        assert_eq!(analysis.entities[0].label, "laptop");
    }

    #[tokio::test]
    async fn reqwest_client_returns_error_on_non_200() {
        use wiremock::matchers::{method, path_regex};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r".*generateContent.*"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&mock_server)
            .await;

        let client = ReqwestGeminiClient::new("fake-key")
            .with_base_url(mock_server.uri());
        let frame = crate::frame::Frame::new("f1", vec![0xFF, 0xD8, 0xFF]).unwrap();
        let result = client.analyze_frame(&frame, "").await;
        assert!(matches!(result, Err(ArgusError::GeminiApi { status: 429, .. })));
    }
}
