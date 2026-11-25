//! Resilient JSON extraction from LLM responses wrapped in markdown fences.

use regex::Regex;
use serde::de::DeserializeOwned;

use crate::MinionError;

/// Extracts and deserializes a JSON object from `text`, handling markdown code
/// blocks (```` ```json ... ``` ````) and surrounding prose.
///
/// # Errors
/// Returns [`MinionError::Protocol`] if no valid JSON object can be extracted.
pub fn extract_json<T: DeserializeOwned>(text: &str) -> Result<T, MinionError> {
    let fence_re =
        Regex::new(r"(?s)```(?:json)?\s*(\{.*?\})\s*```").expect("static fence regex is valid");

    let candidate = if let Some(last_cap) = fence_re.captures_iter(text).last() {
        last_cap
            .get(1)
            .map_or(text, |m| m.as_str())
            .trim()
            .to_string()
    } else if let (Some(start), Some(end)) = (text.find('{'), text.rfind('}')) {
        if start <= end {
            text[start..=end].trim().to_string()
        } else {
            text.trim().to_string()
        }
    } else {
        text.trim().to_string()
    };

    serde_json::from_str::<T>(&candidate).map_err(|err| {
        MinionError::Protocol(format!("failed to parse JSON from model output: {err}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Decision {
        decision: String,
    }

    #[test]
    fn extracts_fenced_json() {
        let raw =
            "Sure, here is the response:\n```json\n{\"decision\": \"provide_final_answer\"}\n```";
        let parsed: Decision = extract_json(raw).expect("valid fenced JSON");
        assert_eq!(parsed.decision, "provide_final_answer");
    }

    #[test]
    fn extracts_unfenced_json_with_prose() {
        let raw = "Thinking... {\"decision\": \"request_additional_info\"} done.";
        let parsed: Decision = extract_json(raw).expect("valid inline JSON");
        assert_eq!(parsed.decision, "request_additional_info");
    }
}
