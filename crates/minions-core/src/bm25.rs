//! Pure-Rust Okapi BM25 retriever for selecting top-k relevant document chunks.

use std::collections::{HashMap, HashSet};

/// A retrieved chunk paired with its original index and BM25 score.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredChunk {
    /// Zero-based index in the input chunk slice.
    pub index: usize,
    /// BM25 relevance score.
    pub score: f64,
    /// Chunk text.
    pub chunk: String,
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|tok| !tok.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Retrieves the top-`k` chunks for `query` using Okapi BM25 (`k1 = 1.5`, `b = 0.75`).
#[must_use]
pub fn bm25_retrieve_top_k_chunks(chunks: &[String], query: &str, k: usize) -> Vec<ScoredChunk> {
    if chunks.is_empty() || k == 0 {
        return Vec::new();
    }

    let k1 = 1.5_f64;
    let b = 0.75_f64;

    let tokenized_docs: Vec<Vec<String>> = chunks.iter().map(|c| tokenize(c)).collect();
    let n = tokenized_docs.len() as f64;
    let avg_dl = tokenized_docs.iter().map(Vec::len).sum::<usize>() as f64 / n;
    let avg_dl = if avg_dl == 0.0 { 1.0 } else { avg_dl };

    let mut df: HashMap<String, usize> = HashMap::new();
    let mut tf_docs: Vec<HashMap<String, usize>> = Vec::with_capacity(tokenized_docs.len());

    for doc in &tokenized_docs {
        let mut tf: HashMap<String, usize> = HashMap::new();
        let mut seen: HashSet<&str> = HashSet::new();
        for tok in doc {
            *tf.entry(tok.clone()).or_insert(0) += 1;
            if seen.insert(tok.as_str()) {
                *df.entry(tok.clone()).or_insert(0) += 1;
            }
        }
        tf_docs.push(tf);
    }

    let query_terms = tokenize(query);
    let mut scored: Vec<ScoredChunk> = tokenized_docs
        .iter()
        .enumerate()
        .map(|(idx, doc)| {
            let dl = doc.len() as f64;
            let tf_map = &tf_docs[idx];
            let mut score = 0.0;
            for term in &query_terms {
                let Some(&tf_val) = tf_map.get(term) else {
                    continue;
                };
                let doc_freq = *df.get(term).unwrap_or(&0) as f64;
                let idf = ((n - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln();
                let tf = tf_val as f64;
                let denom = tf + k1 * (1.0 - b + b * (dl / avg_dl));
                score += idf * (tf * (k1 + 1.0)) / denom;
            }
            ScoredChunk {
                index: idx,
                score,
                chunk: chunks[idx].clone(),
            }
        })
        .collect();

    scored.sort_by(|a, b_item| {
        b_item
            .score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.index.cmp(&b_item.index))
    });
    scored.truncate(k.min(scored.len()));
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bm25_ranks_matching_chunk_first() {
        let chunks = vec![
            "Quarterly revenue grew 12% due to cloud subscriptions.".to_string(),
            "Dental hygiene appointments increased in Q3.".to_string(),
            "Cloud infrastructure costs dropped after local SLM offloading.".to_string(),
        ];
        let top = bm25_retrieve_top_k_chunks(&chunks, "local SLM cloud costs", 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].index, 2);
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn verify_bm25_empty_or_zero_k_safe() {
        let k: usize = kani::any();
        let empty: [String; 0] = [];
        assert!(bm25_retrieve_top_k_chunks(&empty, "query", k).is_empty());

        let one = [String::new()];
        assert!(bm25_retrieve_top_k_chunks(&one, "query", 0).is_empty());
    }
}
