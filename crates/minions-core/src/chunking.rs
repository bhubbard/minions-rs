//! Document chunking utilities for splitting long contexts for local workers.

use regex::Regex;

use crate::MinionError;

/// Splits `doc` into fixed-size character-safe chunks with `overlap` characters.
///
/// Guarantees progress even when `overlap >= max_chunk_size` and respects UTF-8
/// char boundaries.
#[must_use]
pub fn chunk_by_section(doc: &str, max_chunk_size: usize, overlap: usize) -> Vec<String> {
    if doc.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = doc.chars().collect();
    let max_len = max_chunk_size.max(1);
    let step = max_len.saturating_sub(overlap).max(1);

    let mut sections = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + max_len).min(chars.len());
        sections.push(chars[start..end].iter().collect());
        if end == chars.len() {
            break;
        }
        start += step;
    }
    sections
}

/// Splits `doc` on page-break markers (form feeds, `"Page X"`, decorated page headers).
///
/// # Errors
/// Returns [`MinionError::Regex`] if custom `page_markers` fail to compile.
pub fn chunk_by_page(doc: &str, page_markers: Option<&[&str]>) -> Result<Vec<String>, MinionError> {
    if doc.is_empty() {
        return Ok(Vec::new());
    }
    let default_markers = [
        r"\f",
        r"(?mi)^page\s+\d+(\s+of\s+\d+)?\s*$",
        r"(?mi)^[\s_\-()]*\d+[\s_\-()]*$",
        r"(?mi)^[-=#]{3,}\s*.*page.*[-=#]{3,}\s*$",
        r"(?mi)^\s*[\[<(\[]\s*page(?:\s+\d+)?\s*[\]>)\]]\s*$",
    ];
    let markers = page_markers.unwrap_or(&default_markers);
    let pattern = markers.join("|");
    let re = Regex::new(&pattern)?;

    let mut pages = Vec::new();
    let mut start = 0;
    for mat in re.find_iter(doc) {
        let slice = doc[start..mat.start()].trim();
        if !slice.is_empty() {
            pages.push(slice.to_string());
        }
        start = mat.end();
    }
    let tail = doc[start..].trim();
    if !tail.is_empty() {
        pages.push(tail.to_string());
    }
    if pages.is_empty() {
        pages.push(doc.to_string());
    }
    Ok(pages)
}

/// Groups `sentences` into chunks bounded by `max_chunk_size` with sentence overlap.
#[must_use]
pub fn chunk_sentences(
    sentences: &[String],
    max_chunk_size: usize,
    overlap_sentences: usize,
) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut current_len = 0usize;

    for sentence in sentences {
        let sep_len = usize::from(!current.is_empty());
        let new_len = current_len + sep_len + sentence.len();
        if new_len > max_chunk_size && !current.is_empty() {
            chunks.push(current.join(" "));
            let keep = overlap_sentences.min(current.len());
            let mut next = current[current.len() - keep..].to_vec();
            next.push(sentence.clone());
            current_len =
                next.iter().map(String::len).sum::<usize>() + next.len().saturating_sub(1);
            current = next;
        } else {
            current.push(sentence.clone());
            current_len = new_len;
        }
    }
    if !current.is_empty() {
        chunks.push(current.join(" "));
    }
    chunks
}

/// Splits `doc` by paragraph boundaries, falling back to sentence chunking for
/// oversized paragraphs.
#[must_use]
pub fn chunk_by_paragraph(
    doc: &str,
    max_chunk_size: usize,
    overlap_sentences: usize,
) -> Vec<String> {
    if doc.trim().is_empty() {
        return Vec::new();
    }
    let para_re = Regex::new(r"\n\s*\n").expect("static paragraph regex is valid");

    let paragraphs: Vec<String> = para_re
        .split(doc)
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .collect();

    let mut chunks = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut current_len = 0usize;

    for paragraph in paragraphs {
        if paragraph.len() > max_chunk_size {
            if !current.is_empty() {
                chunks.push(current.join("\n\n"));
                current.clear();
                current_len = 0;
            }
            let sentences = split_sentences_simple(&paragraph);
            chunks.extend(chunk_sentences(
                &sentences,
                max_chunk_size,
                overlap_sentences,
            ));
        } else {
            let sep_len = if current.is_empty() { 0 } else { 2 };
            let candidate_len = current_len + sep_len + paragraph.len();
            if candidate_len > max_chunk_size && !current.is_empty() {
                chunks.push(current.join("\n\n"));
                if overlap_sentences > 0 {
                    let prev_sents = current
                        .last()
                        .map(|p| split_sentences_simple(p))
                        .unwrap_or_default();
                    let keep = overlap_sentences.min(prev_sents.len());
                    let overlap = prev_sents[prev_sents.len() - keep..].join(" ");
                    current_len = overlap.len() + 2 + paragraph.len();
                    current = vec![overlap, paragraph];
                } else {
                    current_len = paragraph.len();
                    current = vec![paragraph];
                }
            } else {
                current.push(paragraph);
                current_len = candidate_len;
            }
        }
    }
    if !current.is_empty() {
        chunks.push(current.join("\n\n"));
    }
    chunks
}

fn split_sentences_simple(text: &str) -> Vec<String> {
    // Rust's `regex` crate intentionally omits lookbehind, so split on [.!?] + whitespace.
    let mut sentences = Vec::new();
    let mut start = 0usize;
    let bytes = text.as_bytes();
    for (idx, ch) in text.char_indices() {
        if matches!(ch, '.' | '!' | '?') {
            let next_idx = idx + ch.len_utf8();
            if next_idx < bytes.len() && bytes[next_idx].is_ascii_whitespace() {
                let s = text[start..next_idx].trim();
                if !s.is_empty() {
                    sentences.push(s.to_string());
                }
                start = next_idx;
            }
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        sentences.push(tail.to_string());
    }
    sentences
}

/// Splits source code into function/block chunks (`functions_per_chunk` per batch).
///
/// Supports both brace-delimited (`fn `, `impl `, `struct `, `function `) and
/// indentation-delimited (`def `, `async def `, `class `) code blocks.
#[must_use]
pub fn chunk_by_code(doc: &str, functions_per_chunk: usize) -> Vec<String> {
    let batch_size = functions_per_chunk.max(1);
    let lines: Vec<&str> = doc.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut blocks = Vec::new();
    let mut current = Vec::new();
    let mut in_block = false;

    for line in lines {
        let trimmed = line.trim_start();
        let is_top_decl = !line.starts_with(' ')
            && !line.starts_with('\t')
            && (trimmed.starts_with("fn ")
                || trimmed.starts_with("pub fn ")
                || trimmed.starts_with("async fn ")
                || trimmed.starts_with("pub async fn ")
                || trimmed.starts_with("def ")
                || trimmed.starts_with("async def ")
                || trimmed.starts_with("class ")
                || trimmed.starts_with("function ")
                || trimmed.starts_with("export function "));

        if is_top_decl && in_block && !current.is_empty() {
            blocks.push(current.join("\n"));
            current.clear();
        }
        if is_top_decl {
            in_block = true;
        }
        current.push(line);
    }
    if !current.is_empty() {
        blocks.push(current.join("\n"));
    }

    if blocks.is_empty() {
        return vec![doc.to_string()];
    }

    blocks
        .chunks(batch_size)
        .map(|group| group.join("\n\n"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_chunking_with_overlap() {
        let doc = "abcdefghij";
        let chunks = chunk_by_section(doc, 4, 1);
        assert_eq!(chunks, vec!["abcd", "defg", "ghij"]);
    }

    #[test]
    fn page_chunking_splits_form_feeds() {
        let doc = "First page\x0cSecond page\nPage 3\nThird page";
        let pages = chunk_by_page(doc, None).expect("default regex compiles");
        assert_eq!(pages.len(), 3);
        assert_eq!(pages[0], "First page");
        assert_eq!(pages[2], "Third page");
    }

    #[test]
    fn paragraph_chunking_respects_budget() {
        let doc = "First para.\n\nSecond para.\n\nThird para.";
        let chunks = chunk_by_paragraph(doc, 26, 0);
        assert_eq!(chunks, vec!["First para.\n\nSecond para.", "Third para."]);
    }

    #[test]
    fn code_chunking_groups_functions() {
        let code = "fn a() {\n    1\n}\n\nfn b() {\n    2\n}\n\nfn c() {\n    3\n}";
        let chunks = chunk_by_code(code, 2);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].contains("fn a()") && chunks[0].contains("fn b()"));
        assert!(chunks[1].contains("fn c()"));
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    #[kani::unwind(3)]
    fn verify_chunk_by_section_empty_and_singleton() {
        let max_chunk_size: usize = kani::any();
        let overlap: usize = kani::any();
        assert!(chunk_by_section("", max_chunk_size, overlap).is_empty());

        let single = chunk_by_section("a", max_chunk_size, overlap);
        assert_eq!(single.len(), 1);
        assert_eq!(single[0], "a");
    }

    #[kani::proof]
    fn verify_section_step_progress_all_usize() {
        let max_chunk_size: usize = kani::any();
        let overlap: usize = kani::any();
        let max_len = max_chunk_size.max(1);
        let step = max_len.saturating_sub(overlap).max(1);
        assert!(max_len >= 1);
        assert!(step >= 1);
        assert!(step <= max_len);
    }
}
