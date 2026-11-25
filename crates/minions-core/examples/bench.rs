//! Release benchmark for `minions-core` chunking, BM25 retrieval, and JSON extraction.

use std::hint::black_box;
use std::time::Instant;

use minions_core::{
    bm25_retrieve_top_k_chunks, chunk_by_paragraph, chunk_by_section, extract_json,
};
use serde_json::Value;

fn bench<F: FnMut()>(name: &str, warmup: usize, iters: usize, mut f: F) {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    let median = times[times.len() / 2];
    println!(
        "{name}: median={median:.4} ms (min={:.4} ms, max={:.4} ms)",
        times[0],
        times[times.len() - 1]
    );
}

fn main() {
    let doc = "Cloud infrastructure costs dropped significantly after offloading long-context reading to local SLMs.\n\n".repeat(400);
    let chunks = chunk_by_section(&doc, 1000, 100);

    bench("chunk_by_section(40KB)", 3, 15, || {
        black_box(chunk_by_section(black_box(&doc), 1000, 100));
    });
    bench("chunk_by_paragraph(40KB)", 3, 15, || {
        black_box(chunk_by_paragraph(black_box(&doc), 1000, 1));
    });
    bench("bm25_retrieve_top_k(chunks, k=5)", 3, 15, || {
        black_box(bm25_retrieve_top_k_chunks(
            black_box(&chunks),
            "local SLM cloud costs",
            5,
        ));
    });
    let payload = "Here is the answer:\n```json\n{\"answer\": \"42\"}\n```";
    bench("extract_json(1000x)", 3, 15, || {
        for _ in 0..1000 {
            let _ = black_box(extract_json::<Value>(black_box(payload)));
        }
    });
}
