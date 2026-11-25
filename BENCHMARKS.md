# Benchmarks & Coverage

Measured on **Apple M3 Pro** (`arm64`, 11 cores, 18 GiB RAM, macOS) with `rustc 1.100.0` (`--release`) vs. `Python 3.15.2` (same host) and upstream `HazyResearch/minions`.

## 1. Code Coverage (`cargo llvm-cov --workspace --summary-only`)

| Scope | Region Coverage | Function Coverage | Line Coverage |
| :--- | ---: | ---: | ---: |
| `minions-core` | 73.1% – 100.0% | 76.9% – 100.0% | 64.5% – 100.0% |
| `minions-protocol` | 78.9% – 87.4% | 75.0% – 82.4% | 85.2% – 94.5% |
| **Workspace Total** *(incl. CLI & live HTTP clients)* | **64.21%** | **50.86%** | **59.43%** |

## 2. Footprint, Startup & Microbenchmarks

| Metric / Operation | `minions-rs` (Rust) | Upstream `HazyResearch/minions` (Python) | Comparison |
| :--- | ---: | ---: | :--- |
| **Standalone Binary Size** | **7.30 MiB** | Python + `tiktoken` / `pydantic` / `mcp` venv | Single static-linked binary |
| **Cold Startup (5-run median)** | **6.30 ms** | **23.61 ms** (`python3 -c ""` bare stdlib) | **3.75× faster** cold start |
| `chunk_by_section(40KB)` | **0.0588 ms** | **0.0072 ms** (zero-copy Python slice) | UTF-8 validated char chunks |
| `chunk_by_paragraph(40KB)` | **0.1806 ms** | — | Sentence-fallback splitting |
| `bm25_retrieve_top_k(chunks, k=5)` | **0.6858 ms** | — | Zero-dependency Okapi BM25 |
| `extract_json(1000x)` | **74.79 ms** | — | Markdown-fence resilient parser |

Reproduce locally:
```bash
cargo run --release -p minions-core --example bench
cargo llvm-cov --workspace --summary-only
```
