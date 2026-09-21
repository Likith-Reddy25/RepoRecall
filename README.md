# ⚡ RepoRecall — Event-Driven Codebase RAG & Semantic Search Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **An enterprise-grade, event-driven Codebase RAG system that eliminates stale vector hallucinations and reduces embedding compute costs by ~85% through differential Git indexing, 3-stage hybrid retrieval, and adaptive query routing.**

---

## 🌟 The Core Problem Solved

Standard Codebase RAG architectures suffer from two major production bottlenecks:
1. **Expensive Full Re-indexing:** A single modified line in a large repository usually triggers a complete, expensive vector database rebuild.
2. **Stale Vector Hallucinations:** When files are deleted or refactored in Git, naive vector databases retain obsolete chunks, causing the LLM to hallucinate outdated code.

**RepoRecall solves this by combining Git Webhooks with an atomic differential indexing pipeline and a 3-stage hybrid search engine.**

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Event-Driven Incremental Sync"]
        Push["🐙 GitHub Push Commit"] -->|POST /webhook| API["⚡ FastAPI Gateway"]
        API -->|Async BackgroundTask| DiffCheck{"Inspect Git Diff"}
        DiffCheck -->|Removed Files| Delete["🗑️ Atomic Vector Deletion (where file_path=...)"]
        DiffCheck -->|Added / Modified Files| Fetch["🌐 Fetch Raw Code via GitHub API"]
        Fetch --> Splitter["✂️ Python AST-Aware Splitter"]
        Splitter --> Embed["🔢 MiniLM-L6-v2 Embeddings"]
        Delete --> ChromaDB[("🗄️ ChromaDB")]
        Embed --> ChromaDB
    end

    subgraph Retrieval ["2. Adaptive Hybrid Retrieval & Synthesis"]
        Query["👤 User Query"] --> Router{"🤖 Adaptive Router"}
        Router -->|General Chit-Chat / Greetings| DirectLLM["Direct LLM Response"]
        Router -->|Codebase-Specific Question| Stage1["Stage 1: Dense Vector Search (k=15)"]
        ChromaDB --> Stage1
        Stage1 --> Stage2["Stage 2: BM25 Lexical Keyword Search"]
        Stage2 --> Stage3["Stage 3: FlashRank Cross-Encoder Reranking"]
        Stage3 --> Context["Top-K Grounded Code Snippets"]
        Context --> GroqLLM["🤖 Groq (LLaMA-3.3-70B)"]
        GroqLLM --> Answer["📄 Answer with File Citations"]
    end
