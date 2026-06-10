# Graph RAG — Financial 10-K Analysis

A Retrieval-Augmented Generation (RAG) system that builds a **knowledge graph** from SEC 10-K annual reports and enables intelligent financial querying by combining graph traversal with semantic vector search.

Built as a Final Year Project (PFE) exploring the performance of Graph RAG for corporate group analysis.

---

## How It Works

The pipeline runs in three stages:

### 1. Ingestion
- Loads a 10-K document (Markdown format)
- Splits it into section-aware chunks
- Embeds chunks using `all-MiniLM-L6-v2` and stores them in **Qdrant** (vector store)

### 2. Knowledge Graph Extraction
A 3-agent LLM pipeline (powered by GPT-4.1-mini) processes each chunk:
- **Agent 1 — Entity Finder**: extracts candidate entities (companies, locations, financial figures)
- **Agent 2 — Entity Classifier**: classifies candidates into typed nodes: `Parent`, `Subsidiary`, `Geography`, `BusinessSegment`, `FinancialItem`
- **Agent 3 — Relationship Finder**: extracts typed relationships across 3 focused segments:
  - Structural: `PARENT_OF`, `OPERATES_IN`
  - Financial reporting: `REPORTED`, `GENERATED`
  - Supplementary metrics: `HAS_METRIC`

The resulting graph is deduplicated, enriched, and stored in **Neo4j**.

### 3. Querying
A hybrid RAG query pipeline answers natural language questions:
- **Graph fetch**: LLM generates Cypher queries → executed against Neo4j
- **Vector fetch**: semantic search over Qdrant passages
- Both run in parallel, then a synthesis LLM combines the results into a clean answer with page citations

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | OpenAI GPT-4.1-mini |
| Vector Store | Qdrant |
| Graph Database | Neo4j |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Language | Python 3.12 |

---

## Project Structure

    src/
    ├── main.py          # Pipeline entry point (ingest → extract → store)
    ├── loader.py        # Document loading
    ├── chunker.py       # Section-aware chunking + Qdrant embedding
    ├── agents.py        # 3-agent LLM extraction pipeline
    ├── extractor.py     # Orchestrates extraction, merging, enrichment
    ├── graph.py         # Neo4j graph storage
    ├── guards.py        # Entity and relationship validation guards
    ├── query.py         # Hybrid RAG query engine
    ├── eval.py          # Evaluation suite (4 dimensions)
    ├── checkpoint.py    # Fault-tolerant extraction checkpointing
    └── logger.py        # Structured extraction logging

    data/
    ├── LoewsCompany.md           # Sample 10-K filing (Loews Corporation FY2019)
    ├── NVDA_..._FINAL.md         # Sample 10-K filing (NVIDIA)
    ├── graph_output.json         # Extracted graph (entities + relationships)
    └── logs/                     # Per-run extraction logs

---

## Evaluation

The system includes a built-in evaluation suite (`eval.py`) measuring 4 dimensions against ground-truth data from the Loews Corporation FY2019 10-K:

| Dimension | What it checks |
|---|---|
| Structural completeness | Are expected entities and relationships present? |
| Financial accuracy | Do extracted figures match known ground-truth values? |
| Geographic coverage | Are operational locations correctly linked? |
| Query coverage | Do benchmark questions return non-empty graph results? |

---

## Setup

### Prerequisites
- Python 3.12+
- Neo4j (local or cloud)
- Qdrant (local or cloud)
- OpenAI API key

### Install dependencies

    pip install -r requirements.txt

### Configure environment

Create a `.env` file (never commit this):

    OPENAI_API_KEY=your_key_here
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your_password
    QDRANT_HOST=localhost
    QDRANT_PORT=6333

### Run the pipeline

    # Ingest document, extract graph, store in Neo4j
    python src/main.py

    # Query interactively
    python src/query.py

    # Single question
    python src/query.py "What subsidiaries does Loews Corporation own?"

    # Run evaluation
    python src/eval.py
