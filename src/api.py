"""
REST API for the Graph RAG system.

Exposes the existing query layer (query1.py) over HTTP with auto-generated
Swagger UI. Wraps — does not duplicate — the existing functions so any
improvements to the query layer are immediately reflected in the API.

Run:
    uvicorn api:app --reload --port 8000

Then open:
    http://localhost:8000/docs        — interactive Swagger UI
    http://localhost:8000/redoc       — alternative ReDoc UI
    http://localhost:8000/openapi.json — raw OpenAPI 3 spec
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from query1 import (
    connect_neo4j,
    fetch_entity_catalog,
    _graph_fetch,
    _vector_fetch,
    _synthesize,
)


# ─────────────────────────────────────────────
# APP STATE — Neo4j driver + entity catalog, initialised at startup
# ─────────────────────────────────────────────
class State:
    driver: Any = None
    entity_catalog: str = ""
    filing_company: str = ""


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the Neo4j connection at startup, close it at shutdown."""
    state.driver = connect_neo4j()
    state.entity_catalog = fetch_entity_catalog(state.driver)
    with state.driver.session() as session:
        rec = session.run(
            "MATCH (p:Parent)-[:PARENT_OF]->() RETURN p.name AS name LIMIT 1"
        ).single()
        state.filing_company = rec["name"] if rec else "Unknown Company"
    print(f"  [API] Connected to Neo4j  |  filing company: {state.filing_company}")
    yield
    state.driver.close()
    print("  [API] Neo4j connection closed.")


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Graph RAG API — 10-K Financial Knowledge Graph",
    description=(
        "REST interface to query a Neo4j knowledge graph extracted from a "
        "company's 10-K annual report. Built on top of a deterministic multi-agent "
        "extraction pipeline. Retrieval is **hybrid**: the graph is queried first, "
        "and vector search is used as a fallback when the graph returns no records.\n\n"
        "**Endpoints:**\n"
        "- `POST /ask` — Hybrid question answering (graph → vector fallback) with citations. "
        "The `source` field in the response shows which path was used.\n"
        "- `POST /cypher` — Graph-only retrieval (raw records, no synthesis).\n"
        "- `POST /vector` — Vector-only retrieval (raw passages, no synthesis).\n"
        "- `GET  /health` — Liveness check + Neo4j connectivity.\n"
        "- `GET  /stats` — Graph statistics (entity/relationship counts by type).\n"
        "- `GET  /catalog` — Full entity catalog (names grouped by type).\n"
        "- `GET  /` — API metadata."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Permissive CORS so the API can be called from a separate frontend during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        description="Natural-language question to answer using the graph.",
        examples=["How much net income did CNA Financial Corporation generate in 2019?"],
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    filing_company: str
    source: str = Field(
        description=(
            "Which retrieval path produced the answer: "
            "'graph' (Neo4j Cypher query), 'vector' (Qdrant fallback when graph was empty), "
            "or 'empty' (neither returned anything)."
        ),
        examples=["graph"],
    )
    record_count: int = Field(
        description="Number of records/passages retrieved from the active source."
    )


class CypherResponse(BaseModel):
    question: str
    record_count: int
    records: list[str] = Field(
        description="Plain-text serialized graph records, one per line.",
    )


class VectorResponse(BaseModel):
    question: str
    passage_count: int
    passages: list[str] = Field(
        description="Top-K passages retrieved from the vector store, with page citations in headers.",
    )


class HealthResponse(BaseModel):
    status: str
    neo4j_connected: bool
    filing_company: str


class StatsResponse(BaseModel):
    filing_company: str
    entity_count: int
    relationship_count: int
    entities_by_type: dict[str, int]
    relationships_by_type: dict[str, int]


class CatalogResponse(BaseModel):
    filing_company: str
    total_entities: int
    entities_by_type: dict[str, list[str]]


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/", tags=["Meta"])
def root() -> dict:
    """Lightweight metadata about the API."""
    return {
        "service": "Graph RAG API",
        "filing_company": state.filing_company,
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health() -> HealthResponse:
    """Liveness check. Verifies the Neo4j connection is alive."""
    connected = False
    try:
        state.driver.verify_connectivity()
        connected = True
    except Exception:
        connected = False
    return HealthResponse(
        status="ok" if connected else "degraded",
        neo4j_connected=connected,
        filing_company=state.filing_company,
    )


@app.post("/ask", response_model=AskResponse, tags=["Query"])
def ask(request: AskRequest) -> AskResponse:
    """
    Hybrid retrieval:
      1. Generate Cypher and query the graph.
      2. If the graph returns nothing, fall back to vector search over the source document.
      3. Synthesize a natural-language answer with page citations.

    The `source` field in the response tells you which retrieval path produced
    the answer ('graph', 'vector', or 'empty').
    """
    q = request.question
    try:
        graph_ctx = _graph_fetch(q, state.driver, state.entity_catalog)
        vector_ctx = ""
        source = "graph"

        if not graph_ctx:
            vector_ctx = _vector_fetch(q)
            source = "vector" if vector_ctx else "empty"

        record_count = (
            len([l for l in graph_ctx.split("\n") if l.strip()]) if graph_ctx
            else (vector_ctx.count("\n---\n") + 1 if vector_ctx else 0)
        )
        result = _synthesize(q, graph_ctx, vector_ctx)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Answer pipeline failed: {exc}")

    return AskResponse(
        question=q,
        answer=result,
        filing_company=state.filing_company,
        source=source,
        record_count=record_count,
    )


@app.post("/cypher", response_model=CypherResponse, tags=["Query"])
def cypher(request: AskRequest) -> CypherResponse:
    """
    Run only the graph-fetch step (Cypher generation + execution) and return the raw
    serialized records — no LLM synthesis. Useful for debugging what facts the graph
    actually surfaces for a given question.
    """
    try:
        ctx = _graph_fetch(request.question, state.driver, state.entity_catalog)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph fetch failed: {exc}")
    lines = [l for l in ctx.split("\n") if l.strip()] if ctx else []
    return CypherResponse(
        question=request.question,
        record_count=len(lines),
        records=lines,
    )


@app.post("/vector", response_model=VectorResponse, tags=["Query"])
def vector(request: AskRequest) -> VectorResponse:
    """
    Run only the vector-fetch step (embed question + Qdrant top-K) and return raw
    passages — no graph query, no LLM synthesis. Useful for demoing the fallback
    path directly or for narrative questions the graph can't answer.
    """
    try:
        ctx = _vector_fetch(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vector fetch failed: {exc}")
    passages = [p.strip() for p in ctx.split("\n---\n") if p.strip()] if ctx else []
    return VectorResponse(
        question=request.question,
        passage_count=len(passages),
        passages=passages,
    )


@app.get("/stats", response_model=StatsResponse, tags=["Inspection"])
def stats() -> StatsResponse:
    """Return entity and relationship counts grouped by type."""
    with state.driver.session() as session:
        ent_rows = list(session.run(
            "MATCH (n:Entity) RETURN n.type AS type, count(*) AS cnt ORDER BY cnt DESC"
        ))
        rel_rows = list(session.run(
            "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS cnt ORDER BY cnt DESC"
        ))
    by_type = {r["type"] or "Unknown": r["cnt"] for r in ent_rows}
    by_rel  = {r["rel_type"]: r["cnt"] for r in rel_rows}
    return StatsResponse(
        filing_company=state.filing_company,
        entity_count=sum(by_type.values()),
        relationship_count=sum(by_rel.values()),
        entities_by_type=by_type,
        relationships_by_type=by_rel,
    )


@app.get("/catalog", response_model=CatalogResponse, tags=["Inspection"])
def catalog() -> CatalogResponse:
    """Return every named entity in the graph, grouped by type."""
    with state.driver.session() as session:
        rows = list(session.run(
            "MATCH (n:Entity) RETURN n.type AS type, n.name AS name ORDER BY n.type, n.name"
        ))
    grouped: dict[str, list[str]] = {}
    for r in rows:
        t = r["type"] or "Unknown"
        n = r["name"]
        if n:
            grouped.setdefault(t, []).append(n)
    return CatalogResponse(
        filing_company=state.filing_company,
        total_entities=sum(len(v) for v in grouped.values()),
        entities_by_type=grouped,
    )
