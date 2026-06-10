import os
import sys
import json
import concurrent.futures
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
COLLECTION_NAME = "LoewsCompany"
VECTOR_TOP_K    = 6
MODEL           = "gpt-4.1-mini"

# ─────────────────────────────────────────────
# CLIENTS  (module-level so they are reused across calls)
# ─────────────────────────────────────────────
client_oai    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_qdrant = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", 6333))
)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def connect_neo4j() -> GraphDatabase.driver:
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI",      "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USER",     "neo4j"),
            os.getenv("NEO4J_PASSWORD", "password")
        )
    )
    driver.verify_connectivity()
    return driver


def fetch_entity_catalog(driver) -> str:
    """
    Fetch all named entities from Neo4j grouped by type.
    Returns a plain-text block to inject into the Cypher prompt so the LLM
    always uses exact canonical names instead of abbreviated forms.
    """
    query = """
        MATCH (n:Entity)
        RETURN n.type AS type, n.name AS name
        ORDER BY n.type, n.name
    """
    with driver.session() as session:
        records = [dict(r) for r in session.run(query)]

    grouped: dict[str, list[str]] = {}
    for rec in records:
        t = rec.get("type") or "Unknown"
        n = rec.get("name") or ""
        if n:
            grouped.setdefault(t, []).append(n)

    lines = ["Exact entity names present in the graph (use these verbatim in WHERE clauses):"]
    for entity_type, names in sorted(grouped.items()):
        lines.append(f"\n  {entity_type}:")
        for name in names:
            lines.append(f"    - \"{name}\"")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# GRAPH SCHEMA  (fed to the Cypher-generation LLM)
# ─────────────────────────────────────────────
GRAPH_SCHEMA = """
Node labels and their meaning:
  Parent          — the filing company (e.g. "LOEWS CORPORATION")
  Subsidiary      — companies owned by the Parent (e.g. "CNA Financial Corporation")
  Geography       — physical locations (e.g. "United States", "Illinois")
  BusinessSegment — GAAP reporting segments (e.g. "CNA", "Boardwalk Pipeline")
  FinancialItem   — numeric values from financial statements (e.g. "8,589", "26.6%")

All nodes share the :Entity label and have these properties:
  name, type, page_number, chunk_id, file

Relationship types (read-only):
  (Parent)-[:PARENT_OF]->(Subsidiary)
  (Parent|Subsidiary)-[:OPERATES_IN]->(Geography)
  (Parent)-[:REPORTED]->(FinancialItem)     — r.property: fiscal year e.g. "2019"  r.page_number
  (Subsidiary)-[:GENERATED]->(FinancialItem) — r.property: fiscal year e.g. "2019"  r.page_number
"""

# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────
CYPHER_SYSTEM = f"""You are a Neo4j Cypher expert working with a financial knowledge graph.
Given a user question and the schema below, write as many Cypher READ queries as needed
to retrieve all data relevant to answering the question completely.

{GRAPH_SCHEMA}

Rules:
- Write multiple queries when the question covers several angles (e.g. financials + structure + geography)
- Always add LIMIT 60 to every query
- Always return node names (n.name) and page_number (n.page_number) for nodes
- Always return relationship property (r.property) and page_number (r.page_number) for financial edges
- CRITICAL: When using r.property or r.page_number anywhere in the query, you MUST name the relationship in the MATCH pattern.
  WRONG:   MATCH (a)-[:REPORTED]->(b) WHERE r.property ...
  CORRECT: MATCH (a)-[r:REPORTED]->(b) WHERE r.property ...
- If the question is purely narrative and no graph data can help, return an empty list
- ONLY READ operations — no CREATE, MERGE, SET, DELETE
- Do not use APOC or any plugins
- Return ONLY valid JSON: {{"queries": ["MATCH ... RETURN ...", "MATCH ... RETURN ..."]}}
  or {{"queries": []}} if no graph data is needed
"""

SYNTHESIS_SYSTEM = """You are a financial analyst assistant for a company's 10-K annual report.

You receive two sources of context:
  GRAPH DATA    — structured facts extracted from the document (entities, relationships, figures)
  TEXT PASSAGES — relevant excerpts from the original 10-K document

Your task: write a clean, well-structured answer to the user's question.

Guidelines:
- Use graph data for facts, numbers, ownership structure, and financial figures
- Use text passages for narrative context, explanations, risks, and qualitative insight
- If data is missing or unclear, say so honestly — never fabricate figures
- Format numbers cleanly (prefer "$8.6 billion" over raw "8,589")
- Use bullet points or sections when the answer is multi-part
- Be concise but complete — prioritise insight over raw data dumps
- Always cite the source page number when referencing a specific fact or figure.
  Use the format (page X) inline. Example: "CNA reported net income of $894 million (page 52)"
  For text passages, the page number appears in the passage header as page=X.
  For graph data, the page number appears as page_number in the record.
"""


# ══════════════════════════════════════════════════════
# STEP 1 — CYPHER GENERATION
# ══════════════════════════════════════════════════════
def _generate_queries(question: str, entity_catalog: str = "") -> list[str]:
    """Ask the LLM to produce all Cypher queries needed to answer the question."""
    system = CYPHER_SYSTEM
    if entity_catalog:
        system = CYPHER_SYSTEM + f"\n\n{entity_catalog}"

    response = client_oai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Question: {question}"}
        ],
        temperature=0,
        max_tokens=1024,
        response_format={"type": "json_object"}
    )
    try:
        queries = json.loads(response.choices[0].message.content).get("queries", [])
        return [q for q in queries if isinstance(q, str) and q.strip()]
    except (json.JSONDecodeError, AttributeError):
        return []


# ══════════════════════════════════════════════════════
# STEP 2a — GRAPH FETCH
# ══════════════════════════════════════════════════════
def _run_cypher(cypher: str, driver) -> list[str]:
    """Execute a single Cypher query and return serialized result lines."""
    with driver.session() as session:
        records = [dict(r) for r in session.run(cypher)]

    lines = []
    for rec in records:
        parts = []
        for k, v in rec.items():
            if v is None:
                continue
            # Neo4j Node objects expose items() like a dict
            if hasattr(v, "items"):
                v = dict(v).get("name", str(v))
            parts.append(f"{k}: {v}")
        if parts:
            lines.append(" | ".join(parts))
    return lines


def _graph_fetch(question: str, driver, entity_catalog: str = "") -> str:
    """Generate all Cypher queries, execute each, merge results into plain text."""
    queries = _generate_queries(question, entity_catalog)

    if not queries:
        print("  [Graph]  No queries generated — question is likely narrative-only")
        return ""

    print(f"  [Graph]  {len(queries)} query(ies) generated")

    all_lines = []
    for i, cypher in enumerate(queries, 1):
        preview = cypher[:100] + ("..." if len(cypher) > 100 else "")
        print(f"  [Graph]  Q{i}: {preview}")
        try:
            lines = _run_cypher(cypher, driver)
            print(f"           → {len(lines)} record(s)")
            all_lines.extend(lines)
        except Exception as e:
            print(f"           → Error: {e}")

    if not all_lines:
        print("  [Graph]  All queries returned 0 records")
        return ""

    # Deduplicate identical lines that multiple queries may return
    seen = set()
    deduped = []
    for line in all_lines:
        if line not in seen:
            seen.add(line)
            deduped.append(line)

    print(f"  [Graph]  {len(deduped)} unique record(s) total")
    return "\n".join(deduped)


# ══════════════════════════════════════════════════════
# STEP 2b — VECTOR FETCH
# ══════════════════════════════════════════════════════
def _vector_fetch(question: str) -> str:
    """Embed the question with the same model used at ingestion, search Qdrant."""
    vector = embed_model.encode(question).tolist()

    results = client_qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=VECTOR_TOP_K,
        with_payload=True
    )
    hits = results.points

    if not hits:
        print("  [Vector] No passages found")
        return ""

    passages = []
    for hit in hits:
        section     = hit.payload.get("section", "unknown section")
        text        = hit.payload.get("text", "").strip()
        score       = round(hit.score, 3)
        page_number = hit.payload.get("page_number", "?")
        passages.append(f"[{section}  page={page_number}  score={score}]\n{text}")

    print(f"  [Vector] {len(passages)} passage(s) retrieved")
    return "\n\n---\n\n".join(passages)


# ══════════════════════════════════════════════════════
# STEP 3 — SYNTHESIS
# ══════════════════════════════════════════════════════
def _synthesize(question: str, graph_ctx: str, vector_ctx: str) -> str:
    """Combine graph + vector context and generate a clean natural-language answer."""
    parts = []
    if graph_ctx:
        parts.append(f"=== GRAPH DATA ===\n{graph_ctx}")
    if vector_ctx:
        parts.append(f"=== TEXT PASSAGES ===\n{vector_ctx}")
    if not parts:
        parts.append("No relevant data was found in the knowledge base.")

    context = "\n\n".join(parts)

    response = client_oai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user",   "content": f"Question: {question}\n\nContext:\n{context}"}
        ],
        temperature=0.3,
        max_tokens=1500
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════
# PUBLIC — ANSWER A QUESTION
# ══════════════════════════════════════════════════════
def answer(question: str, driver, entity_catalog: str = "") -> str:
    """
    Full pipeline: parallel graph + vector fetch → synthesis → clean answer.
    Returns the answer as a string.
    """
    print(f"\n{'═' * 60}")
    print(f"Question: {question}")
    print(f"{'─' * 60}")

    # Graph fetch and vector fetch run simultaneously
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_graph  = executor.submit(_graph_fetch,  question, driver, entity_catalog)
        future_vector = executor.submit(_vector_fetch, question)

        graph_ctx  = future_graph.result()
        vector_ctx = future_vector.result()

    print(f"{'─' * 60}")
    result = _synthesize(question, graph_ctx, vector_ctx)
    return result


# ══════════════════════════════════════════════════════
# ENTRY POINT
# usage:  python query.py "your question here"
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    driver = connect_neo4j()

    try:
        # load entity catalog once — reused for every question
        entity_catalog = fetch_entity_catalog(driver)
        print(f"Entity catalog loaded  ({entity_catalog.count(chr(10))} lines)")

        # detect filing company from the Parent node in the graph
        with driver.session() as session:
            rec = session.run("MATCH (p:Parent) RETURN p.name AS name LIMIT 1").single()
            filing_company = rec["name"] if rec else "Unknown Company"

        # ── single-question mode: python query.py "question" ──────────────
        if len(sys.argv) > 1:
            question = " ".join(sys.argv[1:])
            result   = answer(question, driver, entity_catalog)
            print(f"\n{'═' * 60}")
            print("Answer:")
            print(f"{'─' * 60}")
            print(result)
            print(f"{'═' * 60}\n")

        # ── interactive loop mode: python query.py ─────────────────────────
        else:
            print(f"\nGraph RAG — {filing_company} 10-K")
            print("Type your question and press Enter. Type 'exit' to quit.")
            print("─" * 60)

            while True:
                try:
                    question = input("\nYou: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye.")
                    break

                if not question:
                    continue
                if question.lower() in {"exit", "quit", "q"}:
                    print("Goodbye.")
                    break

                result = answer(question, driver, entity_catalog)
                print(f"\n{'═' * 60}")
                print("Answer:")
                print(f"{'─' * 60}")
                print(result)
                print(f"{'═' * 60}")

    finally:
        driver.close()
