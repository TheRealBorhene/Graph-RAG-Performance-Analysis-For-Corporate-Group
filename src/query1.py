import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase

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
client_oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
  Parent          — the filing company (e.g. "Loews Corporation")
  Subsidiary      — companies owned by the Parent (e.g. "CNA Financial Corporation")
  Geography       — physical locations (e.g. "United States", "Illinois")
  BusinessSegment — GAAP reporting segments
  FinancialItem   — enriched financial values whose name encodes both the metric label
                    and the raw value. Format: "Metric Label: $value"
                    Examples: "Net Income: $894", "Total Revenues: $14,931", "Net Income: -$175"

All nodes share the :Entity label and have a `name` property.

Relationship types (read-only):
  (Parent)-[:PARENT_OF]->(Subsidiary)
  (Parent|Subsidiary)-[:OPERATES_IN]->(Geography)
  (Parent)-[:REPORTED]->(FinancialItem)
  (Subsidiary)-[:GENERATED]->(FinancialItem)

CRITICAL — financial relationship properties:
  r.property  = fiscal year only, e.g. "2019", "2018". Never a metric name.
  r.page_number = source page in the document. Use this for citations, not r.property.
  The metric label lives in f.name before the colon, e.g. "Net Income" in "Net Income: $894".

  To find a specific metric, filter on f.name using CONTAINS:
    WRONG:  WHERE r.property CONTAINS 'net income'
    CORRECT: WHERE f.name CONTAINS 'Net Income' AND r.property = '2019'

  To find all figures for a fiscal year:
    WHERE r.property = '2019'

  Always RETURN f.name, r.property, r.page_number together for financial edges.

CRITICAL — comparison queries (highest / lowest / most / least):
  When comparing a metric across subsidiaries, prefer the most aggregate form.
  Use the label that includes "Total" when available (e.g. "Total Revenues" not "Revenues").
  Return all matching records and let the answer synthesize the comparison — do not try
  to compute MAX in Cypher on a string field.

CRITICAL — losses and negative values:
  Losses are NOT stored under a "Net Loss" label.
  A loss is a negative Net Income: e.g. "Net Income: -$175".
  To find entities with a net loss, search for Net Income nodes whose name contains '-':
    WHERE f.name CONTAINS 'Net Income' AND f.name CONTAINS '-'
  Never search for 'Net Loss' — that label does not exist in the graph.
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
- For financial edges always RETURN f.name, r.property, r.page_number
  (r.property is the fiscal year; r.page_number is the source page for citations)
- For comparison questions (most/highest/lowest/least), return all candidates —
  do not compute MAX/MIN on string fields; let the synthesis decide
- For loss/negative questions, search for the positive metric name with '-' in f.name,
  never search for a "loss" label
- For financial metric questions where the relationship type is uncertain
  (REPORTED vs HAS_METRIC), generate TWO queries — one for each type — to maximise recall.
  A metric may be stored under either depending on how it was classified during extraction.
- For revenue/income metric queries on a specific subsidiary, generate TWO queries:
  one filtering f.name CONTAINS 'Total Revenue' (or 'Total Revenues') and one filtering
  f.name CONTAINS 'Revenue' without 'Total'. Some subsidiaries store revenue under the
  aggregate label, others under a shorter label — both must be tried.
- For geographic aggregation questions (most locations, most countries, etc.):
  Geography nodes include cities, states/provinces, and countries mixed together.
  Return the full list of geography names — do NOT return a raw count and call it
  "number of countries". Let the synthesis interpret the list.
- For reverse geographic lookup questions ("which entities operate in [place]"):
  Generate TWO queries — one matching the exact place name (e.g. g.name = 'United States')
  and one matching geography nodes that CONTAIN the place name
  (e.g. g.name CONTAINS 'United States' to catch 'Houston, United States' style entries).
- CRITICAL — multi-relationship type syntax:
  WRONG:   [r:GENERATED|r:HAS_METRIC]   ← repeating the variable is invalid Cypher
  CORRECT: [r:GENERATED|HAS_METRIC]     ← variable declared once, types separated by |
- If the question is purely narrative and no graph data can help, return an empty list
- ONLY READ operations — no CREATE, MERGE, SET, DELETE
- Do not use APOC or any plugins
- Return ONLY valid JSON: {{"queries": ["MATCH ... RETURN ...", "MATCH ... RETURN ..."]}}
  or {{"queries": []}} if no graph data is needed
"""

SYNTHESIS_SYSTEM = """You are a financial analyst assistant for a company's 10-K annual report.

You receive structured facts extracted from a financial knowledge graph.
These facts are ACCURATE and VERIFIED — treat them as ground truth.

Your task: write a clean, well-structured answer to the user's question using only this graph data.

Guidelines:
- TRUST the graph data completely. If a record says "Net Income: $894" for year "2019",
  state that as fact. Never say the data is missing or unclear when records are present.
- NEVER fabricate or infer values not explicitly present in the graph records.
  If a specific figure (e.g. an ownership percentage, a date, a ratio) is not returned
  by the graph query, state clearly that this information is not available in the graph.
  Do not guess, round, or derive values — only report what the records contain.
- FinancialItem node names encode both the metric and its value: "Net Income: $894" means
  the metric is Net Income and the value is $894. The year comes from r.property.
- Geography nodes may contain cities, states, provinces, or countries mixed together.
  When answering geographic questions, list the locations as returned and qualify them
  appropriately (e.g. "cities and countries") — never present a raw location count
  as a "number of countries" unless the records clearly contain only country-level nodes.
- Format numbers cleanly (prefer "$894 million" over raw "894")
- Use bullet points or sections when the answer is multi-part
- When multiple records describe the same metric but with slightly different values
  (e.g. $1,051M on one page and $1,059M on another), do not list them as separate facts.
  Instead, prefer the record whose node name contains 'Total' or is the most aggregate,
  and note the source page. If values differ materially, state the range honestly.
- Only say data is missing if the graph context is literally empty
- After every factual claim cite the source as (Graph, p.<page_number>)
  using the page_number field from the graph record. If page_number is absent write (Graph)
- Be concise but complete
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
        section  = hit.payload.get("section",     "unknown section")
        page     = hit.payload.get("page_number", "?")
        text     = hit.payload.get("text",        "").strip()
        score    = round(hit.score, 3)
        passages.append(f"[{section}  page={page}  score={score}]\n{text}")

    print(f"  [Vector] {len(passages)} passage(s) retrieved")
    return "\n\n---\n\n".join(passages)


# ══════════════════════════════════════════════════════
# STEP 3 — SYNTHESIS
# ══════════════════════════════════════════════════════
def _synthesize(question: str, graph_ctx: str) -> str:
    """Generate a natural-language answer from graph context only."""
    context = graph_ctx if graph_ctx else "No relevant data was found in the knowledge graph."

    response = client_oai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user",   "content": f"Question: {question}\n\nGraph data:\n{context}"}
        ],
        temperature=0,
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

    graph_ctx = _graph_fetch(question, driver, entity_catalog)

    print(f"{'─' * 60}")
    return _synthesize(question, graph_ctx)


# ══════════════════════════════════════════════════════
# ENTRY POINT
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
