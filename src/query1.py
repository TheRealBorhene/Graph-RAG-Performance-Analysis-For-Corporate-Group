import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL            = "gpt-4.1-mini"
COLLECTION_NAME  = "LoewsCompany"      # must match what chunker.py wrote
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # must match what chunker.py used
VECTOR_TOP_K     = 6                   # how many passages to retrieve on fallback

# ─────────────────────────────────────────────
# CLIENTS  (module-level so they are reused across calls)
# ─────────────────────────────────────────────
client_oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Vector clients are lazily initialised (only loaded when the graph returns empty),
# so a graph-only query never pays the SentenceTransformer model-load cost.
_embed_model:  SentenceTransformer | None = None
_qdrant_client: QdrantClient       | None = None


def _ensure_vector_clients() -> None:
    """Lazy-load the embedding model + Qdrant client on first vector-fallback use."""
    global _embed_model, _qdrant_client
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    if _qdrant_client is None:
        _qdrant_client = qdrant_client()


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


def qdrant_client() -> QdrantClient:
    """Single Qdrant connection factory used by chunker, query1 and test_chunks."""
    return QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )


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
  Parent          — the filing company (the entity that issued the 10-K)
  Subsidiary      — companies owned by the Parent
  Geography       — physical locations: countries, states/provinces, or cities
  BusinessSegment — GAAP reportable operating segments disclosed by the Parent
  FinancialItem   — enriched financial values whose name encodes both the metric label
                    and the raw value. Format: "Metric Label: $value"
                    Examples: "Net Income: $894", "Total Revenues: $14,931", "Net Income: -$175"
  Person          — a named individual disclosed as a board member, director, or
                    executive officer of the Parent or a Subsidiary. The node's `name`
                    is the personal name as it appears in the document.
  IncomeStatement — income statement node for a fiscal year (e.g. "IncomeStatement_FY2019")
  BalanceSheet    — balance sheet node for a fiscal year (e.g. "BalanceSheet_FY2019")
  CashFlow        — cash flow statement node for a fiscal year (e.g. "CashFlow_FY2019")

All nodes share the :Entity label and have a `name` property.

CRITICAL — statement-node line items live as PROPERTIES on the node:
  IncomeStatement / BalanceSheet / CashFlow nodes carry their line items as Cypher
  properties (snake_case keys) on the node itself — NOT as separate FinancialItem nodes.
  These are the authoritative source for CONSOLIDATED, COMPANY-WIDE figures
  (the filing company's total revenue, net income, total assets, etc.).

  Two book-keeping property keys are always present:
    fiscal_year, unit
  The remaining property keys are GAAP-standard line items in snake_case, matching what
  the 10-K itself reports. You do not need an exhaustive list — generic GAAP names
  derived from the question (net_income, total_revenues, total_assets, interest,
  cash_from_operations, income_tax_expense, comprehensive_income_loss, etc.) will work.

  Pattern — fetch ALL line items for one year (use when the question is broad or you
  are not sure which specific property key is present):
    MATCH (p:Parent)-[:INCOME_STATEMENT]->(s:IncomeStatement)
    WHERE s.fiscal_year = '<year>'
    RETURN s.name, properties(s) AS line_items
    LIMIT 5

  Pattern — fetch named metrics across years (use when the question names specific
  metrics — derive the snake_case key from the metric):
    MATCH (p:Parent)-[:INCOME_STATEMENT]->(s:IncomeStatement)
    RETURN s.name, s.fiscal_year, s.<metric_key_1>, s.<metric_key_2>, s.unit
    LIMIT 60

  Same patterns apply to BalanceSheet (relationship [:BALANCE_SHEET]) and CashFlow
  (relationship [:CASH_FLOW]). Never hardcode the filing company's name in the MATCH —
  the (p:Parent) label uniquely identifies it.

  CRITICAL — statement-node property keys may include per-entity prefixed variants:
  A single statement node can hold BOTH a consolidated metric (e.g. `net_income`) AND
  per-subsidiary/segment versions of the same metric prefixed with the entity's
  snake_case name (e.g. `<entity>_net_income`,
  `<entity>_net_income_attributable_to_<parent>`). When the question names a SPECIFIC
  subsidiary or segment, you MUST prefer the prefixed key over the generic one,
  otherwise the consolidated figure will be returned as if it were the entity's.

  Strategy when the question names a specific entity:
    1. Always FIRST try the GENERATED-edge path (per-entity metrics are normally stored
       there as (Subsidiary|BusinessSegment)-[:GENERATED]->(FinancialItem)).
    2. For the statement-node query, you MUST use `RETURN properties(s) AS line_items`
       — NEVER `RETURN s.<short_metric_key>`. Reason: the generic key holds the
       CONSOLIDATED figure, not the entity's, so returning it would mislabel the
       consolidated total as the entity's value. The line_items dict lets the synthesis
       layer pick the correctly-prefixed key (e.g. `<entity>_net_income`) by inspection.

       CORRECT for an entity-specific question:
         MATCH (p:Parent)-[:INCOME_STATEMENT]->(s:IncomeStatement {{fiscal_year: '<year>'}})
         RETURN s.name, s.fiscal_year, properties(s) AS line_items
         LIMIT 5
       WRONG for an entity-specific question (returns consolidated, mislabelled):
         RETURN s.name, s.net_income

  When the question is about the consolidated Parent (no specific subsidiary named),
  the generic key (e.g. `s.net_income`, `s.total_revenues`) IS the right one and
  preferred over `properties(s)` for compactness.

Relationship types (read-only):
  (Parent)-[:PARENT_OF]->(Subsidiary)
  (Parent|Subsidiary)-[:OPERATES_IN]->(Geography)
  (Subsidiary|BusinessSegment)-[:GENERATED]->(FinancialItem)
  (Person)-[:BOARD_MEMBER_OF]->(Parent|Subsidiary)
  (Parent)-[:INCOME_STATEMENT]->(IncomeStatement)
  (Parent)-[:BALANCE_SHEET]->(BalanceSheet)
  (Parent)-[:CASH_FLOW]->(CashFlow)

CRITICAL — BOARD_MEMBER_OF edge properties:
  r.property    = the person's role at the company, verbatim from the document.
                  Examples: "Co-Chairman of the Board",
                            "President and Chief Executive Officer",
                            "Senior Vice President and Chief Financial Officer", "Director".
  r.page_number = source page in the document. Use this for citations.

  Always RETURN p.name, r.property, r.page_number together for governance edges.

  To find all people on a company's board / leadership:
    MATCH (p:Person)-[r:BOARD_MEMBER_OF]->(c:Parent {{name: '<company>'}})
    RETURN p.name, r.property, r.page_number
    LIMIT 60
  To find the role of a specific person:
    MATCH (p:Person {{name: '<person name>'}})-[r:BOARD_MEMBER_OF]->(c)
    RETURN c.name, r.property, r.page_number

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
- CRITICAL — source node type for financial metrics:
  Financial metrics (GENERATED relationships) may be stored under EITHER a Subsidiary node
  OR a BusinessSegment node depending on how the LLM classified the source during extraction.
  For ANY financial metric query, ALWAYS generate TWO queries in parallel:
    one matching (s:Subsidiary)-[:GENERATED]->(f:FinancialItem)
    one matching (s:BusinessSegment)-[:GENERATED]->(f:FinancialItem)
  Never query only Subsidiary or only BusinessSegment — always both.
- CRITICAL — consolidated company-wide figures live on statement nodes, not GENERATED edges:
  If the question asks about a CONSOLIDATED, COMPANY-WIDE, or TOTAL figure for the filing
  Parent (total/consolidated revenue, net income, total assets, cash flow from operations,
  etc.), ALSO generate a query that reads the IncomeStatement / BalanceSheet / CashFlow
  node properties — IN ADDITION to any GENERATED-path queries. Without it the consolidated
  total is silently missed.
    Pattern (do NOT include a Parent name filter — :Parent already identifies it):
      MATCH (p:Parent)-[:INCOME_STATEMENT]->(s:IncomeStatement {{fiscal_year: '<year>'}})
      RETURN s.name, s.fiscal_year, s.<metric_key>, s.unit
      LIMIT 5
  Pick the relationship and statement label that matches the metric class:
    income-statement metrics → [:INCOME_STATEMENT] → IncomeStatement
    balance-sheet metrics    → [:BALANCE_SHEET]    → BalanceSheet
    cash-flow metrics        → [:CASH_FLOW]        → CashFlow
  For broad "what's in this statement" questions, RETURN properties(s) AS line_items.
- For revenue/income metric queries, generate queries filtering f.name CONTAINS 'Revenue'
  broadly. Also try f.name CONTAINS 'Total Revenues' as a second query for aggregate totals.
- For geographic aggregation questions (most locations, most countries, etc.):
  Geography nodes include cities, states/provinces, and countries mixed together.
  Return the full list of geography names — do NOT return a raw count and call it
  "number of countries". Let the synthesis interpret the list.
- For ownership lookup questions ("who owns X", "what is the parent of X"):
  A subsidiary may be owned by the filing Parent OR by another Subsidiary (sub-to-sub ownership).
  Always generate TWO queries:
    one matching (p:Parent)-[:PARENT_OF]->(s:Subsidiary {{name: 'X'}})
    one matching (p:Subsidiary)-[:PARENT_OF]->(s:Subsidiary {{name: 'X'}})
  Never query only Parent as the owner — always check both.
- For governance / leadership questions ("who is on the board", "who is the CEO",
  "who are the directors of X", "who serves as chairman"):
  Match the Person -> Parent|Subsidiary BOARD_MEMBER_OF edge. Always RETURN p.name,
  r.property (the role string), AND r.page_number — the page number is required so the
  synthesizer can cite the source page for each governance fact.
  When the question names a specific company, filter on the target node's name.
  When the question is about the filing company, the target is :Parent.
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
- Statement-node records (IncomeStatement / BalanceSheet / CashFlow) deliver data differently:
  line items are KEY:VALUE pairs on the node itself, returned either as individual
  properties (e.g. fields prefixed with `s.`) or as a "line_items" dict (when the query
  used RETURN properties(s)). Read the dict directly, formatting each entry as
  "Metric Name: value <unit>" and converting snake_case keys to readable labels.

- CRITICAL — disambiguating consolidated vs per-entity statement-node keys:
  A statement node can hold BOTH a generic key (e.g. `net_income`, `total_revenues`)
  AND per-entity prefixed variants of the same metric whose snake_case starts with
  the subsidiary's or segment's name (e.g. `<entity_snake_case>_net_income`,
  `<entity_snake_case>_<metric>_attributable_to_<parent_snake_case>`).
    • If the question asks about a SPECIFIC subsidiary or segment, look in the
      line_items dict for a key whose snake_case starts with that entity's name
      and prefer it over the generic metric key. Returning the generic key for an
      entity-specific question means returning the CONSOLIDATED total mislabelled —
      a factual error.
    • If the question is about the consolidated filing company (no specific entity
      named), the generic key IS correct.
    • If the only matching record comes from a GENERATED edge and not from a
      statement-node prefixed key, the GENERATED edge is authoritative.

- When BOTH the statement node and a GENERATED edge return a value for the same
  CONSOLIDATED metric, the statement node is the official 10-K figure; the GENERATED
  edge may be a per-segment contributor.

- KEY SELECTION when a line_items dict has many candidates:
  Prefer the LONGEST matching key — the entity-prefixed full name beats the short
  generic key. Example: when asked about "<entity>'s revenue", prefer
  `<entity_snake_case>_revenue: <value>` over the generic `revenues: <other_value>`.
- Geography nodes may contain cities, states, provinces, or countries mixed together.
  When answering geographic questions, list the locations as returned and qualify them
  appropriately (e.g. "cities and countries") — never present a raw location count
  as a "number of countries" unless the records clearly contain only country-level nodes.
- Format numbers cleanly (prefer "$894 million" over raw "894")
- Use bullet points or sections when the answer is multi-part
- When multiple records describe the same metric but with slightly different values
  (e.g. $1,051M on one page and $1,059M on another), do not list them as separate facts.
  Instead, prefer the record whose node name contains 'Total' or is the most aggregate.
  If values differ materially, state the range honestly.
- Only say data is missing if the graph context is literally empty

ANSWER STYLE — be terse and factual:
- State each fact directly. Do NOT add interpretation, market commentary, business
  reasoning, or causal explanation that is not literally present in the retrieved records.
- Format: "<entity> reported <metric> of <value> in <year> (citation)."
- One sentence per fact. No editorial framing.
- If the records do not contain a "why" or "how", do NOT speculate. Say the records
  do not address that aspect.

CITATION RULES — only cite what is literally in the context:
- For GENERATED-edge records (where r.page_number is present):
  cite as (Graph, p.<N>) using the exact page_number value from the record. Never invent.
- For statement-node records (properties on IncomeStatement / BalanceSheet / CashFlow):
  cite as (Graph, <s.name>) — read the actual node name from the s.name field
  (typical format: <StatementType>_FY<year>). DO NOT invent a page number for these;
  these nodes aggregate data from multiple source pages of the document.
- For vector passages (when the context comes from raw document chunks):
  cite as (Source, p.<N>) using the page number from the passage header. Never invent.
- If no page number or node name is available, omit the citation rather than fabricate.
- NEVER cite a page number that does not literally appear in the retrieved context.
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
    """Execute a single Cypher query and return serialized result lines.

    Dict-valued columns (e.g. `properties(s) AS line_items` from a statement-node query)
    are EXPANDED into one key:value line each, so the synthesis layer can read each
    line item directly instead of having to parse a giant inline Python-repr string.
    """
    with driver.session() as session:
        records = [dict(r) for r in session.run(cypher)]

    lines: list[str] = []
    for rec in records:
        scalar_parts: list[str] = []
        dict_blocks:  list[str] = []
        for k, v in rec.items():
            if v is None:
                continue
            # Neo4j Node objects expose .labels and .items() — treat as a node, get name
            if hasattr(v, "labels"):
                v = dict(v).get("name", str(v))
                scalar_parts.append(f"{k}: {v}")
                continue
            # Plain dict (typically from properties(s) AS line_items) — expand each entry
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if vv is None or str(vv).strip() == "":
                        continue
                    dict_blocks.append(f"  {kk}: {vv}")
                continue
            scalar_parts.append(f"{k}: {v}")
        if scalar_parts or dict_blocks:
            header = " | ".join(scalar_parts) if scalar_parts else ""
            if dict_blocks:
                lines.append(header + ("\n" if header else "") + "\n".join(dict_blocks))
            else:
                lines.append(header)
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
    # Records are joined with a blank line so callers (synthesis, eval) can split
    # on "\n\n" to recover individual record boundaries. Without this, multi-line
    # statement records (header + indented line items) get torn apart when split
    # on single "\n", which destroys evidence cohesion for the RAGAS judge.
    return "\n\n".join(deduped)


# ══════════════════════════════════════════════════════
# STEP 2c — VECTOR FALLBACK FETCH
# ══════════════════════════════════════════════════════
def _vector_fetch(question: str) -> str:
    """
    Fallback retrieval — embed the question with the same model used at ingestion
    and search Qdrant for the top-K most relevant chunks. Returns a plain-text
    block of passages with `[section page=N score=X]` headers.

    Used only when the graph returns no records, so a typical structured question
    never pays the embedding or Qdrant cost.
    """
    _ensure_vector_clients()
    vector = _embed_model.encode(question).tolist()

    try:
        results = _qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=VECTOR_TOP_K,
            with_payload=True,
        )
    except Exception as e:
        print(f"  [Vector] Qdrant query failed: {e}")
        return ""

    hits = results.points
    if not hits:
        print("  [Vector] No passages found")
        return ""

    passages = []
    for hit in hits:
        section = hit.payload.get("section",     "unknown section")
        page    = hit.payload.get("page_number", "?")
        text    = hit.payload.get("text",        "").strip()
        score   = round(hit.score, 3)
        passages.append(f"[{section}  page={page}  score={score}]\n{text}")

    print(f"  [Vector] {len(passages)} passage(s) retrieved (fallback)")
    return "\n\n---\n\n".join(passages)


# ══════════════════════════════════════════════════════
# STEP 3 — SYNTHESIS
# ══════════════════════════════════════════════════════
def _synthesize(question: str, graph_ctx: str, vector_ctx: str = "") -> str:
    """
    Generate a natural-language answer. If `vector_ctx` is non-empty, the synthesis
    operates on raw document passages and cites pages from passage headers
    (`(Source, p.<N>)`) instead of from graph records.
    """
    if vector_ctx:
        context = vector_ctx
        user_msg = (
            f"Question: {question}\n\n"
            "No facts were found in the knowledge graph for this question. "
            "Below are the most relevant passages retrieved from the source document. "
            "Answer ONLY from these passages — do not invent figures. Cite the page "
            "number from each passage's header as (Source, p.<N>). If the passages do "
            "not contain a clear answer, say so honestly.\n\n"
            f"Passages:\n{context}"
        )
    else:
        context = graph_ctx if graph_ctx else "No relevant data was found in the knowledge graph."
        user_msg = f"Question: {question}\n\nGraph data:\n{context}"

    response = client_oai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0,
        max_tokens=1500,
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════
# PUBLIC — ANSWER A QUESTION
# ══════════════════════════════════════════════════════
def answer(question: str, driver, entity_catalog: str = "") -> str:
    """
    Hybrid retrieval pipeline:
      1. Try the graph first (Cypher generation + execution).
      2. If the graph returns no records, fall back to vector retrieval from Qdrant.
      3. Synthesise an answer from whichever context is available.

    Vector is a true fallback — it never runs when the graph already answered.
    """
    print(f"\n{'═' * 60}")
    print(f"Question: {question}")
    print(f"{'─' * 60}")

    graph_ctx = _graph_fetch(question, driver, entity_catalog)

    vector_ctx = ""
    if not graph_ctx:
        print("  [Fallback] Graph returned no records — querying vector store...")
        vector_ctx = _vector_fetch(question)

    print(f"{'─' * 60}")
    return _synthesize(question, graph_ctx, vector_ctx)


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
            rec = session.run(
                "MATCH (p:Parent)-[:PARENT_OF]->() RETURN p.name AS name LIMIT 1"
            ).single()
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
