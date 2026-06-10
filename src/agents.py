import json
import time
from openai import OpenAI
from logger import warn, info, log_error

MODEL                = "gpt-4.1-mini"
SLEEP_BETWEEN_AGENTS = 0.3
MAX_FI_PER_AGENT3    = 40

# ══════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════



ENTITY_FINDER_PROMPT = """You are reading a chunk of a financial document.
Find ALL potentially relevant named entities in the text.

Be PERMISSIVE — include anything that might be:
- A company or organization name
- A geographic location (country, state, city)
- A specific dollar value or percentage (e.g. $8,589, $1.2 billion, 26.6%)
- A raw number or percentage from a financial table row (e.g. 75.0, 25.0%, 100.0)
- A named business division or segment

Do NOT classify or filter — just find and list them as they appear in the text.

Return ONLY valid JSON:
{"candidates": ["entity1", "entity2", ...]}
If nothing found, return: {"candidates": []}"""


def build_classifier_prompt(filing_company: str, known_entities: list[dict] | None = None) -> str:
    known_lines = "\n".join(
        f"  - {e['name']} ({e['type']})"
        for e in (known_entities or [])
    ) or "  None yet."

    return f"""You are classifying entity candidates from a financial document.
The filing company is: {filing_company}
"we", "our", "the Company" refer to {filing_company}.

Previously confirmed entities in this document:
{known_lines}
If a candidate refers to the same entity as one above — even by abbreviation or short form —
you MUST output the EXACT confirmed name from the list above.
Never output a shortened form, partial name, or abbreviation if the full canonical name
is already confirmed. For example: if "CNA Financial Corporation" is confirmed,
output "CNA Financial Corporation" — never "CNA" or "CNA Financial".

Assign exactly one type below — or DISCARD. Omit discarded candidates entirely.

Parent:
  Only the exact legal name "{filing_company}" — character for character. At most one per document.
  If you are not certain the candidate is exactly "{filing_company}", discard it.
  Discard generic references ("the Company", "the Registrant") and any other company.

Subsidiary:
  A company currently owned or controlled by {filing_company}, with explicit ownership
  language nearby: "owns", "wholly-owned subsidiary of", "controlled by", "acquired".
  Discard suppliers, foundries, partners, terminated acquisition targets, and
  peer/competitor companies listed in stock performance comparison tables
  (e.g. "the following graph compares... against a peer group of: Chubb, Travelers...").

  IMPORTANT EXCEPTION — already-confirmed subsidiaries:
  If the candidate matches — even by short form, abbreviation, or operating name —
  an entity already listed under "Previously confirmed entities" above, you MUST
  classify it as Subsidiary using the EXACT confirmed name from that list.
  No ownership language is required for already-confirmed subsidiaries.
  Examples:
    "Loews Hotels & Co"    → confirmed "Loews Hotels Holding Corporation" → output "Loews Hotels Holding Corporation"
    "CNA"                  → confirmed "CNA Financial Corporation"        → output "CNA Financial Corporation"
    "Diamond Offshore"     → confirmed "Diamond Offshore Drilling, Inc."  → output "Diamond Offshore Drilling, Inc."
    "Boardwalk Pipelines"  → confirmed "Boardwalk Pipeline Partners, LP"  → output "Boardwalk Pipeline Partners, LP"

Geography:
  A specific country, state, province, or city — nothing broader or vaguer.
  Even if a company explicitly operates in a region, sea, or bloc, do NOT extract it —
  only extract the specific countries or cities within it.
  Valid examples: United States, France, Illinois, London, Brazil, Malaysia.
  Discard anything that is not a country/state/province/city:
  — regions, sub-regions, seas, and bodies of water (e.g. South America,
    Southeast Asia, East Africa, Mediterranean, Gulf of Mexico)
  — trade blocs and unions (e.g. European Union, E.U., EMEA, APAC)
  — regulatory bodies and institutions (e.g. European Commission, NAIC, SEC,
    IAIS, ComFrame, U.S. Treasury, Federal Reserve)
  — ANY name containing the words: Department, Authority, Commission,
    Association, Commissioners, Supervisors, Superintendent, Monetary,
    Institute, Board, Bureau, Agency, Committee, Council, Office of —
    these are institutions, NOT locations, even if they contain a geographic
    word. Examples to DISCARD:
    "State of Illinois Department of Insurance" (contains "Department")
    "Office of Superintendent of Financial Institutions in Canada" (contains "Office of")
    "Bermuda Monetary Authority" (contains "Monetary Authority")
    "National Association of Insurance Commissioners" (contains "Association")
    "Bermuda Monetary Authority" → discard, extract "Bermuda" only if
    the text separately mentions Bermuda as a place of operations.
  — currencies, street addresses, postal codes, and generic placeholders
    such as "Other", "Worldwide", "Rest of World"
  — names combined with legal or governmental modifiers such as "law",
    "government", "agency", "regulation", "court", "jurisdiction" —
    these describe legal contexts, not physical locations

Business Segment:
  A segment {filing_company} reports under GAAP, with "reportable segment", "operating
  segment", or "business segment" immediately adjacent to the name — OR as a labeled
  row in a table whose header contains "segment".
  Discard product names, end-market categories, and segments of other companies.
  Never classify a legal entity name as a Business Segment — if it is a company,
  it is a Subsidiary or Parent, not a Business Segment.
  Output the segment name only — never append descriptive words to it.
  WRONG: "Compute & Networking segment", "CNA revenue", "Boardwalk Pipeline income"
  CORRECT: "Compute & Networking", "CNA", "Boardwalk Pipeline"

Financial Item:
  A specific numeric value from a financial reporting context — must contain a digit.
  If the candidate contains no digit whatsoever, it is a label — discard it immediately.
  Examples of labels to discard: "Research and development", "Basic", "Diluted", "CCPA", "GDPR".
  Extract from financial statements, their accompanying notes, and business description
  sections where specific monetary values are stated — backlog figures, credit facility
  amounts, contract values, and debt amounts qualify even outside formal statements.
  Extract exactly as they appear (do not convert or reformat).
  If a table header says "(In millions)" and a cell shows "8,589", extract "8,589".

  Discard:
  - Values in rows labeled "per share", "Basic", "Diluted", or "EPS"
  - Non-dollar share quantities: values with no $ sign where the row or column unit is
    shares, units, options, RSUs, or PSUs. Dollar amounts in the same table must still
    be extracted — a $ value is always a financial figure regardless of adjacent share columns.
  - Dates and calendar references
  - Values from a stock return comparison chart (multi-year cumulative return tables
    where a hypothetical investment is indexed to a base value at the start of the period)
  - Change columns ($Change, %Change) — period columns only
  - Any value appearing exclusively in a non-financial context: workforce headcount,
    diversity statistics, sustainability disclosures, or ESG targets. If the same value
    also appears in a financial table or note, keep it.
  - Non-monetary business metrics: counts of products, users, systems, or customers
  - Any candidate that is a label, identifier, or reference rather than an actual
    numeric measurement — a Financial Item must represent an amount, rate, or quantity.
    If it does not start with a digit, $, or ( and contains no % or scale word
    (billion/million/trillion/thousand), discard it.

Return ONLY valid JSON:
{{"entities": [{{"name": "entity name", "type": "entity type"}}]}}
If all discarded, return: {{"entities": []}}"""


def build_structural_prompt(filing_company: str) -> str:
    """Segment 1 — structural/ownership relationships (no Financial Items involved)."""
    return f"""You are finding STRUCTURAL relationships between validated entities in a financial document.
The filing company is: {filing_company}

You will receive a list of validated entities (Parent, Subsidiary, Geography, Business Segment)
and the original text. There are NO Financial Items in this task.

CRITICAL RULES:
1. source and target MUST be taken verbatim from the entity list. Never invent a name.
2. Only extract what is explicitly stated in the text — never infer.
3. If no valid relationship can be formed, return an empty list.

════════════════════════════════════════════
RELATIONSHIP TYPES
════════════════════════════════════════════

PARENT_OF — Parent legally owns or controls a Subsidiary
  source: Parent  |  target: Subsidiary
  Only with explicit ownership language: "owns", "wholly-owned subsidiary of",
  "controlled by", "acquired". Do not extract for failed or terminated acquisitions.

OPERATES_IN — Legal entity has physical presence in a Geography
  source: Parent or Subsidiary  |  target: Geography
  Only extract when the text contains explicit physical presence language — headquarters,
  offices, facilities, or place of incorporation — tied to that location.
  Do NOT extract for: revenue breakdowns, customer locations, workforce statistics,
  export controls, trade restrictions, sanctions, or legal jurisdiction references.

════════════════════════════════════════════
RULES
════════════════════════════════════════════
- Only extract what is explicitly stated — never infer
- When in doubt, do NOT extract

Return ONLY valid JSON:
{{"relationships": [{{"source": "name", "target": "name", "type": "TYPE", "property": null}}]}}
If none found, return: {{"relationships": []}}"""


def build_financial_prompt(filing_company: str) -> str:
    """Segment 2 — consolidated and segment-level financial reporting (REPORTED, GENERATED)."""
    return f"""You are finding FINANCIAL REPORTING relationships between validated entities in a financial document.
The filing company is: {filing_company}

You will receive a list of validated entities and the original text.

CRITICAL RULES:
1. source and target MUST be taken verbatim from the entity list. Never invent a name.
2. The target must be a standalone numeric value containing a digit, taken exactly as it
   appears verbatim in the text. NEVER combine a row label with a value —
   "Net income $72,880" is WRONG; use "$72,880" as target and "net income fy2025" as property.
3. Only use values that appear as standalone numbers in the text — never invent or reformat.
4. If no valid relationship can be formed, return an empty list.

════════════════════════════════════════════
RELATIONSHIP TYPES
════════════════════════════════════════════

REPORTED — Filing company reports a consolidated financial figure
  source: Parent  |  target: Financial Item
  EXCLUSIVELY for {filing_company} (the Parent). NEVER use REPORTED if the source is a Subsidiary.
  For income statement, balance sheet, and cash flow line items at consolidated level.
  Do not extract segment-level figures (use GENERATED instead), per-share values, or share counts.
  When the text presents a subsidiary's own financial statements or a segment table
  (e.g. a table headed with the subsidiary's name), ALL financial line items in that table
  belong to that subsidiary via GENERATED — do NOT use REPORTED even if
  {filing_company} appears in the entity list.
  Store metric name and period in property: e.g. "total revenue fy2025".

GENERATED — A segment or subsidiary produced a revenue or income figure
  source: Business Segment or Subsidiary  |  target: Financial Item
  Only for segment- or subsidiary-level revenue, sales, income, or margin in a financial table.
  Do not extract expenses, costs, or change percentages.
  Do not assign company-level consolidated totals to a segment or subsidiary — the target must
  come from a segment- or subsidiary-specific row, not a total or sum row of the table.
  Store metric name and period in property: e.g. "data center revenue fy2025".

════════════════════════════════════════════
RULES
════════════════════════════════════════════
- Only extract what is explicitly stated — never infer
- Write property values in lowercase with fiscal period at the end:
  e.g. "total revenue fy2025", "net income fy2024"

Return ONLY valid JSON:
{{"relationships": [{{"source": "name", "target": "name", "type": "TYPE", "property": "value or null"}}]}}
If none found, return: {{"relationships": []}}"""


def build_metric_prompt(filing_company: str) -> str:
    """Segment 3 — supplementary financial metrics (HAS_METRIC)."""
    return f"""You are finding SUPPLEMENTARY METRIC relationships between validated entities in a financial document.
The filing company is: {filing_company}

You will receive a list of validated entities and the original text.

CRITICAL RULES:
1. source and target MUST be taken verbatim from the entity list. Never invent a name.
2. The target must be a standalone numeric value containing a digit, taken exactly as it
   appears verbatim in the text. NEVER combine a row label with a value —
   "Raw materials $1,200" is WRONG; use "$1,200" as target and "raw materials inventory fy2025" as property.
3. Only use values that appear as standalone numbers in the text — never invent or reformat.
4. If the entity list contains only the Parent and no Financial Items, read the text directly
   and extract every financial value you find as a HAS_METRIC. Use each dollar amount or
   numeric figure verbatim from the text as the target and describe the metric in property.

════════════════════════════════════════════
RELATIONSHIP TYPE
════════════════════════════════════════════

HAS_METRIC — Filing company or subsidiary discloses a supplementary financial figure
  source: Parent or Subsidiary  |  target: Financial Item
  DEFAULT fallback for Financial Items that do not qualify as REPORTED or GENERATED.
  Always prefer HAS_METRIC over skipping — never leave a Financial Item unlinked.
  Use for notes to financial statements: debt schedules, lease obligations,
  inventory breakdowns, stock-based compensation, acquisition costs, etc.
  Do NOT use for consolidated income statement / balance sheet totals (use REPORTED)
  or for segment/subsidiary revenue figures (use GENERATED).
  Store metric name and period in property: e.g. "raw materials inventory fy2025".

════════════════════════════════════════════
RULES
════════════════════════════════════════════
- Only extract what is explicitly stated — never infer
- Always prefer HAS_METRIC over skipping a Financial Item
- Write property values in lowercase with fiscal period at the end:
  e.g. "long-term debt fy2025", "operating lease liability fy2024"

Return ONLY valid JSON:
{{"relationships": [{{"source": "name", "target": "name", "type": "HAS_METRIC", "property": "value or null"}}]}}
If none found, return: {{"relationships": []}}"""


# ══════════════════════════════════════════════════════
# PUBLIC — PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════

def run_pipeline(text: str, filing_company: str, client: OpenAI,
                 guard_fn, run_timestamp: str, chunk_id: int,
                 known_entities: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    """
    Orchestrate the full 3-agent extraction pipeline.
      guard_fn : callable(entities) -> entities, applied after Agent 2.
    Returns (entities, relationships).
    """
    valid_types = {"Parent", "Subsidiary", "Geography", "Business Segment", "Financial Item"}

    # ── Agent 1 — entity finder ──────────────────────────────────────────────
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ENTITY_FINDER_PROMPT},
            {"role": "user",   "content": text}
        ],
        temperature=0,
        max_tokens=4096,
        response_format={"type": "json_object"}
    )
    try:
        candidates = json.loads(response.choices[0].message.content).get("candidates", [])
    except json.JSONDecodeError:
        candidates = []

    print(f"  Agent 1  -  candidates : {len(candidates)}  {candidates}")
    time.sleep(SLEEP_BETWEEN_AGENTS)

    if not candidates:
        print(f"  No candidates  -  skipping agents 2 & 3")
        return [], []

    # ── Agent 2 — entity classifier ──────────────────────────────────────────
    # Deduplicate known_entities by name before injecting into the classifier prompt.
    # Also exclude Financial Items — they are raw numbers ($72,880, 8,589) that never
    # need canonical name resolution, so they only add noise and waste tokens.
    seen_names: set[str] = set()
    deduped_known: list[dict] = []
    for e in (known_entities or []):
        if e["type"] != "Financial Item" and e["name"] not in seen_names:
            seen_names.add(e["name"])
            deduped_known.append(e)

    user_content = (
        f"Candidates to classify:\n{json.dumps(candidates)}\n\n"
        f"Original text for context:\n{text}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": build_classifier_prompt(filing_company, deduped_known)},
            {"role": "user",   "content": user_content}
        ],
        temperature=0,
        max_tokens=8192,
        response_format={"type": "json_object"}
    )
    try:
        raw_entities = json.loads(response.choices[0].message.content).get("entities", [])
    except json.JSONDecodeError:
        raw_entities = []

    # safety filter — drop any entity the model labelled DISCARD instead of omitting,
    # and drop any entity with an unknown/hallucinated type
    entities = [e for e in raw_entities
                if e.get("type", "").upper() != "DISCARD"
                and e.get("type") in valid_types]

    # post-processing guards (from extractor.py)
    entities = guard_fn(entities)

    print(f"  Agent 2  -  entities   : {len(entities)}")
    for e in entities:
        print(f"    ->[{e['type']}] {e['name']}")
    time.sleep(SLEEP_BETWEEN_AGENTS)

    if not entities:
        return [], []

    # ── Parent injection — ensure filing company is present for Agent 3 ──────
    entities_for_rel = entities
    if not any(e["type"] == "Parent" for e in entities):
        entities_for_rel = [{"name": filing_company, "type": "Parent"}] + entities
        info(f"Parent injected for Agent 3: {filing_company}")

    # ── Agent 3 — relationship finder (3 focused segments) ───────────────────
    non_fi  = [e for e in entities_for_rel if e["type"] != "Financial Item"]
    fi_only = [e for e in entities_for_rel if e["type"] == "Financial Item"]

    def _call_segment(prompt: str, entity_batch: list[dict]) -> list[dict]:
        """Single Agent 3 call for one relationship segment."""
        if not entity_batch:
            return []
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": (
                    f"Validated entities:\n{json.dumps(entity_batch)}\n\n"
                    f"Original text:\n{text}"
                )}
            ],
            temperature=0,
            max_tokens=8192,
            response_format={"type": "json_object"}
        )
        try:
            rels = json.loads(response.choices[0].message.content).get("relationships", [])
        except json.JSONDecodeError:
            return []
        # self-referential guard
        return [r for r in rels if r.get("source") != r.get("target")]

    def _call_with_fi_batching(prompt: str, segment_label: str) -> list[dict]:
        """
        Run a segment prompt that involves Financial Items.
        If fi_only fits within MAX_FI_PER_AGENT3, one call.
        Otherwise split fi_only into batches and merge.
        """
        results = []
        if len(fi_only) <= MAX_FI_PER_AGENT3:
            try:
                results = _call_segment(prompt, non_fi + fi_only)
            except Exception as err:
                warn(f"Agent 3 [{segment_label}] failed: {err}")
                log_error(run_timestamp, chunk_id, f"Agent 3 [{segment_label}] error: {err}")
        else:
            batch_size  = max(MAX_FI_PER_AGENT3 - len(non_fi), 1)
            num_batches = (len(fi_only) + batch_size - 1) // batch_size
            info(f"Batching Agent 3 [{segment_label}]: {len(fi_only)} FIs -> {num_batches} call(s)")
            for b_idx in range(0, len(fi_only), batch_size):
                try:
                    batch_rels = _call_segment(prompt, non_fi + fi_only[b_idx:b_idx + batch_size])
                    results.extend(batch_rels)
                except Exception as err:
                    warn(f"Agent 3 [{segment_label}] batch failed  -  skipping batch: {err}")
                    log_error(run_timestamp, chunk_id, f"Agent 3 [{segment_label}] batch error: {err}")
                time.sleep(SLEEP_BETWEEN_AGENTS)
        return results

    relationships = []

    # ── Segment 1: structural (PARENT_OF, OPERATES_IN)
    # No Financial Items needed — pass only non-FI entities.
    # Skip entirely if non_fi has only the Parent — all structural relationship types
    # require at least a Subsidiary, Geography, or Business Segment as target.
    has_structural_targets = any(
        e["type"] in ("Subsidiary", "Geography", "Business Segment")
        for e in non_fi
    )
    if has_structural_targets:
        try:
            seg1 = _call_segment(build_structural_prompt(filing_company), non_fi)
            info(f"Agent 3 [structural]   : {len(seg1)} relationship(s)")
            relationships.extend(seg1)
        except Exception as err:
            warn(f"Agent 3 [structural] failed: {err}")
            log_error(run_timestamp, chunk_id, f"Agent 3 [structural] error: {err}")
        time.sleep(SLEEP_BETWEEN_AGENTS)
    else:
        info("Agent 3 [structural]   : skipped (no Subsidiary / Geography / Business Segment)")

    # ── Segment 2: financial reporting (REPORTED, GENERATED)
    # ── Segment 3: supplementary metrics (HAS_METRIC)
    # Both require Financial Items as targets — skip entirely if none exist.
    if fi_only:
        seg2 = _call_with_fi_batching(build_financial_prompt(filing_company), "financial")
        info(f"Agent 3 [financial]    : {len(seg2)} relationship(s)")
        relationships.extend(seg2)
        time.sleep(SLEEP_BETWEEN_AGENTS)

        seg3 = _call_with_fi_batching(build_metric_prompt(filing_company), "metric")
        info(f"Agent 3 [metric]       : {len(seg3)} relationship(s)")
        relationships.extend(seg3)
    else:
        info("Agent 3 [financial/metric]: skipped (no Financial Items in this chunk)")

    # ── Seg2 / Seg3 overlap guard ─────────────────────────────────────────────
    # REPORTED and GENERATED are semantically stronger than HAS_METRIC.
    # If the same (source, target) pair was already claimed by REPORTED or GENERATED
    # in segment 2, drop any HAS_METRIC from segment 3 pointing to the same pair.
    strong_pairs = {
        (r["source"], r["target"])
        for r in relationships
        if r["type"] in ("REPORTED", "GENERATED")
    }
    before_overlap = len(relationships)
    relationships = [
        r for r in relationships
        if not (r["type"] == "HAS_METRIC" and (r["source"], r["target"]) in strong_pairs)
    ]
    if len(relationships) < before_overlap:
        info(f"Dropped {before_overlap - len(relationships)} HAS_METRIC(s) already covered by REPORTED/GENERATED")

    # Deduplicate relationships — batched Agent 3 calls can produce identical entries
    seen_rels: set[tuple] = set()
    deduped: list[dict]   = []
    for r in relationships:
        key = (r.get("source"), r.get("target"), r.get("type"), r.get("property"))
        if key not in seen_rels:
            seen_rels.add(key)
            deduped.append(r)
    if len(deduped) < len(relationships):
        info(f"Deduped {len(relationships) - len(deduped)} duplicate relationship(s)")
    relationships = deduped

    print(f"  Agent 3  -  relationships: {len(relationships)}")
    for rel in relationships:
        print(f"    ->{rel['source']} --{rel['type']} -->{rel['target']}  (property: {rel.get('property')})")

    return entities, relationships
