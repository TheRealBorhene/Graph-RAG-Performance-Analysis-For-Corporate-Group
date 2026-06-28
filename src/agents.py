import time
from openai import OpenAI
from logger import info
from entity_finder import find_entities
from entity_classifier import classify_entities
from relationship_finder import find_relationships
from statement_extractor import extract_statements
from guards import promote_financial_items, find_confirmed_subs_in_text

MODEL                = "gpt-4.1-mini"
SLEEP_BETWEEN_AGENTS = 0.3


def run_pipeline(text: str, filing_company: str, client: OpenAI,
                 guard_fn, run_timestamp: str, chunk_id: int,
                 known_entities: list[dict] | None = None,
                 trace: dict | None = None) -> tuple[list[dict], list[dict]]:
    """
    Orchestrate the full extraction pipeline.
      Agent 1 — entity finder       : raw candidate strings
      Agent 2 — entity classifier   : typed entities (Parent / Subsidiary / Geography / Business Segment / Financial Item)
      Agent 3 — relationship finder : PARENT_OF, OPERATES_IN, GENERATED
      Statement extractor           : BALANCE_SHEET / INCOME_STATEMENT / CASH_FLOW nodes

    guard_fn : callable(entities) -> entities, applied after Agent 2.

    trace : optional dict. When provided, it is populated with the raw output of
            every API call so callers (e.g. the logger) can record the full pipeline:
              candidates       — Agent 1 raw candidate strings
              raw_entities     — Agent 2 output before type filter + guards
              entities         — typed entities after filter + guards
              relationships    — Agent 3 relationships
              statement_nodes  — statement entities (with their line items)
              statement_rels   — statement edges

    Returns (entities, relationships) where both lists may include statement nodes and their edges.
    """
    valid_types = {"Parent", "Subsidiary", "Geography", "Business Segment", "Financial Item", "Person"}

    # Initialise trace fields up-front so early returns still leave a complete record.
    if trace is not None:
        trace.setdefault("candidates",      [])
        trace.setdefault("raw_entities",    [])
        trace.setdefault("entities",        [])
        trace.setdefault("relationships",   [])
        trace.setdefault("statement_nodes", [])
        trace.setdefault("statement_rels",  [])

    # ── Agent 1 — entity finder ──────────────────────────────────────────────
    candidates = find_entities(client, text)
    if trace is not None:
        trace["candidates"] = list(candidates)
    print(f"  Agent 1  |  candidates ({len(candidates)}):")
    for c in candidates:
        print(f"             - {c}")
    time.sleep(SLEEP_BETWEEN_AGENTS)

    if not candidates:
        print("  No candidates — skipping agents 2 & 3")
        return [], []

    # ── Agent 2 — entity classifier ──────────────────────────────────────────
    raw_entities = classify_entities(client, candidates, text, filing_company, known_entities)
    if trace is not None:
        trace["raw_entities"] = list(raw_entities)
    entities = [e for e in raw_entities
                if e.get("type", "").upper() != "DISCARD"
                and e.get("type") in valid_types]

    # Deterministic recovery of Financial Items Agent 2 dropped (see guards.promote_financial_items).
    # Runs BEFORE the guards so promoted items are still filtered by the FI guards, and
    # before Agent 3 so it has the values available to link as GENERATED targets.
    promoted = promote_financial_items(candidates, entities)
    if promoted:
        info(f"Promoted {len(promoted)} Financial Item(s) Agent 2 did not classify")
        entities = entities + promoted

    entities = guard_fn(entities)
    if trace is not None:
        trace["entities"] = entities

    print(f"  Agent 2  |  entities ({len(entities)}):")
    for e in entities:
        print(f"             [{e['type']}]  {e['name']}")
    time.sleep(SLEEP_BETWEEN_AGENTS)

    if not entities:
        return [], []

    # ── Parent injection — ensure filing company is present for Agent 3 ──────
    entities_for_rel = entities
    if not any(e["type"] == "Parent" for e in entities):
        entities_for_rel = [{"name": filing_company, "type": "Parent"}] + entities
        info(f"Parent injected for Agent 3: {filing_company}")

    # ── Subsidiary re-injection — recover a GENERATED source when Agent 2 starved Agent 3 ──
    # If this chunk has Financial Items but no Subsidiary/Business Segment, Agent 3's GENERATED
    # step would skip for lack of a source (the chunk-53 collapse). Re-inject any confirmed
    # subsidiary whose leading token appears in the chunk text so Agent 3 can attribute the
    # values to the canonical owner. Grounding + drop_orphaned_financial_items filter any
    # over-match. Derived from the confirmed-subsidiary list — nothing hardcoded.
    has_source = any(e["type"] in ("Subsidiary", "Business Segment") for e in entities)
    has_fi     = any(e["type"] == "Financial Item" for e in entities)
    if has_fi and not has_source and known_entities:
        confirmed_subs = [e["name"] for e in known_entities if e["type"] == "Subsidiary"]
        present  = {e["name"] for e in entities_for_rel}
        injected = [s for s in find_confirmed_subs_in_text(text, confirmed_subs) if s not in present]
        if injected:
            entities_for_rel = entities_for_rel + [{"name": s, "type": "Subsidiary"} for s in injected]
            info(f"Re-injected {len(injected)} confirmed subsidiary(ies) for Agent 3: {injected}")

    # ── Agent 3 — relationship finder (structural + GENERATED) ───────────────
    relationships = find_relationships(
        client, text, filing_company, entities_for_rel, run_timestamp, chunk_id
    )
    if trace is not None:
        trace["relationships"] = relationships
    print(f"  Agent 3  |  relationships ({len(relationships)}):")
    for rel in relationships:
        print(f"             {rel['source']}  --[{rel['type']}]-->  {rel['target']}  (property: {rel.get('property')})")

    # ── Statement extractor — balance sheet / income statement / cash flow ────
    stmt_entities, stmt_rels = extract_statements(client, text, filing_company)
    if trace is not None:
        trace["statement_nodes"] = stmt_entities
        trace["statement_rels"]  = stmt_rels
    print(f"  Stmts    |  statement nodes ({len(stmt_entities)}):")
    for se in stmt_entities:
        print(f"             [{se['type']}]  {se['name']}  ({len(se.get('properties', {}))} items)")
    for sr in stmt_rels:
        print(f"             {sr['source']}  --[{sr['type']}]-->  {sr['target']}")

    return entities + stmt_entities, relationships + stmt_rels
