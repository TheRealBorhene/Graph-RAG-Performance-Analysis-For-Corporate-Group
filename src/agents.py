import time
from openai import OpenAI
from logger import info
from entity_finder import find_entities
from entity_classifier import classify_entities
from relationship_finder import find_relationships
from statement_extractor import extract_statements

MODEL                = "gpt-4.1-mini"
SLEEP_BETWEEN_AGENTS = 0.3


def run_pipeline(text: str, filing_company: str, client: OpenAI,
                 guard_fn, run_timestamp: str, chunk_id: int,
                 known_entities: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    """
    Orchestrate the full extraction pipeline.
      Agent 1 — entity finder       : raw candidate strings
      Agent 2 — entity classifier   : typed entities (Parent / Subsidiary / Geography / Business Segment / Financial Item)
      Agent 3 — relationship finder : PARENT_OF, OPERATES_IN, GENERATED
      Statement extractor           : BALANCE_SHEET / INCOME_STATEMENT / CASH_FLOW nodes

    guard_fn : callable(entities) -> entities, applied after Agent 2.
    Returns (entities, relationships) where both lists may include statement nodes and their edges.
    """
    valid_types = {"Parent", "Subsidiary", "Geography", "Business Segment", "Financial Item"}

    # ── Agent 1 — entity finder ──────────────────────────────────────────────
    candidates = find_entities(client, text)
    print(f"  Agent 1  |  candidates ({len(candidates)}):")
    for c in candidates:
        print(f"             - {c}")
    time.sleep(SLEEP_BETWEEN_AGENTS)

    if not candidates:
        print("  No candidates — skipping agents 2 & 3")
        return [], []

    # ── Agent 2 — entity classifier ──────────────────────────────────────────
    raw_entities = classify_entities(client, candidates, text, filing_company, known_entities)
    entities = [e for e in raw_entities
                if e.get("type", "").upper() != "DISCARD"
                and e.get("type") in valid_types]
    entities = guard_fn(entities)

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

    # ── Agent 3 — relationship finder (structural + GENERATED) ───────────────
    relationships = find_relationships(
        client, text, filing_company, entities_for_rel, run_timestamp, chunk_id
    )
    print(f"  Agent 3  |  relationships ({len(relationships)}):")
    for rel in relationships:
        print(f"             {rel['source']}  --[{rel['type']}]-->  {rel['target']}  (property: {rel.get('property')})")

    # ── Statement extractor — balance sheet / income statement / cash flow ────
    stmt_entities, stmt_rels = extract_statements(client, text, filing_company)
    print(f"  Stmts    |  statement nodes ({len(stmt_entities)}):")
    for se in stmt_entities:
        print(f"             [{se['type']}]  {se['name']}  ({len(se.get('properties', {}))} items)")
    for sr in stmt_rels:
        print(f"             {sr['source']}  --[{sr['type']}]-->  {sr['target']}")

    return entities + stmt_entities, relationships + stmt_rels
