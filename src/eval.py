"""
Graph-only evaluation for the financial knowledge graph.

Three dimensions measured — no vector fetch, no RAGAS, no external LLM scoring:

  1. Structural completeness  — are the expected entities and relationships present?
  2. Financial accuracy       — do key figures match known ground-truth values?
  3. Query coverage           — do benchmark questions return non-empty graph results?
"""

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase
from query1 import connect_neo4j, fetch_entity_catalog, _graph_fetch

load_dotenv()

# ─────────────────────────────────────────────
# GROUND TRUTH  (Loews Corporation 10-K FY2019)
# ─────────────────────────────────────────────

# Expected subsidiaries — all must be present as Subsidiary nodes
EXPECTED_SUBSIDIARIES = [
    "CNA Financial Corporation",
    "Diamond Offshore Drilling, Inc.",
    "Boardwalk Pipeline Partners, LP",
    "Loews Hotels Holding Corporation",
    "Altium Packaging LLC",
]

# Expected PARENT_OF relationships — (parent, subsidiary)
EXPECTED_PARENT_OF = [
    ("Loews Corporation", sub) for sub in EXPECTED_SUBSIDIARIES
]

# Key financial figures — (entity_name, metric_substring, expected_value_substring)
# metric_substring matches against r.property (e.g. "revenues fy2019")
# expected_value_substring matches against fi.name (e.g. "$14,931")
EXPECTED_FINANCIALS = [
    ("Loews Corporation",          "revenues fy2019",                        "$14,931"),
    ("Loews Corporation",          "net income attributable",                 "$932"),
    ("Loews Corporation",          "total assets fy2019",                     "82,243"),
    ("CNA Financial Corporation",  "insurance premiums fy2019",               "$7,428"),
    ("CNA Financial Corporation",  "total revenues fy2019",                   "$10,788"),
    ("Diamond Offshore Drilling",  "contract backlog fy2020",                 "$1.6 billion"),
    ("Boardwalk Pipeline Partners","net income fy2019",                       "$209"),
]

# Expected geographic operations — (entity_name, location_substring)
EXPECTED_LOCATIONS = [
    ("Loews Corporation",          "New York"),
    ("CNA Financial Corporation",  "United Kingdom"),
    ("CNA Financial Corporation",  "Chicago"),
    ("Diamond Offshore Drilling",  "Houston"),
    ("Loews Hotels Holding",       "Orlando"),
    ("Altium Packaging LLC",       "Atlanta"),
]

# Benchmark questions — each must return at least 1 graph record
BENCHMARK_QUESTIONS = [
    "What subsidiaries does Loews Corporation own?",
    "What are the revenues of CNA Financial Corporation?",
    "Where does Diamond Offshore operate?",
    "What is the net income of Loews Corporation?",
    "What is the contract backlog of Diamond Offshore?",
    "Where does Loews Hotels operate?",
    "What are the total assets of Loews Corporation?",
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _run(driver, query: str, **params) -> list[dict]:
    with driver.session() as session:
        return [dict(r) for r in session.run(query, **params)]


def _pass(msg: str):
    print(f"  ✅  {msg}")


def _fail(msg: str):
    print(f"  ❌  {msg}")


def _section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ─────────────────────────────────────────────
# DIMENSION 1 — STRUCTURAL COMPLETENESS
# ─────────────────────────────────────────────

def eval_structure(driver) -> tuple[int, int]:
    passed = failed = 0
    _section("DIMENSION 1 — Structural Completeness")

    # 1a — Parent node present
    print("\n  1a. Parent node")
    rows = _run(driver, "MATCH (p:Parent) RETURN p.name AS name LIMIT 1")
    if rows:
        _pass(f"Parent node found: '{rows[0]['name']}'")
        passed += 1
    else:
        _fail("No Parent node in graph")
        failed += 1

    # 1b — All expected subsidiaries present
    print("\n  1b. Subsidiary nodes")
    existing = {
        r["name"] for r in
        _run(driver, "MATCH (s:Subsidiary) RETURN s.name AS name")
    }
    for sub in EXPECTED_SUBSIDIARIES:
        if sub in existing:
            _pass(f"Subsidiary present: '{sub}'")
            passed += 1
        else:
            _fail(f"Subsidiary MISSING: '{sub}'")
            failed += 1

    # 1c — PARENT_OF relationships
    print("\n  1c. PARENT_OF relationships")
    po_rows = _run(driver,
        "MATCH (p:Parent)-[:PARENT_OF]->(s:Subsidiary) "
        "RETURN p.name AS parent, s.name AS subsidiary"
    )
    found_pairs = {(r["parent"], r["subsidiary"]) for r in po_rows}
    for parent, sub in EXPECTED_PARENT_OF:
        match = any(sub in s for _, s in found_pairs)
        if match:
            _pass(f"PARENT_OF → '{sub}'")
            passed += 1
        else:
            _fail(f"PARENT_OF missing → '{sub}'")
            failed += 1

    # 1d — OPERATES_IN relationships exist
    print("\n  1e. OPERATES_IN relationships")
    oi_count = _run(driver,
        "MATCH ()-[:OPERATES_IN]->(:Geography) RETURN count(*) AS cnt"
    )[0]["cnt"]
    if oi_count > 0:
        _pass(f"OPERATES_IN: {oi_count} relationship(s) found")
        passed += 1
    else:
        _fail("No OPERATES_IN relationships found")
        failed += 1

    return passed, failed


# ─────────────────────────────────────────────
# DIMENSION 2 — FINANCIAL ACCURACY
# ─────────────────────────────────────────────

def eval_financials(driver) -> tuple[int, int]:
    passed = failed = 0
    _section("DIMENSION 2 — Financial Accuracy")

    for entity_substr, metric_substr, value_substr in EXPECTED_FINANCIALS:
        rows = _run(driver,
            "MATCH (e:Entity)-[r]->(fi:FinancialItem) "
            "WHERE toLower(e.name) CONTAINS toLower($entity) "
            "  AND toLower(r.property) CONTAINS toLower($metric) "
            "RETURN fi.name AS value, r.property AS prop "
            "LIMIT 5",
            entity=entity_substr,
            metric=metric_substr,
        )
        found = any(value_substr.lower() in (r["value"] or "").lower() for r in rows)
        label = f"'{entity_substr[:30]}' | {metric_substr[:35]}"
        if found:
            matched_val = next(r["value"] for r in rows if value_substr.lower() in (r["value"] or "").lower())
            _pass(f"{label} → {matched_val}")
            passed += 1
        else:
            retrieved = [r["value"] for r in rows] if rows else ["(no records)"]
            _fail(f"{label} → expected '{value_substr}', got {retrieved[:3]}")
            failed += 1

    return passed, failed


# ─────────────────────────────────────────────
# DIMENSION 3 — GEOGRAPHIC COVERAGE
# ─────────────────────────────────────────────

def eval_geography(driver) -> tuple[int, int]:
    passed = failed = 0
    _section("DIMENSION 3 — Geographic Coverage")

    for entity_substr, location_substr in EXPECTED_LOCATIONS:
        rows = _run(driver,
            "MATCH (e:Entity)-[:OPERATES_IN]->(g:Geography) "
            "WHERE toLower(e.name) CONTAINS toLower($entity) "
            "  AND toLower(g.name) CONTAINS toLower($location) "
            "RETURN g.name AS location LIMIT 1",
            entity=entity_substr,
            location=location_substr,
        )
        label = f"'{entity_substr[:30]}' OPERATES_IN '{location_substr}'"
        if rows:
            _pass(label)
            passed += 1
        else:
            _fail(label)
            failed += 1

    return passed, failed


# ─────────────────────────────────────────────
# DIMENSION 4 — QUERY COVERAGE
# ─────────────────────────────────────────────

def eval_query_coverage(driver, entity_catalog: str) -> tuple[int, int]:
    passed = failed = 0
    _section("DIMENSION 4 — Query Coverage (graph records returned)")

    for question in BENCHMARK_QUESTIONS:
        result = _graph_fetch(question, driver, entity_catalog)
        lines  = [l for l in result.split("\n") if l.strip()] if result else []
        label  = question[:55] + ("..." if len(question) > 55 else "")
        if lines:
            _pass(f"{len(lines):>3} record(s) — {label}")
            passed += 1
        else:
            _fail(f"  0 record(s) — {label}")
            failed += 1

    return passed, failed


# ─────────────────────────────────────────────
# GRAPH STATS
# ─────────────────────────────────────────────

def print_graph_stats(driver):
    _section("GRAPH STATISTICS")

    entity_rows = _run(driver,
        "MATCH (n:Entity) RETURN n.type AS type, count(*) AS cnt "
        "ORDER BY cnt DESC"
    )
    rel_rows = _run(driver,
        "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS cnt "
        "ORDER BY cnt DESC"
    )

    print("\n  Entities by type:")
    total_entities = 0
    for r in entity_rows:
        print(f"    {r['type']:<20} {r['cnt']:>5}")
        total_entities += r["cnt"]
    print(f"    {'TOTAL':<20} {total_entities:>5}")

    print("\n  Relationships by type:")
    total_rels = 0
    for r in rel_rows:
        print(f"    {r['rel_type']:<20} {r['cnt']:>5}")
        total_rels += r["cnt"]
    print(f"    {'TOTAL':<20} {total_rels:>5}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_evaluation():
    driver = connect_neo4j()

    try:
        entity_catalog = fetch_entity_catalog(driver)
        print_graph_stats(driver)

        p1, f1 = eval_structure(driver)
        p2, f2 = eval_financials(driver)
        p3, f3 = eval_geography(driver)
        p4, f4 = eval_query_coverage(driver, entity_catalog)

        total_pass = p1 + p2 + p3 + p4
        total_fail = f1 + f2 + f3 + f4
        total      = total_pass + total_fail
        score      = total_pass / total * 100 if total else 0

        print(f"\n{'═' * 60}")
        print("  FINAL EVALUATION RESULTS")
        print(f"{'─' * 60}")
        print(f"  Structural completeness : {p1}/{p1+f1}")
        print(f"  Financial accuracy      : {p2}/{p2+f2}")
        print(f"  Geographic coverage     : {p3}/{p3+f3}")
        print(f"  Query coverage          : {p4}/{p4+f4}")
        print(f"{'─' * 60}")
        print(f"  Overall score           : {total_pass}/{total}  ({score:.1f}%)")
        print(f"{'═' * 60}\n")

    finally:
        driver.close()


if __name__ == "__main__":
    run_evaluation()
