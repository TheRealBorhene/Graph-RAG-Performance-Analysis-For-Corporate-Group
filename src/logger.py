import os
from datetime import datetime

LOG_DIR = "../data/logs"


def _get_log_path(run_timestamp: str) -> str:
    return f"{LOG_DIR}/extractor_log_{run_timestamp}.md"


# ─────────────────────────────────────────────
# CONSOLE LOG HELPERS
# ─────────────────────────────────────────────

def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")

def info(msg: str) -> None:
    print(f"  [INFO] {msg}")

def drop(msg: str) -> None:
    print(f"  [DROP] {msg}")

def ok(msg: str) -> None:
    print(f"  [OK] {msg}")

def new(msg: str) -> None:
    print(f"  [NEW] {msg}")

def wait(msg: str) -> None:
    print(f"  [WAIT] {msg}")


# ─────────────────────────────────────────────
# FILE LOG FUNCTIONS
# ─────────────────────────────────────────────

def init_log(total_pairs: int) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path      = _get_log_path(run_timestamp)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Extractor Log\n\n")
        f.write(f"**Started:** {run_timestamp.replace('_', ' ')}\n")
        f.write(f"**Pairs to process:** {total_pairs}\n\n")
        f.write("---\n\n")

    print(f"Log file: {log_path}")
    return run_timestamp


def log_chunk(run_timestamp: str, chunk_id: int, entities: list, relationships: list,
              total_entities: int, total_relationships: int, trace: dict | None = None) -> None:
    """
    Append a full per-chunk record to the log.

    When `trace` is provided (populated by agents.run_pipeline), every API call's raw
    output is written: Agent 1 candidates, Agent 2 entities, Agent 3 relationships, and
    the statement extractor's nodes (with all line items) and edges. Without a trace,
    falls back to the legacy entities/relationships-only summary.
    """
    with open(_get_log_path(run_timestamp), "a", encoding="utf-8") as f:
        f.write(f"## Chunk {chunk_id}\n\n")

        if trace is not None:
            _write_full_trace(f, trace)
        else:
            _write_legacy(f, entities, relationships)

        f.write(f"**Running total:** {total_entities} entities, {total_relationships} relationships\n\n")
        f.write("---\n\n")


def _write_full_trace(f, trace: dict) -> None:
    """Write the complete per-stage pipeline output for one chunk."""
    candidates    = trace.get("candidates")      or []
    raw_entities  = trace.get("raw_entities")    or []
    entities      = trace.get("entities")        or []
    relationships = trace.get("relationships")   or []
    stmt_nodes    = trace.get("statement_nodes") or []
    stmt_rels     = trace.get("statement_rels")  or []

    # ── Agent 1 — raw candidates ─────────────────────────────────────────────
    f.write(f"### Agent 1 — Raw candidates ({len(candidates)})\n\n")
    if candidates:
        for c in candidates:
            f.write(f"- {c}\n")
    else:
        f.write("- none\n")
    f.write("\n")

    # ── Agent 2 — classified entities ────────────────────────────────────────
    f.write(f"### Agent 2 — Classified entities ({len(entities)})\n\n")
    if raw_entities and len(raw_entities) != len(entities):
        f.write(f"*Agent 2 returned {len(raw_entities)} raw entity(ies); "
                f"{len(raw_entities) - len(entities)} removed by type filter + guards.*\n\n")
    if entities:
        for entity in entities:
            f.write(f"- `[{entity['type']}]` {entity['name']}\n")
    else:
        f.write("- none\n")
    f.write("\n")

    # ── Agent 3 — relationships ──────────────────────────────────────────────
    f.write(f"### Agent 3 — Relationships ({len(relationships)})\n\n")
    if relationships:
        for rel in relationships:
            f.write(f"- {rel['source']} -- {rel['type']} --> {rel['target']}  "
                    f"(property: {rel.get('property')})\n")
    else:
        f.write("- none\n")
    f.write("\n")

    # ── Final relationships after guard validation (what reaches the graph) ──
    # Only written when it differs from the raw Agent 3 output, so the log makes
    # clear which relationships the guards re-routed or dropped.
    final_rels = trace.get("final_relationships")
    if final_rels is not None and final_rels != relationships:
        f.write(f"### Relationships after validation ({len(final_rels)})\n\n")
        if final_rels:
            for rel in final_rels:
                f.write(f"- {rel['source']} -- {rel['type']} --> {rel['target']}  "
                        f"(property: {rel.get('property')})\n")
        else:
            f.write("- none\n")
        f.write("\n")

    # ── Statement extractor — nodes (with line items) ────────────────────────
    f.write(f"### Statement extractor — Nodes ({len(stmt_nodes)})\n\n")
    if stmt_nodes:
        for node in stmt_nodes:
            props = node.get("properties") or {}
            f.write(f"- `[{node['type']}]` {node['name']}  ({len(props)} item(s))\n")
            for k, v in props.items():
                f.write(f"    - {k}: {v}\n")
    else:
        f.write("- none\n")
    f.write("\n")

    # ── Statement extractor — edges ──────────────────────────────────────────
    f.write(f"### Statement extractor — Edges ({len(stmt_rels)})\n\n")
    if stmt_rels:
        for sr in stmt_rels:
            f.write(f"- {sr['source']} -- {sr['type']} --> {sr['target']}  "
                    f"(property: {sr.get('property')})\n")
    else:
        f.write("- none\n")
    f.write("\n")


def _write_legacy(f, entities: list, relationships: list) -> None:
    """Legacy summary format — used when no trace is supplied."""
    f.write(f"**Entities found:** {len(entities)}\n\n")
    if entities:
        for entity in entities:
            f.write(f"- `[{entity['type']}]` {entity['name']}\n")
    else:
        f.write("- none\n")
    f.write("\n")

    f.write(f"**Relationships found:** {len(relationships)}\n\n")
    if relationships:
        for rel in relationships:
            f.write(f"- {rel['source']} -- {rel['type']} --> {rel['target']}  (property: {rel.get('property')})\n")
    else:
        f.write("- none\n")
    f.write("\n")


def log_skipped(run_timestamp: str, chunk_id: int) -> None:
    with open(_get_log_path(run_timestamp), "a", encoding="utf-8") as f:
        f.write(f"## Chunk {chunk_id}\n\n")
        f.write(f"**Status:** Skipped - already processed in previous run\n\n")
        f.write("---\n\n")


def log_error(run_timestamp: str, chunk_id: int, reason: str) -> None:
    with open(_get_log_path(run_timestamp), "a", encoding="utf-8") as f:
        f.write(f"## Chunk {chunk_id}\n\n")
        f.write(f"**Status:** Error - {reason}\n\n")
        f.write("---\n\n")


def log_summary(run_timestamp: str, total_entities: int, total_relationships: int, total_chunks: int) -> None:
    log_path = _get_log_path(run_timestamp)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"# Summary\n\n")
        f.write(f"**Finished:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Chunks processed:** {total_chunks}\n\n")
        f.write(f"**Total entities extracted:** {total_entities}\n\n")
        f.write(f"**Total relationships extracted:** {total_relationships}\n\n")

    print(f"Log saved to: {log_path}")
