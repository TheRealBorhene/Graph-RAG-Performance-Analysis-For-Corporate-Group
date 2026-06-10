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


def log_chunk(run_timestamp: str, chunk_id: int, entities: list, relationships: list, total_entities: int, total_relationships: int) -> None:
    with open(_get_log_path(run_timestamp), "a", encoding="utf-8") as f:
        f.write(f"## Chunk {chunk_id}\n\n")

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
        f.write(f"**Running total:** {total_entities} entities, {total_relationships} relationships\n\n")
        f.write("---\n\n")


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
