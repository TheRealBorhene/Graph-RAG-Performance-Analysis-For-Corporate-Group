import os
import re
import time
from openai import OpenAI

from checkpoint import load_checkpoint, save_checkpoint, delete_checkpoint, is_already_processed
from logger import init_log, log_chunk, log_skipped, log_error, log_summary, warn, info, ok, wait
from agents import MODEL, run_pipeline
from guards import apply_entity_guards, _normalize_sub, _segment_canonical
from relationship_validator import validate_relationships

# ─────────────────────────────────────────────
# RATE LIMIT CONFIG
# ─────────────────────────────────────────────

TOKENS_PER_PAIR      = 7600
TPM_LIMIT            = 200_000
SAFE_TPM             = TPM_LIMIT * 0.8
PAIRS_PER_MINUTE     = int(SAFE_TPM / TOKENS_PER_PAIR)
SLEEP_BETWEEN_CHUNKS = 60 / PAIRS_PER_MINUTE


# ─────────────────────────────────────────────
# FILING COMPANY DETECTION
# ─────────────────────────────────────────────

def detect_filing_company(chunks: list[dict], client) -> str:
    sample_text = "\n".join(c["text"] for c in chunks[:3])
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are reading a financial document. Return ONLY the legal name of the company that filed this document. No explanation, no punctuation, just the company name."},
                {"role": "user",   "content": sample_text}
            ],
            temperature=0,
            max_tokens=20
        )
        name = response.choices[0].message.content.strip()
        if not name:
            raise ValueError("Model returned an empty company name")
        if len(name.split()) > 8:
            raise ValueError(f"Model returned a sentence instead of a name: '{name}'")
        # Normalise to title case so "LOEWS CORPORATION" and "Loews Corporation"
        # always produce the same Parent node name regardless of which chunk
        # the model reads the name from.
        return name.title()
    except Exception as e:
        warn(f"detect_filing_company failed ({e})  -  attempting text fallback")
        for chunk in chunks[:3]:
            for line in chunk["text"].splitlines():
                line = line.strip()
                if 3 <= len(line.split()) <= 8 and line[0].isupper() and "." not in line:
                    info(f"Filing company fallback result: '{line}'")
                    return line
        warn("Could not detect filing company  -  using placeholder 'Unknown Company'")
        return "Unknown Company"


# ─────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────

def merge_graph(graph: dict) -> dict:
    """Deduplicate entities and relationships produced across chunks.

    Statement nodes (BalanceSheet, IncomeStatement, CashFlow) seen in multiple
    chunks have their properties dicts merged so that partial views of the same
    statement (e.g. Assets chunk + Liabilities chunk) are combined into one node.
    """
    STATEMENT_TYPES = {"BalanceSheet", "IncomeStatement", "CashFlow"}

    seen_entities: dict[tuple, dict] = {}
    for entity in graph["entities"]:
        key = (entity["name"], entity["type"])
        if key not in seen_entities:
            seen_entities[key] = entity
        elif entity["type"] in STATEMENT_TYPES and entity.get("properties"):
            # Merge partial statement views from different chunks
            existing_props = seen_entities[key].setdefault("properties", {})
            existing_props.update(entity["properties"])

    seen_relationships: dict[tuple, dict] = {}
    for rel in graph["relationships"]:
        key = (rel["source"], rel["target"], rel["type"])
        if key not in seen_relationships:
            seen_relationships[key] = rel
        elif seen_relationships[key].get("property") is None and rel.get("property") is not None:
            seen_relationships[key]["property"] = rel["property"]

    unique_entities = list(seen_entities.values())
    deduped_rels    = list(seen_relationships.values())

    # Secondary pass: same (source, type, property) but different target format
    # e.g. "$1,161 million" and "$1,161" are the same metric — keep the more explicit one.
    prop_groups: dict[tuple, list[dict]] = {}
    for rel in deduped_rels:
        prop = rel.get("property")
        if prop and rel.get("type") == "GENERATED":
            group_key = (rel["source"], rel["type"], prop)
            prop_groups.setdefault(group_key, []).append(rel)

    dropped_format_dups = 0
    dup_ids: set[int] = set()
    for group in prop_groups.values():
        if len(group) == 1:
            continue
        # Prefer target with an explicit unit suffix; otherwise prefer longer string
        best = max(group, key=lambda r: (
            any(u in r["target"].lower() for u in ("million", "billion", "trillion")),
            len(r["target"])
        ))
        for rel in group:
            if rel is not best:
                dup_ids.add(id(rel))
                dropped_format_dups += 1

    unique_relationships = [r for r in deduped_rels if id(r) not in dup_ids]

    print(f"Merge complete:")
    print(f"  Entities      : {len(graph['entities'])} -> {len(unique_entities)}")
    print(f"  Relationships : {len(graph['relationships'])} -> {len(unique_relationships)}"
          + (f"  ({dropped_format_dups} format-duplicate(s) removed)" if dropped_format_dups else ""))

    return {"entities": unique_entities, "relationships": unique_relationships}


# ─────────────────────────────────────────────
# GLOBAL SEGMENT NAME CANONICALISATION
# ─────────────────────────────────────────────

def canonicalize_segment_names(graph: dict) -> dict:
    """
    Global, post-merge canonicalisation of Business Segment node names.

    The per-chunk canonicalisation in guards.apply_entity_guards() runs with only the
    subsidiaries confirmed *so far*, so a segment name extracted before its owning
    subsidiary was seen can survive in a non-canonical form (e.g. "Boardwalk Pipeline
    Partners" in an early chunk vs "Boardwalk Pipeline" in a later one). After merge,
    the full subsidiary list is finally known, so we re-run the SAME canonicalisation
    once, globally — reusing the existing _normalize_sub / _segment_canonical helpers.

    Steps:
      1. Map each Business Segment name to the canonical short form of the subsidiary
         it matches.
      2. Apply renames to Business Segment *nodes*. Re-point relationship endpoints
         ONLY for segment-only names — a name shared with a Subsidiary (e.g.
         "CNA Financial Corporation") keeps its edges on the subsidiary, so the
         subsidiary's metrics are never moved.
      3. Deduplicate, then drop Business Segment nodes left with no relationships —
         these are redundant duplicates of a subsidiary that carry no extracted fact.
    """
    entities      = graph["entities"]
    relationships = graph["relationships"]

    sub_names = {e["name"] for e in entities if e["type"] == "Subsidiary"}
    sub_norms = {s: _normalize_sub(s) for s in sub_names}

    # 1. Build the segment rename map (Business Segment entities only)
    rename: dict[str, str] = {}
    for ent in entities:
        if ent["type"] != "Business Segment":
            continue
        seg_norm = _normalize_sub(ent["name"])
        if not seg_norm:
            continue
        matched_sub = next(
            (s for s, n in sub_norms.items()
             if n and (seg_norm in n or n in seg_norm)),
            None,
        )
        if matched_sub:
            canonical = _segment_canonical(matched_sub)
            if canonical and canonical != ent["name"]:
                rename[ent["name"]] = canonical

    # 2. Apply renames to Business Segment nodes…
    for ent in entities:
        if ent["type"] == "Business Segment" and ent["name"] in rename:
            ent["name"] = rename[ent["name"]]
    # …and to relationship endpoints, but ONLY for segment-only names. A name that is
    # also a Subsidiary (collision) keeps its edges so the subsidiary retains its metrics.
    seg_only = {old for old in rename if old not in sub_names}
    for rel in relationships:
        if rel["source"] in seg_only:
            rel["source"] = rename[rel["source"]]
        if rel["target"] in seg_only:
            rel["target"] = rename[rel["target"]]

    # 3a. Deduplicate entities by (name, type) — merge statement props if any collide
    seen_e: dict[tuple, dict] = {}
    for ent in entities:
        key = (ent["name"], ent["type"])
        if key not in seen_e:
            seen_e[key] = ent
        elif ent.get("properties"):
            seen_e[key].setdefault("properties", {}).update(ent["properties"])
    merged_entities = list(seen_e.values())

    # 3b. Deduplicate relationships by (source, target, type)
    seen_r: dict[tuple, dict] = {}
    for rel in relationships:
        key = (rel["source"], rel["target"], rel["type"])
        if key not in seen_r:
            seen_r[key] = rel
        elif seen_r[key].get("property") is None and rel.get("property") is not None:
            seen_r[key]["property"] = rel["property"]
    merged_rels = list(seen_r.values())

    # 3c. Drop Business Segment nodes left with no relationships (redundant duplicates)
    used = {r["source"] for r in merged_rels} | {r["target"] for r in merged_rels}
    before = len(merged_entities)
    final_entities = [
        e for e in merged_entities
        if e["type"] != "Business Segment" or e["name"] in used
    ]
    dropped_orphans = before - len(final_entities)

    print("Canonicalisation (Business Segments):")
    if rename:
        for old, new in rename.items():
            print(f"  '{old}' -> '{new}'")
    else:
        print("  no segment names needed collapsing")
    print(f"  Entities      : {len(entities)} -> {len(final_entities)}"
          + (f"  ({dropped_orphans} orphan segment node(s) dropped)" if dropped_orphans else ""))
    print(f"  Relationships : {len(relationships)} -> {len(merged_rels)}")

    return {"entities": final_entities, "relationships": merged_rels}


# ─────────────────────────────────────────────
# FINANCIAL ITEM ENRICHMENT
# ─────────────────────────────────────────────

_FY_PROP_PATTERN = re.compile(r'^(.*?)\s+fy(\d{4})$', re.IGNORECASE)


def enrich_financial_items(entities: list[dict], relationships: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Rename Financial Item nodes from raw values to 'Label: value' format
    and simplify relationship properties to just the fiscal year.

    Before: entity name = "$72,880"           property = "net income fy2025"
    After:  entity name = "Net Income: $72,880"   property = "2025"

    Rules:
    - Only REPORTED / GENERATED / HAS_METRIC relationships are affected.
    - Relationships with a null or unparseable property are left unchanged.
    - If a raw value is still referenced by an unchanged relationship, the
      original entity is kept alongside the new labelled one.
    - Called after merge_graph() so it operates on clean, deduplicated data.
    """
    financial_types = {"GENERATED"}

    rename_ops: list[tuple[int, str, str, str]] = []
    for idx, rel in enumerate(relationships):
        if rel.get("type") not in financial_types:
            continue
        prop = (rel.get("property") or "").strip()
        tgt  = (rel.get("target")   or "").strip()
        if not prop or not tgt:
            continue
        m = _FY_PROP_PATTERN.match(prop)
        if not m:
            continue
        label    = m.group(1).strip().title()
        year     = m.group(2)
        new_name = f"{label}: {tgt}"
        rename_ops.append((idx, tgt, new_name, year))

    if not rename_ops:
        return entities, relationships

    renamed_indices   = {idx for idx, *_ in rename_ops}
    renamed_old_names = {old for _, old, _, _ in rename_ops}

    still_used: set[str] = set()
    for idx, rel in enumerate(relationships):
        tgt = (rel.get("target") or "").strip()
        if tgt in renamed_old_names and idx not in renamed_indices:
            still_used.add(tgt)

    old_to_new: dict[str, set[str]] = {}
    for _, old, new_name, _ in rename_ops:
        old_to_new.setdefault(old, set()).add(new_name)

    added_names: set[str] = {e["name"] for e in entities}
    new_entities: list[dict] = []

    for e in entities:
        if e["type"] != "Financial Item" or e["name"] not in old_to_new:
            new_entities.append(e)
            continue
        if e["name"] in still_used:
            new_entities.append(e)
        for new_name in old_to_new[e["name"]]:
            if new_name not in added_names:
                new_e = dict(e)
                new_e["name"] = new_name
                new_entities.append(new_e)
                added_names.add(new_name)

    for idx, _, new_name, year in rename_ops:
        relationships[idx]["target"]   = new_name
        relationships[idx]["property"] = year

    print(f"  Enriched {len(rename_ops)} Financial Item relationship(s) → 'Label: value' node names")
    return new_entities, relationships


# ─────────────────────────────────────────────
# ORPHANED FINANCIAL ITEM FILTER
# ─────────────────────────────────────────────

def drop_orphaned_financial_items(entities: list[dict], relationships: list[dict]) -> list[dict]:
    """
    Remove Financial Item entities that are not the target of any relationship.
    These are numeric values extracted from tables (reserve development rows,
    footnote tables, etc.) that Agent 3 never linked to a segment or subsidiary.
    They add noise to the graph without contributing any queryable information.
    Called after merge_graph() and enrich_financial_items().
    """
    rel_targets = {rel["target"] for rel in relationships if rel.get("target")}
    before = len(entities)
    kept = [
        e for e in entities
        if e["type"] != "Financial Item" or e["name"] in rel_targets
    ]
    dropped = before - len(kept)
    if dropped:
        print(f"  Dropped {dropped} orphaned Financial Item(s) with no incoming relationship")
    return kept


# ─────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────

def extract_graph(chunks: list[dict]) -> dict:

    client         = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    filing_company = detect_filing_company(chunks, client)
    print(f"Filing company detected: {filing_company}\n")

    checkpoint    = load_checkpoint()
    last_chunk_id = checkpoint["last_chunk_id"]
    all_entities  = checkpoint["entities"]
    all_relations = checkpoint["relationships"]

    total_pairs   = (len(chunks) + 1) // 2
    run_timestamp = init_log(
        (len([c for c in chunks if c["metadata"]["chunk_id"] > last_chunk_id]) + 1) // 2
    )

    for i in range(0, len(chunks), 2):
        chunk_a  = chunks[i]
        chunk_b  = chunks[i + 1] if i + 1 < len(chunks) else None
        pair_idx = i // 2 + 1

        chunk_id     = chunk_b["metadata"]["chunk_id"] if chunk_b else chunk_a["metadata"]["chunk_id"]
        current_text = chunk_a["text"] + ("\n\n---\n\n" + chunk_b["text"] if chunk_b else "")
        page_number  = chunk_a["metadata"]["page_number"]

        if is_already_processed(chunk_id, last_chunk_id):
            print(f"[{pair_idx}/{total_pairs}] Skipping page {page_number}  -  already processed [OK]")
            log_skipped(run_timestamp, chunk_id)
            continue

        print(f"[{pair_idx}/{total_pairs}] Extracting from page {page_number}...")

        chunk_entities      = []
        chunk_relationships = []

        chunk_trace: dict = {}
        try:
            confirmed_subs = {e["name"] for e in all_entities if e["type"] == "Subsidiary"}
            chunk_entities, chunk_relationships = run_pipeline(
                current_text, filing_company, client,
                lambda ents: apply_entity_guards(ents, current_text, filing_company, confirmed_subs),
                run_timestamp, chunk_id,
                known_entities=all_entities,
                trace=chunk_trace
            )

            if not chunk_entities:
                log_chunk(run_timestamp, chunk_id, [], [], len(all_entities), len(all_relations), trace=chunk_trace)
                last_chunk_id = chunk_id
                save_checkpoint(last_chunk_id, all_entities, all_relations)
                if pair_idx < total_pairs:
                    time.sleep(SLEEP_BETWEEN_CHUNKS)
                continue

        except Exception as e:
            warn(f"API error on chunk {chunk_id}: {e}")
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait(f"Rate limit hit  -  waiting 60 seconds...")
                time.sleep(60)
            log_error(run_timestamp, chunk_id, str(e))
            save_checkpoint(last_chunk_id, all_entities, all_relations)
            continue

        for entity in chunk_entities:
            entity["chunk_id"]    = chunk_id
            entity["file"]        = chunk_a["metadata"]["file"]
            entity["page_number"] = page_number
            all_entities.append(entity)

        entity_names    = {e["name"] for e in all_entities} | {filing_company}
        entity_type_map = {e["name"]: e["type"] for e in all_entities} | {filing_company: "Parent"}
        chunk_relationships = validate_relationships(
            chunk_relationships, entity_names, entity_type_map,
            chunk_id, chunk_a["metadata"]["file"], page_number, run_timestamp,
            all_entities, current_text)
        all_relations.extend(chunk_relationships)

        print(f"  -----------------------------------------")
        print(f"  Total so far: {len(all_entities)} entities, {len(all_relations)} relationships")
        print(f"  -----------------------------------------")

        # Record the post-validation relationships (what actually reaches the graph) so the
        # log shows both the raw Agent 3 output (trace["relationships"]) and the final set.
        chunk_trace["final_relationships"] = chunk_relationships
        log_chunk(run_timestamp, chunk_id, chunk_entities, chunk_relationships, len(all_entities), len(all_relations), trace=chunk_trace)
        last_chunk_id = chunk_id
        save_checkpoint(last_chunk_id, all_entities, all_relations)
        ok(f"Checkpoint saved at page {page_number}")

        if pair_idx < total_pairs:
            time.sleep(SLEEP_BETWEEN_CHUNKS)

    log_summary(run_timestamp, len(all_entities), len(all_relations), len(chunks))
    delete_checkpoint()

    print(f"\n{'-' * 50}")
    print(f"Extraction complete")
    print(f"Total entities extracted     : {len(all_entities)}")
    print(f"Total relationships extracted: {len(all_relations)}")
    print(f"{'-' * 50}")

    return {"entities": all_entities, "relationships": all_relations}
