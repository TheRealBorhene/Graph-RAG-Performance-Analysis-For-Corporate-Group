"""
Post-Agent-3 relationship validation.

Validates and normalises every relationship the relationship_finder agent (Agent 3) emits
before it reaches the graph. Each per-step concern is a small helper; the top-level
`validate_relationships` is the orchestrator that runs them in the correct order per edge.

Extracted from guards.py to keep the entity-side guards and the relationship-side
validation cleanly separated. Reuses shared name-normalisation and pattern helpers
from guards.py — no logic is duplicated.
"""

from guards import (
    FINANCIAL_VALUE_PATTERN,
    _normalize_sub,
    _normalize_paren_negative,
    _resolve_source_by_content,
)
from logger import info, warn, new, log_error


# ─────────────────────────────────────────────
# PER-STEP HELPERS
# ─────────────────────────────────────────────

def _normalise_target(rel: dict, tgt, entity_names: set, dollar_normalised: dict):
    """Run all three target normalisations in order: $ prefix correction, $- flip,
    parenthetical-negative. Mutates rel in place and returns the (possibly updated) tgt."""
    # $ prefix correction
    if tgt and tgt not in entity_names:
        canonical = dollar_normalised.get(str(tgt).lstrip("$").strip())
        if canonical:
            rel["target"] = canonical
            tgt = canonical
    # $-value → -$value
    if tgt and str(tgt).startswith("$-"):
        tgt = "-$" + str(tgt)[2:]
        rel["target"] = tgt
    # parenthetical negative
    if tgt and tgt not in entity_names:
        normalised = _normalize_paren_negative(tgt)
        if normalised:
            info(f"Target parenthetical normalised: '{tgt}' → '{normalised}'")
            rel["target"] = normalised
            tgt = normalised
    return tgt


def _resolve_source(rel: dict, src, rel_type, entity_names: set,
                    entity_type_map: dict, chunk_text: str):
    """Run the three source-resolution stages in order: exact-name alias, content resolver
    (for GENERATED only), then chunk-text scan fallback (GENERATED only, single-match only).
    Mutates rel in place and returns the (possibly updated) src."""
    # Exact-name alias resolution
    if src and src not in entity_names:
        src_norm = _normalize_sub(src)
        canonical_src = next(
            (name for name in entity_names
             if src_norm and _normalize_sub(name) == src_norm),
            None
        )
        if canonical_src:
            info(f"Source alias resolved: '{src}' → '{canonical_src}'")
            rel["source"] = canonical_src
            src = canonical_src

    # Content resolution — resolve from the source label's own words (e.g. "CNA segment" →
    # the confirmed subsidiary whose leading token matches "cna"). Runs BEFORE the chunk-text
    # scan below, so an abbreviated owner is canonicalised from the source string itself and
    # is never misattributed to an unrelated subsidiary that merely appears in the chunk text.
    if src and src not in entity_names and rel_type == "GENERATED":
        sub_names = [n for n in entity_names if entity_type_map.get(n) == "Subsidiary"]
        content_src = _resolve_source_by_content(src, sub_names)
        if content_src:
            info(f"Source resolved by content: '{src}' → '{content_src}'")
            rel["source"] = content_src
            src = content_src

    # Fix 1 — Sub-segment GENERATED fallback. Agent 3 sometimes uses a sub-segment row
    # header ("Specialty", "Commercial") as the source instead of the owning subsidiary.
    # Scan the chunk TEXT for confirmed subsidiary names; if exactly one appears it owns
    # the metric. Ambiguous (0 or >1) → fall through and let the existence check drop it.
    if src and src not in entity_names and rel_type == "GENERATED" and chunk_text:
        text_lower = chunk_text.lower()
        text_matched_subs = list(dict.fromkeys(
            name for name in entity_names
            if entity_type_map.get(name) == "Subsidiary"
            and (name.lower() in text_lower
                 or (_normalize_sub(name) and _normalize_sub(name) in text_lower))
        ))
        if len(text_matched_subs) == 1:
            info(f"Sub-segment source re-routed: '{src}' → '{text_matched_subs[0]}'")
            rel["source"] = text_matched_subs[0]
            src = text_matched_subs[0]
    return src


def _auto_register_financial_item(tgt, rel_type, src_in, tgt_in,
                                  entity_names: set, all_entities: list[dict],
                                  chunk_id: int, file: str, page_number: int) -> bool:
    """If a GENERATED edge has a valid source and a target that looks like a financial
    value but isn't yet an entity, register it as a Financial Item on the fly. Returns
    the updated tgt_in flag."""
    if (rel_type == "GENERATED"
            and src_in and not tgt_in
            and FINANCIAL_VALUE_PATTERN.match(str(tgt or ""))):
        all_entities.append({"name": tgt, "type": "Financial Item",
                              "chunk_id": chunk_id, "file": file, "page_number": page_number})
        entity_names.add(tgt)
        new(f"Auto-registered Financial Item: {tgt}")
        return True
    return tgt_in


def _correct_inverted_parent_of(rel: dict, src, tgt, rel_type, src_type, entity_type_map):
    """If a PARENT_OF edge points Subsidiary → Parent, flip it. Mutates rel in place and
    returns the updated (src, tgt, src_type)."""
    if rel_type == "PARENT_OF" and src_type == "Subsidiary" and entity_type_map.get(tgt, "") == "Parent":
        warn(f"Inverted PARENT_OF corrected: '{src}' ↔ '{tgt}'")
        rel["source"], rel["target"] = tgt, src
        src, tgt = rel["source"], rel["target"]
        src_type = entity_type_map.get(src, "")
    return src, tgt, src_type


def _reroute_segment_operates_in(rel: dict, src, src_type, rel_type,
                                 entity_names: set, entity_type_map: dict,
                                 all_entities: list[dict]):
    """Fix 2 — OPERATES_IN allows only Parent/Subsidiary sources but Agent 3 sometimes uses a
    segment name. Two cases:
       2a: dual-type conflict — same name also exists as a Subsidiary in all_entities → treat as such.
       2b: short segment name → find the real subsidiary by normalised-name match.
    Returns (src, src_type, dropped) where dropped is True if the edge must be discarded."""
    if rel_type != "OPERATES_IN" or src_type != "Business Segment":
        return src, src_type, False
    # 2a: dual-type
    if any(e["name"] == src and e["type"] == "Subsidiary" for e in all_entities):
        info(f"Dual-type resolved: '{src}' treated as Subsidiary for OPERATES_IN")
        return src, "Subsidiary", False
    # 2b: short segment name
    src_norm = _normalize_sub(src)
    matched_sub = next(
        (name for name in entity_names
         if entity_type_map.get(name) == "Subsidiary"
         and src_norm
         and _normalize_sub(name)
         and (src_norm in _normalize_sub(name) or _normalize_sub(name) in src_norm)),
        None
    )
    if matched_sub:
        info(f"Segment OPERATES_IN source re-routed: '{src}' → '{matched_sub}'")
        rel["source"] = matched_sub
        return matched_sub, "Subsidiary", False
    return src, src_type, True   # caller drops


def _enforce_relationship_type(rel_type, src, src_type, tgt, tgt_type) -> str | None:
    """Run the three terminal type checks. Returns None if the edge is valid, or a string
    explaining the drop reason (for logging by the caller)."""
    if rel_type == "OPERATES_IN" and src_type not in ("Parent", "Subsidiary"):
        return f"Dropped OPERATES_IN: source '{src}' is [{src_type}]  -  must be Parent or Subsidiary"
    if rel_type == "OPERATES_IN" and tgt_type != "Geography":
        return f"Dropped OPERATES_IN: target '{tgt}' is [{tgt_type}]  -  must be Geography"
    if rel_type == "GENERATED" and src_type not in ("Business Segment", "Subsidiary"):
        return f"Dropped GENERATED: source '{src}' is [{src_type}]  -  must be Business Segment or Subsidiary"
    if rel_type == "BOARD_MEMBER_OF" and src_type != "Person":
        return f"Dropped BOARD_MEMBER_OF: source '{src}' is [{src_type}]  -  must be Person"
    if rel_type == "BOARD_MEMBER_OF" and tgt_type not in ("Parent", "Subsidiary"):
        return f"Dropped BOARD_MEMBER_OF: target '{tgt}' is [{tgt_type}]  -  must be Parent or Subsidiary"
    return None


# ─────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────

def validate_relationships(relationships: list[dict], entity_names: set,
                           entity_type_map: dict, chunk_id: int,
                           file: str, page_number: int,
                           run_timestamp: str, all_entities: list[dict],
                           chunk_text: str = "") -> list[dict]:
    """
    Validate Agent 3 relationships: normalise target value forms, resolve source aliases,
    auto-register Financial Item targets, correct inverted PARENT_OF, re-route Business
    Segment OPERATES_IN sources, and enforce relationship-type constraints. Each step is
    a small helper above. The order matters: source/target normalisation runs before
    existence checks, which run before type checks.
    """
    dollar_normalised = {
        name.lstrip("$").strip(): name
        for name in entity_names if name.startswith("$")
    }

    valid: list[dict] = []
    for rel in relationships:
        src      = rel.get("source")
        tgt      = rel.get("target")
        rel_type = rel.get("type")

        tgt = _normalise_target(rel, tgt, entity_names, dollar_normalised)
        src = _resolve_source(rel, src, rel_type, entity_names, entity_type_map, chunk_text)

        src_in = src in entity_names
        tgt_in = tgt in entity_names
        tgt_in = _auto_register_financial_item(
            tgt, rel_type, src_in, tgt_in, entity_names, all_entities,
            chunk_id, file, page_number,
        )

        if not src_in or not tgt_in:
            warn(f"Dropped invalid relationship: [{src}] --{rel_type} --> [{tgt}]")
            log_error(run_timestamp, chunk_id,
                      f"Invalid relationship dropped: source='{src}' target='{tgt}' type='{rel_type}'")
            continue

        src_type = entity_type_map.get(src, "")
        src, tgt, src_type = _correct_inverted_parent_of(
            rel, src, tgt, rel_type, src_type, entity_type_map,
        )

        src, src_type, dropped = _reroute_segment_operates_in(
            rel, src, src_type, rel_type, entity_names, entity_type_map, all_entities,
        )
        if dropped:
            warn(f"Dropped OPERATES_IN: source '{src}' is [Business Segment] and no matching subsidiary found")
            log_error(run_timestamp, chunk_id, f"Invalid OPERATES_IN: source='{src}' type=[{src_type}]")
            continue

        drop_reason = _enforce_relationship_type(
            rel_type, src, src_type, tgt, entity_type_map.get(tgt, ""),
        )
        if drop_reason:
            warn(drop_reason)
            log_error(run_timestamp, chunk_id, drop_reason.replace("Dropped ", "Invalid "))
            continue

        rel["chunk_id"]    = chunk_id
        rel["file"]        = file
        rel["page_number"] = page_number
        valid.append(rel)

    return valid
