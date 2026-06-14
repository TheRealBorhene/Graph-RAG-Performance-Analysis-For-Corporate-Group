import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from extractor import detect_filing_company, merge_graph
from guards import (
    apply_entity_guards,
    FINANCIAL_VALUE_PATTERN, _normalize_sub
)

# Geography checks — defined locally because the guards were removed from guards.py.
# These now test that the prompt alone correctly rejects bad geographies.
# Any hit here means the LLM ignored the prompt rule.
_GEO_REGULATORY = re.compile(
    r'\b(department|authority|commission|association|commissioners|supervisors|'
    r'superintendent|monetary|institute|bureau|committee|council|'
    r'office\s+of|board\s+of)\b',
    re.IGNORECASE
)
_GEO_TOO_BROAD = re.compile(
    r'^(north america|south america|latin america|central america|'
    r'europe|asia|africa|middle east|oceania|pacific|'
    r'apac|emea|americas|worldwide|global|international|'
    r'rest of world|other|western europe|eastern europe|'
    r'southeast asia|east asia|south asia|sub-saharan africa|'
    r'gulf coast|midwest|northeast|northwest|southeast|southwest)$',
    re.IGNORECASE
)
from agents import run_pipeline

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
COLLECTION_NAME = "LoewsCompany"

VALID_REL_TYPES     = {"PARENT_OF", "OPERATES_IN", "GENERATED",
                       "BALANCE_SHEET", "INCOME_STATEMENT", "CASH_FLOW"}
FINANCIAL_REL_TYPES = {"GENERATED"}

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
client_oai    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_qdrant = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", 6333))
)

# Detect filing company from first 3 chunks
first_points = client_qdrant.retrieve(
    collection_name=COLLECTION_NAME,
    ids=[0, 1, 2],
    with_payload=True,
    with_vectors=False
)
first_chunks   = [{"text": p.payload["text"]} for p in sorted(first_points, key=lambda p: p.id)]
filing_company = detect_filing_company(first_chunks, client_oai)

# Fetch ALL chunk IDs from the collection dynamically
all_points = client_qdrant.scroll(
    collection_name=COLLECTION_NAME,
    limit=10000,
    with_payload=False,
    with_vectors=False
)[0]
ALL_CHUNK_IDS = sorted(p.id for p in all_points)

# ── VERIFICATION SUBSET ───────────────────────────────────────────
# Set VERIFY_FIXES_ONLY = True to run only the targeted chunks that
# cover the recent fixes. Set to False for the full document scan.
VERIFY_FIXES_ONLY = True

FIXES_CHUNK_IDS = [
    3,    # Fix 3 — CNA dual-type conflict (CNA Financial Corporation as both Subsidiary and Business Segment)
    53,   # Fix 1 — CNA sub-segments (Specialty/Commercial/International) as GENERATED sources
    55,   # Fix 1 — Loews Corporation as GENERATED source (Parent fallback)
    157,  # Fix 2 — OPERATES_IN from Business Segment source (Loews Hotels / Boardwalk Pipeline)
    171,  # Prompt fix — attribution grounding (Boardwalk Pipeline mistakenly got CNA insurance metrics)
]

if VERIFY_FIXES_ONLY:
    ALL_CHUNK_IDS = FIXES_CHUNK_IDS
    print(f"Mode: VERIFY FIXES ONLY — {len(ALL_CHUNK_IDS)} targeted chunks\n")

# Test fixture — simulates confirmed_subsidiaries as the pipeline builds it
# dynamically after processing the document's intro chunks.
# Used only by apply_entity_guards() in Section 1 so alias resolution works
# during testing without running the full pipeline first.
KNOWN_ENTITIES = [
    {"name": filing_company,                     "type": "Parent"},
    {"name": "CNA Financial Corporation",        "type": "Subsidiary"},
    {"name": "Diamond Offshore Drilling, Inc.",  "type": "Subsidiary"},
    {"name": "Boardwalk Pipeline Partners, LP",  "type": "Subsidiary"},
    {"name": "Loews Hotels Holding Corporation", "type": "Subsidiary"},
    {"name": "Altium Packaging LLC",             "type": "Subsidiary"},
]
confirmed_subs = {e["name"] for e in KNOWN_ENTITIES if e["type"] == "Subsidiary"}

print(f"Filing company  : {filing_company}")
print(f"Total chunks    : {len(ALL_CHUNK_IDS)}")
print(f"Known entities  : {len(KNOWN_ENTITIES)} ({len(confirmed_subs)} subsidiaries)\n")

# ══════════════════════════════════════════════════════
# SECTION 1 — FULL DOCUMENT SCAN (all chunks)
# ══════════════════════════════════════════════════════
print("█" * 65)
print("  SECTION 1 — FULL DOCUMENT QUALITY SCAN")
print("█" * 65)


total_pass = 0
total_fail = 0
total_warn = 0
chunk_results: dict[int, tuple] = {}  # chunk_id -> (entities, relationships)

# Issue trackers for summary
issues: dict[str, list] = {
    "geo_regulatory_body":   [],   # Geography classified as regulatory body
    "geo_too_broad":         [],   # Geography too vague (region/bloc)
    "invalid_rel_type":      [],   # Unknown relationship type
    "self_referential":      [],   # source == target
    "bad_parent_of_source":  [],   # PARENT_OF source is not Parent
    "fi_target_invalid":     [],   # GENERATED target not FI and not auto-registerable
    "fi_rel_no_fi_entity":   [],   # GENERATED produced with no FI entities
    "sub_false_positive":    [],   # Subsidiary entity not matching any confirmed canonical
}

# Precompute normalised keys for confirmed subsidiaries once
confirmed_norms: dict[str, str] = {_normalize_sub(s): s for s in confirmed_subs}

# Tracks which confirmed subsidiaries were seen at least once across all chunks
confirmed_subs_seen: dict[str, int] = {s: 0 for s in confirmed_subs}

for chunk_id in ALL_CHUNK_IDS:
    result = client_qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[chunk_id],
        with_payload=True,
        with_vectors=False
    )
    if not result:
        continue

    point   = result[0]
    text    = point.payload["text"]
    section = point.payload.get("section", "")

    entities, relationships = run_pipeline(
        text           = text,
        filing_company = filing_company,
        client         = client_oai,
        guard_fn       = lambda ents, t=text: apply_entity_guards(
                             ents, t, filing_company, confirmed_subs
                         ),
        run_timestamp  = "test",
        chunk_id       = chunk_id,
        known_entities = KNOWN_ENTITIES
    )

    chunk_results[chunk_id] = (entities, relationships)
    chunk_issues = []

    # Build entity type map (include injected Parent)
    entity_type_map = {e["name"]: e["type"] for e in entities}
    entity_type_map[filing_company] = "Parent"
    fi_names = {e["name"] for e in entities if e["type"] == "Financial Item"}

    # ── Check 1: Geography — regulatory body slipthrough ─────────
    # No guard in guards.py anymore — prompt-only. A hit here = prompt failure.
    for e in entities:
        if e["type"] == "Geography" and _GEO_REGULATORY.search(e["name"]):
            msg = f"Chunk {chunk_id}: [Geography] '{e['name']}' looks like a regulatory body (prompt missed it)"
            chunk_issues.append(msg)
            issues["geo_regulatory_body"].append(msg)

    # ── Check 2: Geography — too broad ───────────────────────────
    # No guard in guards.py anymore — prompt-only. A hit here = prompt failure.
    for e in entities:
        if e["type"] == "Geography" and _GEO_TOO_BROAD.match(e["name"].strip()):
            msg = f"Chunk {chunk_id}: [Geography] '{e['name']}' is too broad (prompt missed it)"
            chunk_issues.append(msg)
            issues["geo_too_broad"].append(msg)

    # ── Check 3: Invalid relationship types ──────────────────────
    for r in relationships:
        if r.get("type") not in VALID_REL_TYPES:
            msg = f"Chunk {chunk_id}: unknown rel type '{r.get('type')}'"
            chunk_issues.append(msg)
            issues["invalid_rel_type"].append(msg)

    # ── Check 4: Self-referential relationships ───────────────────
    for r in relationships:
        if r.get("source") == r.get("target"):
            msg = f"Chunk {chunk_id}: self-ref '{r.get('source')}' --{r.get('type')}--> '{r.get('target')}'"
            chunk_issues.append(msg)
            issues["self_referential"].append(msg)

    # ── Check 6: PARENT_OF source must be Parent ─────────────────
    for r in relationships:
        if r.get("type") == "PARENT_OF" and entity_type_map.get(r.get("source")) != "Parent":
            msg = f"Chunk {chunk_id}: PARENT_OF source '{r.get('source')}' is not a Parent"
            chunk_issues.append(msg)
            issues["bad_parent_of_source"].append(msg)

    # ── Check 7: Financial relationship targets are valid ─────────
    for r in relationships:
        if r.get("type") in FINANCIAL_REL_TYPES:
            tgt = r.get("target") or ""
            if tgt not in fi_names and not FINANCIAL_VALUE_PATTERN.match(str(tgt)):
                msg = f"Chunk {chunk_id}: {r.get('type')} target '{tgt}' is not a Financial Item"
                chunk_issues.append(msg)
                issues["fi_target_invalid"].append(msg)

    # ── Check 8: No financial rels when no FI entities ────────────
    has_fi  = any(e["type"] == "Financial Item" for e in entities)
    has_fri = any(r["type"] in FINANCIAL_REL_TYPES for r in relationships)
    if not has_fi and has_fri:
        msg = f"Chunk {chunk_id}: financial relationships found but no Financial Item entities"
        chunk_issues.append(msg)
        issues["fi_rel_no_fi_entity"].append(msg)

    # ── Check 9: Confirmed subsidiary alias resolution ────────────
    # Verify no relationship uses an unresolved alias as source when a
    # canonical confirmed subsidiary exists for that alias.
    sub_names = {e["name"] for e in entities if e["type"] == "Subsidiary"}
    for r in relationships:
        src = r.get("source") or ""
        if src and src != filing_company and src not in sub_names:
            # Check if this looks like an alias of a confirmed subsidiary
            src_norm = _normalize_sub(src)
            for conf in confirmed_subs:
                conf_norm = _normalize_sub(conf)
                if src_norm and conf_norm and (src_norm in conf_norm or conf_norm in src_norm):
                    msg = (f"Chunk {chunk_id}: relationship source '{src}' looks like alias "
                           f"of confirmed '{conf}' but wasn't resolved")
                    chunk_issues.append(msg)
                    break

    # ── Check 10: Subsidiary extraction quality ──────────────────
    # For every Subsidiary entity found in this chunk:
    #   a) Track coverage — mark the canonical name as seen
    #   b) Flag false positives — names that don't resolve to any canonical
    for e in entities:
        if e["type"] != "Subsidiary":
            continue
        norm = _normalize_sub(e["name"])
        matched = confirmed_norms.get(norm)

        # Try substring fallback if exact normalised key didn't match
        if not matched:
            matched = next(
                (canon for canon, canon_norm in confirmed_norms.items()
                 if norm and canon_norm and (norm in canon_norm or canon_norm in norm)),
                None
            )
            if matched:
                matched = confirmed_norms.get(matched, matched)

        if matched:
            confirmed_subs_seen[matched] = confirmed_subs_seen.get(matched, 0) + 1
        else:
            msg = (f"Chunk {chunk_id}: [Subsidiary] '{e['name']}' "
                   f"(norm: '{norm}') does not match any confirmed subsidiary")
            chunk_issues.append(msg)
            issues["sub_false_positive"].append(msg)

    # ── Report per chunk ──────────────────────────────────────────
    if chunk_issues:
        print(f"\n  [Chunk {chunk_id:>3}] {section[:50]}")
        for issue in chunk_issues:
            print(f"    ⚠️  {issue}")
        total_warn += len(chunk_issues)
    else:
        total_pass += 1
        print(f"  [Chunk {chunk_id:>3}] OK  — {len(entities):>2} entities, {len(relationships):>2} relationships")


# ── Post-scan: chunk 171 — attribution grounding check ───────────
if 171 in chunk_results:
    _, rels_171 = chunk_results[171]
    INSURANCE_KEYWORDS = {"premium", "written premium", "earned premium"}
    boardwalk_insurance_rels = [
        r for r in rels_171
        if r.get("type") == "GENERATED"
        and "boardwalk" in (r.get("source") or "").lower()
        and any(kw in (r.get("property") or "").lower() for kw in INSURANCE_KEYWORDS)
    ]
    print(f"\n{'─' * 65}")
    print(f"  CHUNK 171 — Attribution grounding check (prompt fix)")
    print(f"{'─' * 65}")
    if not boardwalk_insurance_rels:
        print(f"  ✅ PASS  — Boardwalk Pipeline has 0 insurance-metric GENERATED rels (was 9 before fix)")
        total_pass += 1
    else:
        print(f"  ❌ FAIL  — Boardwalk Pipeline still has {len(boardwalk_insurance_rels)} insurance-metric GENERATED rel(s):")
        for r in boardwalk_insurance_rels:
            print(f"      {r['source']} --GENERATED--> {r['target']}  (property: {r.get('property')})")
        total_warn += len(boardwalk_insurance_rels)

# ── Post-scan: subsidiary extraction coverage ─────────────────────
print(f"\n{'─' * 65}")
print(f"  SUBSIDIARY EXTRACTION REPORT  ({len(ALL_CHUNK_IDS)} chunks scanned)")
print(f"{'─' * 65}")

print(f"\n  Coverage — confirmed subsidiaries seen at least once:")
for sub in sorted(confirmed_subs):
    count = confirmed_subs_seen.get(sub, 0)
    mark  = "✅" if count > 0 else "❌"
    print(f"  {mark}  '{sub}'  ({count} chunk(s))")

false_positives = issues["sub_false_positive"]
print(f"\n  False positives — extracted as Subsidiary but not in confirmed list:")
if not false_positives:
    print(f"  ✅  None")
else:
    for msg in false_positives:
        print(f"  ⚠️   {msg}")

# ══════════════════════════════════════════════════════
# SECTION 2 — UNIT TESTS (no API)
# ══════════════════════════════════════════════════════
print("\n" + "█" * 65)
print("  SECTION 2 — UNIT TESTS  (no API)")
print("█" * 65)

unit_pass = 0
unit_fail = 0

# ── 2a: Fix A — format-duplicate GENERATED dedup ─────────────────
print("\n--- 2a: Fix A — format-duplicate GENERATED dedup ---")

_dedup_graph = {
    "entities": [
        {"name": "CNA Financial Corporation", "type": "Subsidiary"},
        {"name": "$1,161 million",            "type": "Financial Item"},
        {"name": "$1,161",                    "type": "Financial Item"},
        {"name": "$982",                      "type": "Financial Item"},
        {"name": "$982 million",              "type": "Financial Item"},
    ],
    "relationships": [
        # Duplicate pair 1 — same metric, two target formats; "million" version should win
        {"source": "CNA Financial Corporation", "target": "$1,161 million", "type": "GENERATED",
         "property": "non-insurance warranty revenue fy2019"},
        {"source": "CNA Financial Corporation", "target": "$1,161",         "type": "GENERATED",
         "property": "non-insurance warranty revenue fy2019"},
        # Duplicate pair 2 — same metric, two target formats; "million" version should win
        {"source": "Diamond Offshore Drilling, Inc.", "target": "$982 million", "type": "GENERATED",
         "property": "contract drilling revenue fy2019"},
        {"source": "Diamond Offshore Drilling, Inc.", "target": "$982",          "type": "GENERATED",
         "property": "contract drilling revenue fy2019"},
        # Non-duplicate — same source/type but DIFFERENT property → both must survive
        {"source": "CNA Financial Corporation", "target": "$7,428", "type": "GENERATED",
         "property": "insurance premiums fy2019"},
    ],
}

_merged = merge_graph(_dedup_graph)
_rels   = _merged["relationships"]

# Test 1: total relationships after dedup = 3 (2 pairs collapsed + 1 unique)
if len(_rels) == 3:
    print(f"  ✅ PASS  — 5 rels → 3 after format-dedup (2 duplicates removed)")
    unit_pass += 1
else:
    print(f"  ❌ FAIL  — expected 3 relationships, got {len(_rels)}: {[(r['source'], r['target']) for r in _rels]}")
    unit_fail += 1

# Test 2: "million" target wins for pair 1
_pair1_targets = {r["target"] for r in _rels if r.get("property") == "non-insurance warranty revenue fy2019"}
if _pair1_targets == {"$1,161 million"}:
    print(f"  ✅ PASS  — '$1,161 million' kept, '$1,161' dropped")
    unit_pass += 1
else:
    print(f"  ❌ FAIL  — pair 1 targets: {_pair1_targets}")
    unit_fail += 1

# Test 3: "million" target wins for pair 2
_pair2_targets = {r["target"] for r in _rels if r.get("property") == "contract drilling revenue fy2019"}
if _pair2_targets == {"$982 million"}:
    print(f"  ✅ PASS  — '$982 million' kept, '$982' dropped")
    unit_pass += 1
else:
    print(f"  ❌ FAIL  — pair 2 targets: {_pair2_targets}")
    unit_fail += 1

# Test 4: the unique relationship (different property) survived untouched
_unique = [r for r in _rels if r.get("property") == "insurance premiums fy2019"]
if len(_unique) == 1 and _unique[0]["target"] == "$7,428":
    print(f"  ✅ PASS  — unique relationship (different property) survived untouched")
    unit_pass += 1
else:
    print(f"  ❌ FAIL  — unique relationship missing or wrong: {_unique}")
    unit_fail += 1


# ── 2b: _normalize_sub entity resolution ─────────────────────────
print("\n--- 2b: Subsidiary name normalisation ---")
from extractor import _normalize_sub

norm_cases = [
    # (variant,                              expected_normalized)
    ("Loews Hotels Holding Corporation",     "loews hotels"),
    ("Loews Hotels & Co",                    "loews hotels"),
    ("Loews Hotels",                         "loews hotels"),
    ("Boardwalk Pipeline Partners, LP",      "boardwalk pipeline"),
    ("Boardwalk Pipelines",                  "boardwalk pipeline"),
    ("Boardwalk Pipelines Holding Corp",     "boardwalk pipeline"),
    ("CNA Financial Corporation",            "cna financial"),
    ("CNA Financial",                        "cna financial"),
    ("Diamond Offshore Drilling, Inc.",      "diamond offshore drilling"),
    ("Diamond Offshore",                     "diamond offshore"),
    ("Altium Packaging LLC",                 "altium packaging"),
]

for name, expected in norm_cases:
    result = _normalize_sub(name)
    if result == expected:
        print(f"  ✅ PASS  — '{name}' → '{result}'")
        unit_pass += 1
    else:
        print(f"  ❌ FAIL  — '{name}' → '{result}'  (expected '{expected}')")
        unit_fail += 1

# ── 2e: Ownership guard — filing company context check ───────────
print("\n--- 2e: Ownership guard — competitor vs true subsidiary ---")


# Text 1: competitor in stock comparison table — "acquired" is near the company
# but the filing company (Loews) is NOT in the vicinity → should be dropped.
competitor_text = (
    "The following graph compares the five-year cumulative total return on common stock. "
    "The Peer Group consists of: Chubb Limited, ACE Limited (which was acquired by Chubb "
    "in 2016), W.R. Berkley Corporation, The Travelers Companies, Inc., and Transocean Ltd. "
    "Each company's stock return is indexed to $100 at the start of the period."
)
competitor_entities = [
    {"name": "Chubb Limited",                  "type": "Subsidiary"},
    {"name": "ACE Limited",                    "type": "Subsidiary"},
    {"name": "W.R. Berkley Corporation",       "type": "Subsidiary"},
    {"name": "The Travelers Companies, Inc.",  "type": "Subsidiary"},
    {"name": "Transocean Ltd.",                "type": "Subsidiary"},
]
result_competitors = apply_entity_guards(
    competitor_entities, competitor_text, filing_company, confirmed_subs
)
leftover = [e["name"] for e in result_competitors if e["type"] == "Subsidiary"]
if not leftover:
    print(f"  ✅ PASS  — all 5 competitor companies dropped (no filing company in ownership window)")
    unit_pass += 1
else:
    print(f"  ❌ FAIL  — {len(leftover)} competitor(s) survived: {leftover}")
    unit_fail += 1

# Text 2: true subsidiary — ownership language WITH filing company nearby → should survive.
subsidiary_text = (
    f"{filing_company} owns 100% of CNA Financial Corporation. "
    "CNA Financial Corporation is a wholly-owned subsidiary of Loews Corporation "
    "and is consolidated into its parent's financial statements."
)
subsidiary_entities = [{"name": "CNA Financial Corporation", "type": "Subsidiary"}]
result_sub = apply_entity_guards(
    subsidiary_entities, subsidiary_text, filing_company, confirmed_subs
)
survived = [e["name"] for e in result_sub if e["type"] == "Subsidiary"]
if "CNA Financial Corporation" in survived:
    print(f"  ✅ PASS  — 'CNA Financial Corporation' kept (filing company present in ownership window)")
    unit_pass += 1
else:
    print(f"  ❌ FAIL  — 'CNA Financial Corporation' was incorrectly dropped")
    unit_fail += 1

# Text 3: ownership language present but filing company is NOT nearby → should be dropped.
orphan_text = (
    "Boardwalk Pipeline was acquired by an unnamed private equity firm in 2005. "
    "Its operations span Louisiana and Texas. The pipeline network carries natural gas "
    "across several states under long-term contracts with industrial customers."
)
orphan_entities = [{"name": "Some Unrelated Corp", "type": "Subsidiary"}]
result_orphan = apply_entity_guards(
    orphan_entities, orphan_text, filing_company, set()  # empty confirmed set
)
survived_orphan = [e["name"] for e in result_orphan if e["type"] == "Subsidiary"]
if not survived_orphan:
    print(f"  ✅ PASS  — unrelated subsidiary dropped (filing company absent from ownership window)")
    unit_pass += 1
else:
    print(f"  ❌ FAIL  — unrelated subsidiary survived: {survived_orphan}")
    unit_fail += 1

# ══════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)

print(f"\n  Section 1 — Full scan ({len(ALL_CHUNK_IDS)} chunks):")
print(f"    Clean chunks   : {total_pass}")
print(f"    Chunks w/issues: {len(ALL_CHUNK_IDS) - total_pass}")
print(f"    Total warnings : {total_warn}")

print(f"\n  Issue breakdown:")
for category, items in issues.items():
    label = category.replace("_", " ").title()
    mark  = "⚠️ " if items else "✅"
    print(f"    {mark} {label:<33}: {len(items)}")

print(f"\n  Section 2 — Unit tests:")
print(f"    ✅ Passed : {unit_pass}")
print(f"    ❌ Failed : {unit_fail}")

print("=" * 65)
