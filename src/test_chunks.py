import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from extractor import detect_filing_company, enrich_financial_items
from guards import (
    apply_entity_guards, validate_relationships,
    PAREN_NEGATIVE_PATTERN, REGULATORY_BODY_PATTERN,
    FINANCIAL_VALUE_PATTERN, _normalize_sub
)
from agents import run_pipeline

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
COLLECTION_NAME = "LoewsCompany"

VALID_REL_TYPES     = {"PARENT_OF", "OPERATES_IN", "REPORTED", "GENERATED"}
FINANCIAL_REL_TYPES = {"REPORTED", "GENERATED"}

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
    5,    # regulatory body fix — NAIC, IAIS, State of Illinois DOI
    21,   # regulatory body fix — NAIC, Luxembourg
    33,   # duplicate entity fix — Boardwalk Pipeline Partners LP x7
    59,   # Loews Hotels variants — Holding Corporation / & Co
    127,  # Geography too broad — Europe, Canada
    145,  # Boardwalk Pipelines name variant
    0,    # Parent normalisation — LOEWS CORPORATION vs Loews Corporation
    49,   # financial tables — GENERATED/REPORTED mix
    45,   # competitor companies — stock performance comparison table (Chubb, Travelers, etc.)
    7,    # Diamond Offshore Drilling coverage — first chunk where it appears
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

# Accumulators for enrichment (Section 3)
all_collected_entities:      list[dict] = []
all_collected_relationships: list[dict] = []

# Issue trackers for summary
issues: dict[str, list] = {
    "geo_regulatory_body":   [],   # Geography classified as regulatory body
    "geo_too_broad":         [],   # Geography too vague (region/bloc)
    "invalid_rel_type":      [],   # Unknown relationship type
    "self_referential":      [],   # source == target
    "bad_reported_source":   [],   # REPORTED source is not Parent
    "bad_parent_of_source":  [],   # PARENT_OF source is not Parent
    "fi_target_invalid":     [],   # Financial rel target not FI and not auto-registerable
    "fi_rel_no_fi_entity":   [],   # Financial rels produced with no FI entities
    "sub_false_positive":    [],   # Subsidiary entity not matching any confirmed canonical
}

# Precompute normalised keys for confirmed subsidiaries once
confirmed_norms: dict[str, str] = {_normalize_sub(s): s for s in confirmed_subs}

# Tracks which confirmed subsidiaries were seen at least once across all chunks
confirmed_subs_seen: dict[str, int] = {s: 0 for s in confirmed_subs}

TOO_BROAD = re.compile(
    r'^(north america|south america|latin america|europe|asia|africa|'
    r'middle east|oceania|apac|emea|worldwide|global|rest of world|other)$',
    re.IGNORECASE
)

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

    all_collected_entities.extend(entities)
    all_collected_relationships.extend(relationships)

    chunk_issues = []

    # Build entity type map (include injected Parent)
    entity_type_map = {e["name"]: e["type"] for e in entities}
    entity_type_map[filing_company] = "Parent"
    fi_names = {e["name"] for e in entities if e["type"] == "Financial Item"}

    # ── Check 1: Geography — regulatory body slipthrough ─────────
    for e in entities:
        if e["type"] == "Geography" and REGULATORY_BODY_PATTERN.search(e["name"]):
            msg = f"Chunk {chunk_id}: [Geography] '{e['name']}' looks like a regulatory body"
            chunk_issues.append(msg)
            issues["geo_regulatory_body"].append(msg)

    # ── Check 2: Geography — too broad ───────────────────────────
    for e in entities:
        if e["type"] == "Geography" and TOO_BROAD.match(e["name"].strip()):
            msg = f"Chunk {chunk_id}: [Geography] '{e['name']}' is too broad (region/bloc)"
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

    # ── Check 5: REPORTED source must be Parent ───────────────────
    for r in relationships:
        if r.get("type") == "REPORTED" and entity_type_map.get(r.get("source")) != "Parent":
            msg = f"Chunk {chunk_id}: REPORTED source '{r.get('source')}' is not a Parent"
            chunk_issues.append(msg)
            issues["bad_reported_source"].append(msg)

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

# ── 2a: Parenthetical negative normalization ─────────────────────
print("\n--- 2a: Parenthetical negative normalization ---")

paren_cases = [
    ("(72,880)",       "-72,880",       True),
    ("($1,200)",       "-$1,200",       True),
    ("(1.2 billion)",  "-1.2 billion",  True),
    ("(0.5%)",         "-0.5%",         True),
    ("($44,870)",      "-$44,870",      True),
    ("(500)",          "-500",          True),
    ("$(31)",          "-$31",          True),
    ("$(1,200)",       "-$1,200",       True),
    ("$(69) million",  "-$69 million",  True),   # scale word outside parens
    ("$(73) million",  "-$73 million",  True),   # scale word outside parens
    ("(28) million",   "-28 million",   True),   # scale word outside parens, no $
    ("72,880",         None,            False),
    ("$1,200",         None,            False),
    ("(some text)",    None,            False),
    ("()",             None,            False),
]

for raw, expected_out, should_match in paren_cases:
    name = raw.strip()
    # Mirror the normalization logic in extractor.py
    m_outside = re.match(
        r'^(\$?)\((\d[\d,]*(?:\.\d+)?)\)\s*(billion|million|trillion|thousand)',
        name, re.IGNORECASE
    )
    if m_outside:
        prefix, digits, scale = m_outside.groups()
        name = f"({prefix}{digits} {scale})"
    elif re.match(r'^\$\(', name):
        name = "($" + name[2:]
    m = PAREN_NEGATIVE_PATTERN.match(name)
    if should_match:
        if m:
            result = "-" + m.group(1)
            if result == expected_out:
                print(f"  ✅ PASS  — '{raw}' → '{result}'")
                unit_pass += 1
            else:
                print(f"  ❌ FAIL  — '{raw}' → '{result}'  (expected '{expected_out}')")
                unit_fail += 1
        else:
            print(f"  ❌ FAIL  — '{raw}' did not match but should have")
            unit_fail += 1
    else:
        if not m:
            print(f"  ✅ PASS  — '{raw}' correctly not matched")
            unit_pass += 1
        else:
            print(f"  ❌ FAIL  — '{raw}' matched but should NOT have")
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

# ── 2c: enrich_financial_items() ─────────────────────────────────
print("\n--- 2c: enrich_financial_items() ---")

mock_entities = [
    {"name": filing_company,                 "type": "Parent",         "chunk_id": 1},
    {"name": "CNA Financial Corporation",    "type": "Subsidiary",     "chunk_id": 1},
    {"name": "$72,880",                      "type": "Financial Item", "chunk_id": 5},
    {"name": "44,870",                       "type": "Financial Item", "chunk_id": 5},
    {"name": "$500",                         "type": "Financial Item", "chunk_id": 6},
    {"name": "$200",                         "type": "Financial Item", "chunk_id": 7},
]
mock_relationships = [
    {"source": filing_company,            "target": "$72,880", "type": "REPORTED",   "property": "total revenue fy2024"},
    {"source": "CNA Financial Corporation","target": "44,870", "type": "GENERATED",  "property": "net income fy2023"},
    {"source": filing_company,            "target": "$500",    "type": "HAS_METRIC", "property": "long-term debt fy2024"},
    {"source": filing_company,            "target": "$200",    "type": "HAS_METRIC", "property": None},
    {"source": filing_company,            "target": "CNA Financial Corporation", "type": "PARENT_OF", "property": None},
]

enriched_e, enriched_r = enrich_financial_items(
    [dict(e) for e in mock_entities],
    [dict(r) for r in mock_relationships]
)
enriched_names = [e["name"] for e in enriched_e]

checks_2c = [
    (next((r for r in enriched_r if r["type"] == "REPORTED"), None),
     lambda r: r and r["target"] == "Total Revenue: $72,880" and r["property"] == "2024",
     "REPORTED renamed to 'Total Revenue: $72,880', property='2024'"),
    (next((r for r in enriched_r if r["type"] == "GENERATED"), None),
     lambda r: r and r["target"] == "Net Income: 44,870" and r["property"] == "2023",
     "GENERATED renamed to 'Net Income: 44,870', property='2023'"),
    (next((r for r in enriched_r if r["type"] == "HAS_METRIC" and r.get("property") == "2024"), None),
     lambda r: r and r["target"] == "Long-Term Debt: $500",
     "HAS_METRIC renamed to 'Long-Term Debt: $500', property='2024'"),
    (next((r for r in enriched_r if r["type"] == "HAS_METRIC" and r.get("property") is None), None),
     lambda r: r and r["target"] == "$200",
     "HAS_METRIC with null property left unchanged"),
    ("$200" in enriched_names,       lambda x: x,  "raw '$200' kept (still referenced)"),
    ("$72,880" not in enriched_names, lambda x: x,  "raw '$72,880' removed (fully replaced)"),
    ("Total Revenue: $72,880" in enriched_names, lambda x: x, "new node 'Total Revenue: $72,880' exists"),
    (next((r for r in enriched_r if r["type"] == "PARENT_OF"), None),
     lambda r: r and r["source"] == filing_company and r["target"] == "CNA Financial Corporation",
     "PARENT_OF unchanged by enrichment"),
]

for val, check, label in checks_2c:
    if check(val):
        print(f"  ✅ PASS  — {label}")
        unit_pass += 1
    else:
        print(f"  ❌ FAIL  — {label}")
        unit_fail += 1

# ── 2d: FINANCIAL_VALUE_PATTERN — negative values ────────────────
print("\n--- 2d: FINANCIAL_VALUE_PATTERN negative value matching ---")

neg_cases = [
    # (value,               should_match)
    ("-$224",               True),
    ("-$224 million",       True),
    ("-$1,200",             True),
    ("-$1.4 billion",       True),
    ("-224",                True),
    ("-1,200",              True),
    ("-72,880",             True),
    ("-12.5%",              True),
    ("-0.5",                True),
    # $-value format (Agent 3 sometimes outputs dollar before minus)
    # — these are normalised to -$value before pattern matching
    ("-$112",               True),   # already correct after normalisation
    ("-$161",               True),
    # positives must still match
    ("$224",                True),
    ("224",                 True),
    ("72,880",              True),
    # non-values must still not match
    ("-",                   False),
    ("--",                  False),
    ("-revenue",            False),
    ("net income",          False),
]

for val, should_match in neg_cases:
    matched = bool(FINANCIAL_VALUE_PATTERN.match(str(val)))
    ok_flag = matched == should_match
    symbol  = "✅ PASS" if ok_flag else "❌ FAIL"
    expected_lbl = "match" if should_match else "no match"
    print(f"  {symbol}  — '{val}'  (expected {expected_lbl})")
    if ok_flag:
        unit_pass += 1
    else:
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

# ── 2f: Source alias resolution in validate_relationships ─────────
print("\n--- 2f: Source alias resolution ($-value + subsidiary alias) ---")

from extractor import validate_relationships

# Build a minimal entity set matching confirmed subsidiaries
mock_entity_names    = {e["name"] for e in KNOWN_ENTITIES} | {filing_company}
mock_entity_type_map = {e["name"]: e["type"] for e in KNOWN_ENTITIES}
mock_entity_type_map[filing_company] = "Parent"
mock_all_entities    = list(KNOWN_ENTITIES)

# Test 1 — "Boardwalk Pipelines" alias resolved to canonical via _normalize_sub
alias_rels = [
    {"source": "Boardwalk Pipelines", "target": "$96 million",  "type": "GENERATED", "property": "revenue fy2019"},
    {"source": "Boardwalk Pipelines", "target": "$416 million", "type": "GENERATED", "property": "revenue fy2018"},
]
# Pre-register the targets so they exist in entity_names
for r in alias_rels:
    mock_entity_names.add(r["target"])
    mock_all_entities.append({"name": r["target"], "type": "Financial Item", "chunk_id": 99})

valid_alias = validate_relationships(
    [dict(r) for r in alias_rels],
    set(mock_entity_names), dict(mock_entity_type_map),
    chunk_id=99, file="test", page_number=0,
    run_timestamp="test", all_entities=mock_all_entities
)
if len(valid_alias) == 2 and all(r["source"] == "Boardwalk Pipeline Partners, LP" for r in valid_alias):
    print(f"  ✅ PASS  — 'Boardwalk Pipelines' resolved to canonical in {len(valid_alias)} relationship(s)")
    unit_pass += 1
else:
    resolved   = [r["source"] for r in valid_alias]
    unresolved = [r["source"] for r in alias_rels if r not in valid_alias]
    print(f"  ❌ FAIL  — resolved: {resolved}  dropped: {len(alias_rels) - len(valid_alias)}")
    unit_fail += 1

# Test 2 — "$-112" normalised to "-$112" and then auto-registered as Financial Item
dollar_minus_rels = [
    {"source": filing_company, "target": "$-112", "type": "HAS_METRIC", "property": "net loss fy2019"},
    {"source": filing_company, "target": "$-161", "type": "HAS_METRIC", "property": "net loss fy2018"},
]
entity_names_dm    = {filing_company}
entity_type_map_dm = {filing_company: "Parent"}
all_entities_dm    = [{"name": filing_company, "type": "Parent"}]

valid_dm = validate_relationships(
    [dict(r) for r in dollar_minus_rels],
    entity_names_dm, entity_type_map_dm,
    chunk_id=99, file="test", page_number=0,
    run_timestamp="test", all_entities=all_entities_dm
)
if len(valid_dm) == 2 and all(r["target"].startswith("-$") for r in valid_dm):
    print(f"  ✅ PASS  — '$-112' / '$-161' normalised to '-$112' / '-$161' and auto-registered")
    unit_pass += 1
else:
    print(f"  ❌ FAIL  — expected 2 valid rels with -$ prefix, got: {[(r['target']) for r in valid_dm]}")
    unit_fail += 1

# Test 3 — bare parenthetical "(57)" normalised to "-57" and auto-registered
paren_rels = [
    {"source": "CNA Financial Corporation", "target": "(57)",  "type": "HAS_METRIC", "property": "investment gains losses fy2018"},
    {"source": "CNA Financial Corporation", "target": "(151)", "type": "HAS_METRIC", "property": "income tax expense fy2018"},
    {"source": "CNA Financial Corporation", "target": "$(224)","type": "HAS_METRIC", "property": "income tax expense fy2019"},
]
entity_names_p    = {"CNA Financial Corporation", filing_company}
entity_type_map_p = {"CNA Financial Corporation": "Subsidiary", filing_company: "Parent"}
all_entities_p    = [
    {"name": "CNA Financial Corporation", "type": "Subsidiary"},
    {"name": filing_company,              "type": "Parent"},
]

valid_p = validate_relationships(
    [dict(r) for r in paren_rels],
    entity_names_p, entity_type_map_p,
    chunk_id=99, file="test", page_number=0,
    run_timestamp="test", all_entities=all_entities_p
)
expected_targets = {"-57", "-151", "-$224"}
actual_targets   = {r["target"] for r in valid_p}
if len(valid_p) == 3 and actual_targets == expected_targets:
    print(f"  ✅ PASS  — '(57)' → '-57', '(151)' → '-151', '$(224)' → '-$224' — all 3 normalised and saved")
    unit_pass += 1
else:
    print(f"  ❌ FAIL  — expected {expected_targets}, got {actual_targets}  ({len(valid_p)}/3 saved)")
    unit_fail += 1

# ══════════════════════════════════════════════════════
# SECTION 3 — ENRICHMENT ON REAL DATA
# ══════════════════════════════════════════════════════
print("\n" + "█" * 65)
print("  SECTION 3 — enrich_financial_items() ON REAL DATA")
print("█" * 65)

if not any(e["name"] == filing_company and e["type"] == "Parent"
           for e in all_collected_entities):
    all_collected_entities.append({"name": filing_company, "type": "Parent"})

fi_before = [e for e in all_collected_entities if e["type"] == "Financial Item"]
print(f"\nBefore enrichment:")
print(f"  Total entities      : {len(all_collected_entities)}")
print(f"  Total relationships : {len(all_collected_relationships)}")
print(f"  Financial Items     : {len(fi_before)}")

enriched_entities, enriched_rels = enrich_financial_items(
    [dict(e) for e in all_collected_entities],
    [dict(r) for r in all_collected_relationships]
)

fi_after  = [e for e in enriched_entities if e["type"] == "Financial Item"]
fin_rels  = [r for r in enriched_rels if r["type"] in FINANCIAL_REL_TYPES]

print(f"\nAfter enrichment:")
print(f"  Total entities      : {len(enriched_entities)}")
print(f"  Total relationships : {len(enriched_rels)}")
print(f"  Financial Item nodes: {len(fi_after)}")

print(f"\nSample — Financial Item nodes (first 20):")
for e in fi_after[:20]:
    print(f"  [{e['type']}] {e['name']}")

print(f"\nSample — Financial relationships (first 20):")
for r in fin_rels[:20]:
    print(f"  {r['source']} --{r['type']}--> {r['target']}  (year: {r.get('property')})")

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

print(f"\n  Section 3 — Enrichment:")
print(f"    FI nodes before : {len(fi_before)}")
print(f"    FI nodes after  : {len(fi_after)}")

print("=" * 65)
