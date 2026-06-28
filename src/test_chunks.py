"""
Targeted test: BOARD_MEMBER_OF relationship extraction only.

Runs the multi-agent pipeline on the chunk(s) that contain executive officers
or board members in the source 10-K, then reports the Person entities and
BOARD_MEMBER_OF relationships that were extracted.

Generic — no hardcoded chunk IDs, person names, or company names. The chunk
selection scans the Qdrant collection for sections matching governance markers
(executive officers, directors, board), so this test works on any 10-K ingested
into the same collection.

Run:
    cd src && python test_chunks.py
"""

import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

from extractor import detect_filing_company
from guards import apply_entity_guards
from agents import run_pipeline
from query1 import qdrant_client

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
COLLECTION_NAME = "LoewsCompany"

# Section / text markers that identify a chunk likely to contain board members
# or executive officers. Generic — matches the standard 10-K Item 10 wording.
GOVERNANCE_MARKERS = re.compile(
    r"executive officers|board of directors|directors of the registrant|"
    r"information about our (executive )?officers|"
    r"chairman of the board|chief executive officer",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
client_oai    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_qdrant = qdrant_client()

# Detect filing company from the first few chunks
first_points = client_qdrant.retrieve(
    collection_name=COLLECTION_NAME,
    ids=[0, 1, 2],
    with_payload=True,
    with_vectors=False,
)
first_chunks   = [{"text": p.payload["text"]} for p in sorted(first_points, key=lambda p: p.id)]
filing_company = detect_filing_company(first_chunks, client_oai)

print("█" * 65)
print("  TARGETED TEST — BOARD_MEMBER_OF EXTRACTION")
print("█" * 65)
print(f"\nFiling company  : {filing_company}")
print(f"Collection      : {COLLECTION_NAME}\n")


# ─────────────────────────────────────────────
# STEP 1 — locate governance-relevant chunks dynamically
# ─────────────────────────────────────────────
print("─" * 65)
print("  Scanning chunks for governance markers...")
print("─" * 65)

all_points = client_qdrant.scroll(
    collection_name=COLLECTION_NAME,
    limit=10000,
    with_payload=True,
    with_vectors=False,
)[0]

governance_chunks = []
for p in all_points:
    text    = p.payload.get("text", "")
    section = p.payload.get("section", "")
    page    = p.payload.get("page_number")
    if GOVERNANCE_MARKERS.search(text) or GOVERNANCE_MARKERS.search(section):
        governance_chunks.append({
            "chunk_id":    p.id,
            "page":        page,
            "section":     section,
            "text":        text,
        })

# Sort by chunk_id so the report is in document order
governance_chunks.sort(key=lambda c: c["chunk_id"])

if not governance_chunks:
    print("  No chunks matched the governance markers — nothing to test.")
    raise SystemExit(0)

print(f"  Found {len(governance_chunks)} candidate chunk(s):")
for c in governance_chunks:
    section_short = (c["section"] or "")[:60]
    print(f"    chunk_id={c['chunk_id']:>4}  page={c['page']:>4}  section='{section_short}'")
print()


# ─────────────────────────────────────────────
# STEP 2 — run the pipeline on each governance chunk
# ─────────────────────────────────────────────
# Known entities act as the cross-chunk memory the real extractor builds up.
# Here we seed it with the parent so the classifier knows the filing company,
# and let any discovered subsidiaries surface from the chunk itself.
KNOWN_ENTITIES = [
    {"name": filing_company, "type": "Parent"},
]
confirmed_subs: set = set()

# Aggregate results across all governance chunks
all_persons:   list[dict] = []
all_board_rels: list[dict] = []

for c in governance_chunks:
    print("─" * 65)
    print(f"  Pipeline run — chunk_id={c['chunk_id']}, page={c['page']}")
    print("─" * 65)

    entities, relationships = run_pipeline(
        text           = c["text"],
        filing_company = filing_company,
        client         = client_oai,
        guard_fn       = lambda ents, t=c["text"]: apply_entity_guards(
            ents, t, filing_company, confirmed_subs
        ),
        run_timestamp  = "test_board",
        chunk_id       = c["chunk_id"],
        known_entities = KNOWN_ENTITIES,
    )

    persons    = [e for e in entities      if e.get("type") == "Person"]
    board_rels = [r for r in relationships if r.get("type") == "BOARD_MEMBER_OF"]

    # Tag results with the source page so the final report can show it
    for p in persons:
        p["_page"] = c["page"]
    for r in board_rels:
        r["_page"] = c["page"]

    all_persons.extend(persons)
    all_board_rels.extend(board_rels)

    print(f"\n  ⮕ Person entities extracted   : {len(persons)}")
    print(f"  ⮕ BOARD_MEMBER_OF edges       : {len(board_rels)}\n")


# ─────────────────────────────────────────────
# STEP 3 — report
# ─────────────────────────────────────────────
print("█" * 65)
print("  RESULTS")
print("█" * 65)

print(f"\nTotal Person entities      : {len(all_persons)}")
print(f"Total BOARD_MEMBER_OF edges: {len(all_board_rels)}\n")

if all_persons:
    print("── Persons ─────────────────────────────────────────────────────")
    seen = set()
    for p in all_persons:
        if p["name"] in seen:
            continue
        seen.add(p["name"])
        print(f"  • {p['name']:<40} (page {p['_page']})")
    print()

if all_board_rels:
    print("── BOARD_MEMBER_OF relationships ───────────────────────────────")
    print(f"  {'Person':<35}  {'Company':<35}  {'Role':<50}  Page")
    print(f"  {'-'*35}  {'-'*35}  {'-'*50}  ----")
    for r in all_board_rels:
        person = (r.get("source") or "")[:35]
        target = (r.get("target") or "")[:35]
        role   = (r.get("property") or "")[:50]
        page   = r.get("_page", "?")
        print(f"  {person:<35}  {target:<35}  {role:<50}  {page}")
    print()
else:
    print("  ⚠ No BOARD_MEMBER_OF edges were produced.")
    print("    Possible reasons:")
    print("      - Persons were classified but the role text didn't match the prompt's trigger list")
    print("      - The filing company name in the chunk didn't match the Parent entity name")
    print("      - No Subsidiary entities were available in the chunk as edge targets\n")
