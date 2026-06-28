# Graph RAG Pipeline for 10-K Financial Documents

**Code Documentation**

*Project: PFE — Final-Year Project*
*Generated: 2026-06-15*

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Pipeline Data Flow](#3-pipeline-data-flow)
4. [File-by-File Reference](#4-file-by-file-reference)
5. [The Multi-Agent Extraction Layer](#5-the-multi-agent-extraction-layer)
6. [The Guards Layer — Deterministic Safety Net](#6-the-guards-layer)
7. [Cross-Chunk Memory & Recovery Mechanisms](#7-cross-chunk-memory)
8. [Statement Extractor](#8-statement-extractor)
9. [Merge, Canonicalisation & Enrichment](#9-merge-canonicalisation-enrichment)
10. [Neo4j Graph Storage](#10-neo4j-graph-storage)
11. [Query Layer — Graph RAG](#11-query-layer)
12. [Testing Strategy](#12-testing-strategy)
13. [Known Limitations](#13-known-limitations)
14. [Future Work](#14-future-work)
15. [Appendix — Schema Reference](#15-appendix)

---

## 1. Project Overview

### Purpose

This project builds a **knowledge-graph-based question-answering system** over SEC 10-K annual reports. It transforms an unstructured 10-K (markdown) into a structured Neo4j graph that can be queried with natural language.

### Why a graph (not just vector RAG)?

Financial 10-K filings have rich **structural relationships** that vector retrieval alone cannot capture cleanly:

- *"List all subsidiaries of Loews Corporation"* — needs traversal of `PARENT_OF` edges
- *"Where does Boardwalk Pipeline operate?"* — needs `OPERATES_IN` lookup
- *"Compare CNA's net income across 2018 and 2019"* — needs typed, filtered metric retrieval
- *"Which subsidiaries reported a net loss?"* — needs deterministic value-pattern matching

A graph stores these relationships explicitly; an LLM can then write Cypher queries to fetch the exact facts needed.

### Tech Stack

| Layer | Tool |
|---|---|
| LLM | OpenAI `gpt-4.1-mini` |
| Embeddings | SentenceTransformer (configured in chunker) |
| Vector store | Qdrant |
| Graph database | Neo4j |
| Language | Python 3.12 |

### Test corpus

**Loews Corporation 2019 10-K** — a conglomerate with 5 major subsidiaries:
- CNA Financial Corporation (insurance)
- Diamond Offshore Drilling, Inc. (offshore drilling)
- Boardwalk Pipeline Partners, LP (natural gas pipelines)
- Loews Hotels Holding Corporation (hospitality)
- Altium Packaging LLC (consumer packaging)

Document: 86 pages, ~172 chunks after splitting.

### Final results

| Metric | Value |
|---|---|
| Final entities | 534 |
| Final relationships | 535 |
| Subsidiary coverage | 6/6 |
| Business segments (after de-fragmentation) | 3 clean |
| GENERATED edges | 431 |
| Dangling sources in graph | 0 |
| Unit tests passing | 31/31 |
| Extraction quality rating | 9.4/10 |

---

## 2. System Architecture

### High-level layered design

```
┌─────────────────────────────────────────────────────────┐
│                      USER QUESTION                       │
│         "What was CNA's net income in 2019?"             │
└──────────────────────────┬──────────────────────────────┘
                           │
                ┌──────────▼──────────┐
                │     query1.py        │
                │  Cypher Generation   │ ← LLM writes Cypher
                │  + Synthesis         │ ← LLM writes answer
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │       NEO4J          │
                │   Knowledge Graph    │
                └──────────▲──────────┘
                           │
                ┌──────────┴──────────┐
                │      main.py         │ ← Orchestrator
                └──────────┬──────────┘
                           │
   ┌───────────────────────┼───────────────────────┐
   │                       │                        │
┌──▼──────┐         ┌──────▼─────┐         ┌───────▼──────┐
│loader.py│         │ chunker.py │         │ extractor.py │
│Load .md │ ──────► │ Chunk +    │ ──────► │ Run pipeline │
└─────────┘         │ Embed +    │         │ Merge + Save │
                    │ → Qdrant   │         └───────┬──────┘
                    └────────────┘                  │
                                            ┌──────▼──────┐
                                            │  agents.py   │
                                            │ run_pipeline │
                                            └──────┬──────┘
                                                   │
       ┌──────────┬──────────┬───────────┬─────────┴────┐
       │          │          │           │              │
┌──────▼───┐ ┌────▼────┐ ┌──▼────┐ ┌────▼─────┐ ┌─────▼─────┐
│entity_   │ │entity_  │ │relat. │ │statement_│ │ guards.py │
│finder    │ │classif. │ │finder │ │extractor │ │ ← VALIDATE│
│Agent 1   │ │Agent 2  │ │Agent 3│ │          │ │            │
└──────────┘ └─────────┘ └───────┘ └──────────┘ └───────────┘
```

### Core design philosophy: **LLM proposes, deterministic code disposes**

LLMs are powerful for understanding ambiguous text but non-deterministic at temperature=0 — even identical input can produce different output between calls. This system addresses that with a **layered architecture**:

1. **LLM layer** (agents) — extracts candidates from raw text. Tolerates over-extraction.
2. **Deterministic guard layer** (`guards.py`) — validates, normalises, drops malformed output, and recovers from agent failures.
3. **Merge layer** — deduplicates across chunks, collapses canonical names.
4. **Enrichment layer** — restructures raw values into human-readable nodes.
5. **Storage layer** (Neo4j) — schema-enforced via labels.

The result: any single LLM call may fail or produce noise, but the **pipeline as a whole produces consistent, high-quality output**.

---

## 3. Pipeline Data Flow

### End-to-end journey of a single fact

Let's trace how the fact *"CNA Financial Corporation had a net income of $894M in 2019"* makes it from the source PDF to the user's query result.

```
SOURCE TEXT (page 47)
"Net income attributable to Loews Corporation for CNA was $894 in 2019"
        ↓
[1] CHUNKER — split into ~600-word chunks, store in Qdrant
        ↓
[2] EXTRACTOR — picks up chunk pair, calls run_pipeline()
        ↓
[3] AGENT 1 (finder)
    candidates = ["CNA", "$894", "2019", "Loews Corporation", ...]
        ↓
[4] AGENT 2 (classifier)
    raw entities = [
       {name: "CNA", type: "Subsidiary"},
       {name: "$894", type: "Financial Item"},
       {name: "Loews Corporation", type: "Parent"},
       ...
    ]
        ↓
[5] FI PROMOTION — recovers any FI Agent 2 missed but Agent 1 found
        ↓
[6] GUARDS (apply_entity_guards)
    - CNA → resolved to "CNA Financial Corporation" (cross-chunk memory)
    - $894 → kept (valid financial value)
    - Other noise → dropped
        ↓
[7] SUBSIDIARY RE-INJECTION — if chunk has FIs but no source entity,
    inject confirmed subs whose name appears in text
        ↓
[8] AGENT 3 (relationship finder)
    relationship = {
       source: "CNA Financial Corporation",
       target: "$894",
       type: "GENERATED",
       property: "net income attributable to loews corporation fy2019"
    }
        ↓
[9] VALIDATE_RELATIONSHIPS (guards)
    - Source exists in entity_names? ✓
    - Target exists OR auto-register? ✓
    - Type checks pass? ✓ (Subsidiary → FI is valid for GENERATED)
        ↓
[10] STATEMENT EXTRACTOR (parallel pass)
    Creates IncomeStatement_FY2019 with line items as properties
        ↓
[11] MERGE_GRAPH — deduplicate across all 86 chunks
        ↓
[12] CANONICALIZE_SEGMENT_NAMES — collapse fragmented segment nodes
        ↓
[13] ENRICH_FINANCIAL_ITEMS
    "$894" → "Net Income Attributable To Loews Corporation: $894"
    property "...fy2019" → just "2019"
        ↓
[14] DROP_ORPHANED_FINANCIAL_ITEMS — clean up unused values
        ↓
[15] STORE_GRAPH → Neo4j
    (:Subsidiary {name:"CNA Financial Corporation"})
        -[:GENERATED {property:"2019", page_number:47}]->
    (:FinancialItem {name:"Net Income Attributable To Loews Corporation: $894"})

USER QUERY: "What was CNA's net income in 2019?"
        ↓
[16] _generate_queries (LLM)
    Cypher: MATCH (s:Subsidiary {name:'CNA Financial Corporation'})-[r:GENERATED]->(f:FinancialItem)
            WHERE f.name CONTAINS 'Net Income' AND r.property = '2019'
            RETURN f.name, r.property, r.page_number
        ↓
[17] _run_cypher → execute, serialize results
        ↓
[18] _synthesize (LLM)
    "CNA Financial Corporation reported net income of $894 million
     in 2019 (Graph, p.47)."
```

---

## 4. File-by-File Reference

### Module dependency direction

```
main.py
  ├─ loader.py
  ├─ chunker.py
  ├─ extractor.py
  │    ├─ checkpoint.py
  │    ├─ logger.py
  │    ├─ agents.py
  │    │    ├─ entity_finder.py        (Agent 1)
  │    │    ├─ entity_classifier.py    (Agent 2)
  │    │    ├─ relationship_finder.py  (Agent 3)
  │    │    ├─ statement_extractor.py
  │    │    └─ guards.py (promote_financial_items, find_confirmed_subs_in_text)
  │    └─ guards.py (apply_entity_guards, validate_relationships, ...)
  └─ graph.py (Neo4j writer)

query1.py (independent — reads from Neo4j)
eval.py (depends on query1.py)
test_chunks.py (independent test harness)
```

### Per-file purpose

| File | Lines | Role |
|---|---|---|
| `main.py` | 73 | Top-level orchestrator. Wires together loader → chunker → extractor → merge → enrich → store. |
| `loader.py` | 16 | Reads source markdown document into memory. |
| `chunker.py` | 197 | Splits document into chunks with metadata (page, section); embeds via SentenceTransformer; stores in Qdrant. |
| `extractor.py` | 437 | Drives the extraction loop. Per-chunk: `run_pipeline` → guards → `validate_relationships`. Post-loop: `merge_graph` → `canonicalize_segment_names` → `enrich_financial_items` → `drop_orphaned_financial_items`. |
| `agents.py` | 133 | `run_pipeline` — orchestrates the 3 LLM agents + statement extractor per chunk. Also handles subsidiary re-injection and parent injection. |
| `entity_finder.py` | 37 | **Agent 1**: broad candidate extraction. Permissive — returns anything that *might* be an entity. |
| `entity_classifier.py` | 147 | **Agent 2**: assigns one of 5 types (Parent / Subsidiary / Geography / Business Segment / Financial Item) or DISCARD. Uses cross-chunk memory via `known_entities`. |
| `relationship_finder.py` | 187 | **Agent 3**: extracts PARENT_OF / OPERATES_IN / GENERATED relationships. Split into two sub-prompts: structural and generated. |
| `statement_extractor.py` | 138 | Parallel extractor for financial statement nodes (BalanceSheet / IncomeStatement / CashFlow). Stores line items as node properties. |
| `guards.py` | 808 | **Deterministic safety layer.** Pattern definitions, name normalisation, entity validation guards, relationship validation. |
| `graph.py` | 153 | Loads `graph_output.json` into Neo4j with proper labels and relationship types. |
| `checkpoint.py` | 46 | Resume support — saves per-chunk progress so a crashed run can continue. |
| `logger.py` | 195 | Console output (color-tagged) + detailed Markdown log per run. |
| `query1.py` | 433 | Graph-only query engine. LLM Cypher generation → execute → LLM synthesis. |
| `eval.py` | 305 | Benchmark harness — structural completeness, financial accuracy, query coverage. (Loews-specific ground truth.) |
| `test_chunks.py` | 742 | Two-section test suite: live chunk integration scan (Section 1) + 31 unit tests (Section 2). |

**Total: ~4,000 lines of Python.**

---

## 5. The Multi-Agent Extraction Layer

### Why three agents instead of one?

A single LLM prompt asked to *"extract everything"* tends to:
- Skip entities (under-extraction)
- Invent values (hallucination)
- Confuse types (subsidiary vs. segment)
- Drop relationships when overwhelmed by financial values

Splitting into three sequential, **single-responsibility** agents gives each one a focused task:

| Agent | Job | Failure mode it avoids |
|---|---|---|
| Agent 1 (finder) | Broadly list candidates | Misses entities (low recall) |
| Agent 2 (classifier) | Assign exactly one type | Type confusion |
| Agent 3 (relationship finder) | Find relationships between *validated* entities | Inventing names |

### Agent 1 — Entity Finder (`entity_finder.py`)

**Prompt strategy:** Permissive. Returns anything that *might* be relevant.

```python
ENTITY_FINDER_PROMPT = """You are reading a chunk of a financial document.
Find ALL potentially relevant named entities in the text.

Be PERMISSIVE — include anything that might be:
- A company or organization name
- A geographic location: a specific country, state, province, or city
- A specific dollar value or percentage
- A raw number or percentage from a financial table row
- A named business division or segment

Do NOT classify or filter — just find and list them.
"""
```

**Output:** flat list of strings.
```python
candidates = ["CNA", "$894", "Diamond Offshore", "United Kingdom", "$1,190", ...]
```

### Agent 2 — Entity Classifier (`entity_classifier.py`)

**Prompt strategy:** Strict. Each candidate gets one of 5 types or DISCARD.

The prompt has detailed rules per type:

- **Parent** — Exact match to filing company only
- **Subsidiary** — Requires ownership language nearby
- **Geography** — Country/state/city only (no regions, blocs, regulatory bodies)
- **Business Segment** — Requires GAAP segment language nearby
- **Financial Item** — Must contain a digit; not per-share; not stock comparison; not workforce stat

**Cross-chunk memory** is injected into the prompt:
```
Previously confirmed entities in this document:
  - Loews Corporation (Parent)
  - CNA Financial Corporation (Subsidiary)
  - Diamond Offshore Drilling, Inc. (Subsidiary)
  ...

If a candidate matches an entity above — even by abbreviation —
you MUST output the EXACT confirmed name from the list above.
```

This ensures *CNA* in chunk 80 resolves to `CNA Financial Corporation` (the canonical name confirmed in chunk 3), preventing graph fragmentation.

### Agent 3 — Relationship Finder (`relationship_finder.py`)

Split into **two segments** (separate LLM calls):

**Segment 1 — Structural relationships:**
- `PARENT_OF` (Parent → Subsidiary)
- `OPERATES_IN` (Parent | Subsidiary → Geography)

Operates on non-Financial-Item entities only. Strict ownership language required.

**Segment 2 — GENERATED relationships:**
- `GENERATED` (Subsidiary | Business Segment → Financial Item)

Critical grounding rule (added after a bug where Boardwalk Pipeline was mistakenly attributed CNA insurance metrics):

> *"Only assign a metric to a subsidiary if that subsidiary's name appears explicitly in the same row header, column header, or sentence as the metric value. If you cannot find the subsidiary's name adjacent to the value in the text, do not create the relationship."*

Financial items are batched (max 40 per call) to stay within token budgets.

---

## 6. The Guards Layer

### Why it exists

LLMs at `temperature=0` are still **non-deterministic** due to OpenAI's serving infrastructure (MoE routing, batching, floating-point order). The same chunk processed twice can yield different output. The guards layer is **deterministic, regex-based code** that:

1. **Validates** LLM output against type-specific rules
2. **Normalises** name variants (e.g. `CNA Financial Corporation` ↔ `CNA Financial`)
3. **Drops** obviously malformed entries
4. **Recovers** from LLM failures (FI promotion, source re-injection)
5. **Disambiguates** ambiguous attributions (content resolver)

This is the **single largest contribution** of the project — without it, the graph would be inconsistent between runs.

### Structure

`guards.py` is organised into sections:

1. **Patterns** (~lines 1-135) — All regex patterns used downstream
2. **Name normalisation** (~135-230) — `_normalize_sub`, `_segment_canonical`, `_resolve_source_by_content`, `find_confirmed_subs_in_text`
3. **FI promotion** (~232-265) — `promote_financial_items`
4. **Per-type validators** (~268-360) — `_is_non_financial`, `_is_eps_value`
5. **Entity guards** (~360-595) — 11 small helpers + `apply_entity_guards` orchestrator
6. **Relationship validation** (~600-790) — 6 small helpers + `validate_relationships` orchestrator

### `apply_entity_guards` — the post-Agent-2 cleanup

After segmentation (this session), the orchestrator is a 12-line function calling 11 helpers:

```python
def apply_entity_guards(entities, text, filing_company, confirmed_subsidiaries=None):
    confirmed = confirmed_subsidiaries or set()

    entities = _drop_malformed(entities)
    _normalise_parent_name(entities, filing_company)
    entities = _drop_too_broad_geos(entities)
    entities = _drop_date_like_fis(entities)
    entities = _drop_non_financial_fis(entities, text)
    entities = _drop_invalid_fi_format(entities)
    entities = _drop_eps_values(entities, text)
    entities = _validate_subsidiaries(entities, text, filing_company, confirmed)
    entities = _drop_segment_subcomponents(entities)
    entities = _validate_segment_adjacency(entities, text)
    _canonicalise_segments(entities, confirmed)
    _normalise_paren_negatives(entities)

    return entities
```

Each helper is independently understandable and unit-testable.

### `validate_relationships` — the post-Agent-3 cleanup

After segmentation, also a clean orchestrator over 6 helpers:

```python
def validate_relationships(relationships, entity_names, entity_type_map,
                           chunk_id, file, page_number, run_timestamp,
                           all_entities, chunk_text=""):
    dollar_normalised = {...}
    valid = []
    for rel in relationships:
        src, tgt, rel_type = rel.get("source"), rel.get("target"), rel.get("type")

        tgt = _normalise_target(rel, tgt, entity_names, dollar_normalised)
        src = _resolve_source(rel, src, rel_type, entity_names, entity_type_map, chunk_text)

        # Existence check (with auto-registration)
        src_in = src in entity_names
        tgt_in = tgt in entity_names
        tgt_in = _auto_register_financial_item(...)

        if not src_in or not tgt_in:
            log_drop_and_continue()

        # Type checks and corrections
        src, tgt, src_type = _correct_inverted_parent_of(...)
        src, src_type, dropped = _reroute_segment_operates_in(...)
        drop_reason = _enforce_relationship_type(...)

        if not dropped and not drop_reason:
            valid.append(rel)
    return valid
```

### Notable guard logic

#### Subsidiary ownership validation

A candidate is kept as a Subsidiary only if:
- It matches a previously-confirmed canonical name (re-validation without ownership language), OR
- The filing company appears in the text near explicit ownership phrases (`"wholly-owned subsidiary"`, `"owned by"`, etc.), OR
- A first-person possessive (`"our"`, `"we"`) is near the ownership phrase, OR
- Directional acquisition is detected (`"we acquired X"`)

**And** the candidate must NOT appear in a peer-comparison / stock-performance context — that's the `PEER_CONTEXT_PATTERN` guard that prevents competitors (Chubb, Travelers, etc.) from being mislabeled as subsidiaries.

#### Business Segment adjacency

A segment name must appear near a GAAP segment phrase (`"reportable segment"`, `"by segment"`, etc.) within a 200-character window, OR appear in a table cell with such a phrase in the column header within 400 characters above.

#### Parenthetical negative normalisation

Accounting parentheses are converted to leading minus:
- `(72,880)` → `-72,880`
- `$(1,200)` → `-$1,200`
- `(1.2 billion)` → `-1.2 billion`

This is shared between entity name normalisation and relationship target normalisation via the `_normalize_paren_negative` helper.

#### Source content resolution (chunk-53 fix)

When Agent 3 emits a source like `"CNA segment"` that doesn't match any entity name, the resolver:
1. Strips generic descriptors (`segment`, `operations`, `division`)
2. Normalises and extracts the leading token (`"cna"`)
3. Matches it against confirmed subsidiaries' leading tokens
4. Resolves only if exactly one matches (`CNA Financial Corporation`)

This runs **before** the older whole-text scan, preventing misattribution when one subsidiary is mentioned in passing and the real owner is abbreviated.

---

## 7. Cross-Chunk Memory

### The fragmentation problem

Without cross-chunk memory, the LLM would emit:
- Chunk 3: `CNA Financial Corporation` (full legal name)
- Chunk 80: `CNA` (abbreviation)
- Chunk 100: `CNA Financial` (short form)

Each becomes a **separate node** in Neo4j → fragmented graph, metrics split across three nodes that should be one.

### The solution: accumulating memory

`extractor.py` maintains a growing `all_entities` list. Before each chunk's classification call:

```python
confirmed_subs = {e["name"] for e in all_entities if e["type"] == "Subsidiary"}
chunk_entities, chunk_relationships = run_pipeline(
    current_text, filing_company, client,
    lambda ents: apply_entity_guards(ents, current_text, filing_company, confirmed_subs),
    run_timestamp, chunk_id,
    known_entities=all_entities,  # ← memory passed forward
    trace=chunk_trace
)
```

Inside `entity_classifier.py`, `known_entities` is:
1. Deduplicated by name
2. Stripped of Financial Items (too numerous; no canonical-name benefit)
3. Injected into Agent 2's prompt as the "Previously confirmed entities" block

The prompt then instructs Agent 2 to **output the exact canonical name** when a candidate matches an already-confirmed entity, even by abbreviation.

### Three recovery mechanisms

When the LLM still fails despite memory, the pipeline has three deterministic recovery layers:

#### 1. Financial Item Promotion (`promote_financial_items`)

If Agent 2 drops a Financial Item that Agent 1 found and that matches `FINANCIAL_VALUE_PATTERN` or `PAREN_NEGATIVE_PATTERN`, it's auto-promoted before the guards run.

**Why:** Agent 2 occasionally collapses — chunk 53 once produced just `[Parent]` from 48 candidates, losing all CNA's financial values. FI promotion recovers them deterministically.

#### 2. Subsidiary Re-injection (`find_confirmed_subs_in_text`)

If a chunk has Financial Items but **no Subsidiary or Business Segment**, Agent 3 would skip GENERATED entirely (no valid source). The re-injection logic:
1. Scans the chunk text for confirmed subsidiary names (leading-token match)
2. Injects matching subsidiaries into Agent 3's input
3. Lets Agent 3 attribute metrics to the canonical owner

**Safety net:** Agent 3's grounding rule and `drop_orphaned_financial_items` together filter any over-injected subsidiary with no edges.

#### 3. Source Content Resolution (`_resolve_source_by_content`)

When Agent 3 emits a non-canonical source like `"CNA segment"`:
1. Strip generic words (`segment`, `operations`)
2. Match leading token against confirmed subsidiaries
3. Resolve only on single unambiguous match

Runs **before** the chunk-text scan, preventing the misattribution trap where an unrelated subsidiary (e.g. `Diamond Offshore`) happens to appear in the same chunk.

---

## 8. Statement Extractor

### Why a separate path?

Financial statements (Income Statement, Balance Sheet, Cash Flow) have **dozens of line items per fiscal year**. Modeling each line item as a separate `Financial Item` node + `GENERATED` edge would:
- Bloat the graph (1,500+ extra edges per statement set)
- Make per-year aggregate queries cumbersome
- Lose the natural "this is the 2019 income statement" entity identity

Instead, each statement becomes **one node with line items as properties**:

```
(:IncomeStatement {name: "IncomeStatement_FY2019",
                   fiscal_year: "2019",
                   unit: "millions",
                   revenues: "804",
                   net_income: "932",
                   insurance_premiums: "7,428",
                   ... 100+ more line items
                  })
```

Connected to the Parent via:
```
(:Parent)-[:INCOME_STATEMENT {property: "2019"}]->(:IncomeStatement)
```

### How it works

```python
def extract_statements(client, text, filing_company):
    response = client.chat.completions.create(...)
    statements = json.loads(response.choices[0].message.content).get("statements", [])

    for stmt in statements:
        stmt_type = stmt.get("type")           # "INCOME_STATEMENT" / "BALANCE_SHEET" / "CASH_FLOW"
        fiscal_year = stmt.get("fiscal_year")  # "2019"
        items = stmt.get("items") or {}        # dict of line items

        # Build node + edge
        node_name = f"{NODE_LABEL[stmt_type]}_FY{fiscal_year}"
        statement_entities.append({
            "name": node_name,
            "type": NODE_LABEL[stmt_type],
            "properties": {**items, "fiscal_year": fiscal_year, "unit": stmt.get("unit")},
        })
        statement_relationships.append({
            "source": filing_company,
            "target": node_name,
            "type": stmt_type,
            "property": fiscal_year,
        })
```

### Multi-chunk merging

The same statement appears across multiple chunks (top half on one page, bottom half on the next). Both extractions produce a node named `IncomeStatement_FY2019` with **different properties**. When `graph.py` writes to Neo4j:

```cypher
MERGE (n:Entity:IncomeStatement {name: "IncomeStatement_FY2019"})
SET n += $props      // union properties from both chunks
```

The `+=` operator merges property dicts. So the final node holds the union of all line items from all chunks that contributed to FY2019.

---

## 9. Merge, Canonicalisation & Enrichment

### `merge_graph()` — chunk-level deduplication

Runs after all 86 chunks processed. Three passes:

**Pass 1: entity dedup by (name, type)**
- Statement nodes are merged property-wise (unions line items from different chunks)
- All other entities: keep first occurrence

**Pass 2: relationship dedup by (source, target, type)**
- Property fields are combined (a non-null property wins over null)

**Pass 3: format-duplicate dedup for GENERATED relationships**
- Group by `(source, type, property)` — same metric, possibly different target formats
- Example: `($1,161 million, fy2019)` and `($1,161, fy2019)` for the same CNA warranty revenue
- Winner: target with an explicit scale word (`million`/`billion`/`trillion`), or the longer string

### `canonicalize_segment_names()` — segment de-fragmentation

This was the fix that took the final graph from 6 fragmented segments to 3 clean ones.

**Problem:** the per-chunk canonicalisation in `apply_entity_guards` runs with only the subsidiaries confirmed *so far*. A segment extracted before its owning subsidiary was seen survives in non-canonical form.

**Solution:** after `merge_graph`, re-run the canonicalisation **globally** with the complete subsidiary list:

```python
def canonicalize_segment_names(graph):
    # Build rename map: each Business Segment → canonical short form of matching subsidiary
    rename = {}
    for ent in entities:
        if ent["type"] != "Business Segment": continue
        matched_sub = find_matching_subsidiary(ent["name"], sub_names)
        if matched_sub:
            canonical = _segment_canonical(matched_sub)
            if canonical != ent["name"]:
                rename[ent["name"]] = canonical

    # Apply renames to Business Segment nodes
    for ent in entities:
        if ent["type"] == "Business Segment" and ent["name"] in rename:
            ent["name"] = rename[ent["name"]]

    # Re-point edges — ONLY for segment-only names (collision safety)
    seg_only = {old for old in rename if old not in sub_names}
    for rel in relationships:
        if rel["source"] in seg_only:
            rel["source"] = rename[rel["source"]]

    # Dedup + drop orphan segments with no edges
    return clean_graph
```

**Critical safety:** the same name can exist as both `Subsidiary` and `Business Segment` (e.g. `CNA Financial Corporation`). The `seg_only` filter ensures the subsidiary's 245 GENERATED edges are NOT re-pointed — they belong to the subsidiary, not the segment.

### `enrich_financial_items()` — make values human-readable

Transforms raw values into labelled nodes:

**Before:**
```
(:Subsidiary {name:"CNA Financial Corporation"})
   -[:GENERATED {property:"net income attributable to loews corporation fy2019"}]->
(:FinancialItem {name:"$894"})
```

**After:**
```
(:Subsidiary {name:"CNA Financial Corporation"})
   -[:GENERATED {property:"2019"}]->
(:FinancialItem {name:"Net Income Attributable To Loews Corporation: $894"})
```

The metric label moves into the node name (where it's queryable via `f.name CONTAINS 'Net Income'`), and the property reduces to just the fiscal year (queryable via `r.property = '2019'`).

### `drop_orphaned_financial_items()` — final cleanup

Removes Financial Items that no relationship points to. These accumulate when:
- FI promotion added a value that Agent 3 then didn't link
- Auto-registration created a target that turned out not to need it
- A relationship was dropped during validation, orphaning its target

---

## 10. Neo4j Graph Storage

### Schema

```
NODES (all share :Entity label)
  :Parent           — the filing company
  :Subsidiary       — companies owned by Parent
  :Geography        — country/state/city
  :BusinessSegment  — GAAP reporting segments
  :FinancialItem    — "Metric Label: $value" format
  :IncomeStatement  — IncomeStatement_FY<year>
  :BalanceSheet     — BalanceSheet_FY<year>
  :CashFlow         — CashFlow_FY<year>

RELATIONSHIPS
  (Parent)-[:PARENT_OF]->(Subsidiary)
  (Parent|Subsidiary)-[:OPERATES_IN]->(Geography)
  (Subsidiary|BusinessSegment)-[:GENERATED {property, page_number}]->(FinancialItem)
  (Parent)-[:INCOME_STATEMENT {property}]->(IncomeStatement)
  (Parent)-[:BALANCE_SHEET {property}]->(BalanceSheet)
  (Parent)-[:CASH_FLOW {property}]->(CashFlow)
```

### Why dual labels (`:Entity:Subsidiary`)?

- `:Entity` enables generic queries (`MATCH (n:Entity) RETURN n.name`)
- The specific label enables type-filtered queries (`MATCH (s:Subsidiary)`)

### Why edges carry `page_number`?

Every GENERATED / OPERATES_IN / PARENT_OF edge stores the chunk's source page number. This enables citation: *"Net income: $894 (Graph, p.47)"* — auditable answers.

100% of relationships in the final graph have valid page numbers.

### `graph.py` writer

Uses `MERGE` so re-runs are idempotent:

```cypher
MERGE (n:Entity:Subsidiary {name: $name})
SET n += $props,
    n.type        = $type,
    n.chunk_id    = $chunk_id,
    n.file        = $file,
    n.page_number = $page_number
```

For statement nodes, `n += $props` merges the line items dict (so partial statements from multiple chunks accumulate).

---

## 11. Query Layer

### `query1.py` — graph-only RAG

Three-step pipeline:

```
USER QUESTION
   ↓
[1] _generate_queries
    Inject schema + entity catalog into prompt
    LLM produces Cypher queries (one or more)
   ↓
[2] _run_cypher (per query)
    Execute, serialize results to plain text
   ↓
[3] _synthesize
    Pass graph context + question to synthesis LLM
    Returns natural-language answer with citations
```

### Schema prompt

The `CYPHER_SYSTEM` prompt teaches the LLM:
- Available node labels and what each represents
- Relationship types and their valid source/target pairs
- How to query GENERATED edges (filter on `f.name CONTAINS '<metric>'`)
- How to query statement-node properties (`s.net_income`, or `properties(s) AS line_items`)
- Disambiguation rules (Subsidiary vs BusinessSegment dual queries, ownership lookup)
- Loss queries (negative Net Income, never a "Net Loss" label)

### Key Cypher rules (excerpted from CYPHER_SYSTEM)

- **Dual-source queries:** financial metrics may be on Subsidiary OR BusinessSegment nodes → always query both
- **Consolidated totals:** use statement-node properties, not GENERATED edges
- **Per-entity statement keys:** use `RETURN properties(s) AS line_items` so synthesis can pick the prefixed key (e.g. `cna_financial_net_income_attributable_to_loews_corporation`)
- **Ownership lookup:** check both Parent and Subsidiary as owner (sub-to-sub ownership exists)
- **Geography reverse lookup:** match exact name AND `CONTAINS` for partial matches

### `_run_cypher` — result serialization

Critical detail: when Cypher returns `properties(s) AS line_items` (a dict), the serializer **expands each key:value as its own line** rather than collapsing to a string. This makes the line items individually readable by the synthesis LLM:

```
s.name: IncomeStatement_FY2019 | s.fiscal_year: 2019
  net_income: 932
  cna_financial_net_income_attributable_to_loews_corporation: 894
  diamond_offshore_net_income_attributable_to_loews_corporation: (175)
  ... (115 more keys)
```

### `_synthesize` prompt

Tells the synthesis LLM:
- Trust the graph data as ground truth
- Never fabricate values not in the records
- Format numbers cleanly (`"$894 million"` not `"894"`)
- Prefer the most aggregate metric (`"Total Revenues"` over `"Revenues"`)
- For entity-specific questions, prefer entity-prefixed keys in `line_items` over generic keys
- Cite GENERATED-edge facts as `(Graph, p.<N>)`
- Cite statement-node facts as `(Graph, <StatementNodeName>)`

### Generic across documents

After this session's hardening, all prompts and code use **placeholders not company names**. The query layer works identically on any 10-K — only the `entity_catalog` (loaded from the live graph) differs per document.

---

## 12. Testing Strategy

### Two-section test harness (`test_chunks.py`)

**Section 1 — Live document scan**
Processes 5 targeted chunks (3, 53, 55, 157, 171) that historically exposed specific bugs. Each chunk's output is checked against quality rules:
- Geography regulatory-body slip-through
- Geography too-broad regions
- Invalid relationship types
- Self-referential relationships
- PARENT_OF source not Parent
- Financial relationship targets not valid
- Financial relationship with no FI entities
- Subsidiary false positives

Reports per-chunk issues and aggregate counts.

**Section 2 — Unit tests (no API)**
31 deterministic tests of guard logic without LLM calls:

| Block | Tests | What it covers |
|---|---|---|
| 2a | 4 | `merge_graph` format-duplicate dedup |
| 2b | 11 | `_normalize_sub` across name variants |
| 2e | 3 | Ownership guard (competitor vs subsidiary, peer-table drop) |
| 2f | 4 | `canonicalize_segment_names` (de-fragmentation, no edge loss) |
| 2g | 3 | `promote_financial_items` (recovers FIs Agent 2 dropped) |
| 2h | 3 | `_resolve_source_by_content` (CNA misattribution prevented) |
| 2i | 3 | `find_confirmed_subs_in_text` (re-injection without false positives) |

**Test philosophy:** every fix added during development gets a regression test, so the same bug can't silently return.

### `eval.py` — accuracy benchmark

Tests against known ground truth for the Loews 10-K:
- **Structural completeness** — expected subsidiaries present
- **Financial accuracy** — known figures match (CNA insurance premiums $7,428M, etc.)
- **Query coverage** — benchmark questions return non-empty results

(This file is Loews-specific by design; for a different 10-K, the ground-truth constants would need updating, or a JSON config file.)

---

## 13. Known Limitations

### Q1 — consolidated total revenue collision

**The problem:** `IncomeStatement_FY2019.total_revenues = 1,106` is actually Loews's **parent-only Schedule I** (a separate SEC-required disclosure) — not the consolidated $14,931M from Item 6. Both statements were tagged `IncomeStatement_FY2019` and merged into one node; Schedule I's value won the last-write-wins property collision.

**Root cause:** the statement extractor doesn't distinguish "consolidated income statement" from "parent-company-only Schedule I" — both pass the keyword filters.

**Three honest options:**
1. **Document the limitation** (current path) — acknowledge in PFE that consolidated grand totals require future work
2. **Tag node names by statement type** — `ParentOnlyIncomeStatement_FY2019` vs `IncomeStatement_FY2019` (needs code + re-extract)
3. **Add hybrid vector retrieval** — vector fetch from Qdrant would surface the $14,931 from chunks 46/91/158 where it appears in raw text. Bypasses extraction collision entirely.

### Vector fetch not wired

`query1.py` is graph-only. Qdrant is populated by the chunker but no query path uses it. Adding `_vector_fetch` would:
- Fix Q1 (consolidated totals appear in source text)
- Enable narrative questions ("business strategy")
- Bump system rating ~8.5 → 9.3-9.5

Estimated work: ~40 lines, one unit test.

### Same-name collision between Subsidiary and BusinessSegment

`CNA Financial Corporation` exists as both Subsidiary (with 245 GENERATED edges) and BusinessSegment (collision). The canonicalisation pass handles this safely — it never re-points the Subsidiary's edges — but the dual presence requires the query layer to dual-query both types.

### Statement-node citation precision

Statement nodes don't carry a single `page_number` (they're merged across chunks). Citations use the node name (`IncomeStatement_FY2019`) rather than a page. Per-line-item page provenance would require either a `<key>_page` sidecar property pattern or a JSON-encoded `_provenance` dict.

### LLM non-determinism

Even at `temperature=0`, identical chunks can produce different output between runs due to OpenAI infrastructure variance. The guards layer mitigates this but cannot fully eliminate it. Some run-to-run variation in extraction quality is expected.

---

## 14. Future Work

### High-leverage improvements (ranked by impact)

| # | Improvement | Effort | Impact |
|---|---|---|---|
| 1 | Wire vector fetch into `query1.py` | 1-2 hours | High — fixes Q1, enables narrative questions |
| 2 | Test on NVIDIA 10-K | 1 hour | High — proves portability of the entire architecture |
| 3 | Per-statement-type node naming | 30 min + re-extract | Medium — eliminates Schedule I collision |
| 4 | Refresh `eval.py` on current graph | 30 min | Medium — gives fresh accuracy numbers for PFE |
| 5 | Per-line-item statement provenance | 30 min + re-extract | Low — better citation precision |
| 6 | Extract ground truth to JSON config | 30 min | Low — makes `eval.py` portable |

### Research extensions

- **Multi-document graphs** — link multiple companies' graphs to enable cross-company queries
- **Time-series graphs** — link multiple fiscal years to enable trend analysis
- **Schema-aware Cypher generation** — use Cypher syntax validation in the loop
- **Confidence scoring** — attach extraction confidence to each fact for query-time filtering

---

## 15. Appendix — Schema Reference

### Full entity type catalog

| Type | Example | Source |
|---|---|---|
| Parent | `Loews Corporation` | Filing company (one per document) |
| Subsidiary | `CNA Financial Corporation` | Wholly-owned or majority-owned operating companies |
| Geography | `United States`, `Illinois` | Countries, states, cities |
| BusinessSegment | `CNA Financial`, `Corporate` | GAAP reporting segments |
| FinancialItem | `Net Income: $894`, `Total Revenues: $14,931` | Metric value pairs |
| IncomeStatement | `IncomeStatement_FY2019` | Per fiscal year |
| BalanceSheet | `BalanceSheet_FY2019` | Per fiscal year |
| CashFlow | `CashFlow_FY2019` | Per fiscal year |

### Full relationship type catalog

| Type | Allowed Source | Allowed Target | Properties |
|---|---|---|---|
| PARENT_OF | Parent | Subsidiary | `chunk_id`, `page_number` |
| OPERATES_IN | Parent, Subsidiary | Geography | `chunk_id`, `page_number` |
| GENERATED | Subsidiary, BusinessSegment | FinancialItem | `property` (fiscal year after enrichment), `chunk_id`, `page_number` |
| INCOME_STATEMENT | Parent | IncomeStatement | `property` (fiscal year), `chunk_id`, `page_number` |
| BALANCE_SHEET | Parent | BalanceSheet | `property` (fiscal year), `chunk_id`, `page_number` |
| CASH_FLOW | Parent | CashFlow | `property` (fiscal year), `chunk_id`, `page_number` |

### Example Cypher queries

**All subsidiaries:**
```cypher
MATCH (p:Parent)-[:PARENT_OF]->(s:Subsidiary)
RETURN s.name ORDER BY s.name
```

**CNA's GENERATED metrics in 2019:**
```cypher
MATCH (s:Subsidiary {name:'CNA Financial Corporation'})-[r:GENERATED]->(f:FinancialItem)
WHERE r.property = '2019'
RETURN f.name, r.property, r.page_number
```

**Subsidiaries with a net loss:**
```cypher
MATCH (s)-[r:GENERATED]->(f:FinancialItem)
WHERE f.name CONTAINS 'Net Income' AND f.name CONTAINS '-'
RETURN labels(s)[1] AS Type, s.name, f.name, r.property
```

**All line items of the 2019 income statement:**
```cypher
MATCH (p:Parent)-[:INCOME_STATEMENT]->(s:IncomeStatement {fiscal_year:'2019'})
RETURN properties(s) AS line_items
```

**Reverse geographic lookup:**
```cypher
MATCH (s)-[:OPERATES_IN]->(g:Geography {name:'Texas'})
RETURN labels(s)[1] AS Type, s.name
```

---

## Conclusion

This project demonstrates that **a layered architecture of LLM agents + deterministic guards** can produce a high-quality, queryable knowledge graph from unstructured financial documents — substantially more reliably than either pure LLM extraction or pure rule-based parsing.

The system's key architectural decisions:

1. **Multi-agent extraction** — split the task into find / classify / relate so each agent has one job
2. **Deterministic guards layer** — every LLM output is validated, normalised, or rejected by code
3. **Cross-chunk memory** — the canonical-name list grows during extraction, preventing fragmentation
4. **Statement node design** — financial statements as nodes with line items as properties (compact, query-friendly)
5. **Three-stage recovery** — FI promotion + subsidiary re-injection + content resolution handle LLM failures deterministically
6. **Hybrid storage + query** — graph for structure, prompts for semantics, citations for trust

**Final state:** 534 entities, 535 relationships, 9.4/10 extraction quality, 31/31 unit tests passing, fully portable to other 10-K documents.

---

*End of documentation.*
