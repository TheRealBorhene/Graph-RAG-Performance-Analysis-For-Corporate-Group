from openai import OpenAI
import json
import time
from logger import warn, info, log_error

MODEL                = "gpt-4.1-mini"
SLEEP_BETWEEN_AGENTS = 0.3
MAX_FI_PER_AGENT3    = 40


def build_structural_prompt(filing_company: str) -> str:
    """Segment 1 — structural/ownership relationships (no Financial Items involved)."""
    return f"""You are finding STRUCTURAL relationships between validated entities in a financial document.
The filing company is: {filing_company}

You will receive a list of validated entities (Parent, Subsidiary, Geography, Business Segment)
and the original text. There are NO Financial Items in this task.

CRITICAL RULES:
1. source and target MUST be taken verbatim from the entity list. Never invent a name.
2. Only extract what is explicitly stated in the text — never infer.
3. If no valid relationship can be formed, return an empty list.

════════════════════════════════════════════
RELATIONSHIP TYPES
════════════════════════════════════════════

PARENT_OF — Parent legally owns or controls a Subsidiary
  source: Parent  |  target: Subsidiary
  Only with explicit ownership language: "owns", "wholly-owned subsidiary of",
  "controlled by", "acquired". Do not extract for failed or terminated acquisitions.

OPERATES_IN — Legal entity has physical presence in a Geography
  source: Parent or Subsidiary  |  target: Geography
  Only extract when the text contains explicit physical presence language — headquarters,
  offices, facilities, or place of incorporation — tied to that location.
  Do NOT extract for: revenue breakdowns, customer locations, workforce statistics,
  export controls, trade restrictions, sanctions, or legal jurisdiction references.

════════════════════════════════════════════
RULES
════════════════════════════════════════════
- Only extract what is explicitly stated — never infer
- When in doubt, do NOT extract

Return ONLY valid JSON:
{{"relationships": [{{"source": "name", "target": "name", "type": "TYPE", "property": null}}]}}
If none found, return: {{"relationships": []}}"""


def build_governance_prompt(filing_company: str) -> str:
    """Segment 3 — board / governance relationships (BOARD_MEMBER_OF only)."""
    return f"""You are finding GOVERNANCE relationships between validated Person entities
and the companies they govern in a financial document.
The filing company is: {filing_company}

You will receive a list of validated entities (Persons, Parent, Subsidiaries) and the
original text.

CRITICAL RULES:
1. source and target MUST be taken verbatim from the entity list. Never invent a name.
2. Only extract what is explicitly stated in the text — never infer.
3. If no valid relationship can be formed, return an empty list.

════════════════════════════════════════════
RELATIONSHIP TYPE
════════════════════════════════════════════

BOARD_MEMBER_OF — A Person serves on the board of (or as a senior governance officer of)
  a Parent or Subsidiary entity
    source: Person  |  target: Parent or Subsidiary
    Only extract when the text states the person holds a board or senior governance
    role at the target company. Trigger on roles containing any of:
      "Director", "Chairman", "Vice Chairman", "Board of Directors",
      "Chief Executive Officer", "Chief Financial Officer", "President",
      "Senior Vice President", "Vice President", "General Counsel", "Secretary",
      "Treasurer", "Chief Investment Officer", "Chief Operating Officer",
      "Office of the President"

    Store the role string verbatim in `property`. Examples (generic):
      "Co-Chairman of the Board"
      "President and Chief Executive Officer"
      "Senior Vice President and Chief Financial Officer"

    Do NOT extract for:
      - People mentioned only as employees, advisors, or consultants
      - People described in third-party contexts (litigation, citations)
      - Companies where the person is NOT serving in a governance capacity

════════════════════════════════════════════
RULES
════════════════════════════════════════════
- Only extract what is explicitly stated — never infer
- The property field holds the role as it appears in the document, verbatim

Return ONLY valid JSON:
{{"relationships": [{{"source": "person name", "target": "company name", "type": "BOARD_MEMBER_OF", "property": "role string"}}]}}
If none found, return: {{"relationships": []}}"""


def build_generated_prompt(filing_company: str) -> str:
    """Segment 2 — segment and subsidiary revenue/income (GENERATED only)."""
    return f"""You are finding GENERATED relationships between validated entities in a financial document.
The filing company is: {filing_company}

You will receive a list of validated entities (including Business Segments, Subsidiaries, and Financial Items)
and the original text.

CRITICAL RULES:
1. source and target MUST be taken verbatim from the entity list. Never invent a name.
2. The target must be a standalone numeric value containing a digit, taken exactly as it
   appears verbatim in the text. NEVER combine a row label with a value —
   "CNA Revenue $9,800" is WRONG; use "$9,800" as target and "cna revenue fy2024" as property.
3. Only use values that appear as standalone numbers in the text — never invent or reformat.
4. If no valid relationship can be formed, return an empty list.
5. {filing_company} is the Parent (filing company) — it CANNOT be the source of a GENERATED
   relationship. Only Subsidiaries and Business Segments generate revenue figures.

════════════════════════════════════════════
RELATIONSHIP TYPE
════════════════════════════════════════════

GENERATED — A segment or subsidiary produced a revenue or income figure
  source: Business Segment or Subsidiary  |  target: Financial Item
  Only for segment- or subsidiary-level revenue, sales, income, or margin figures.
  Do not extract expenses, costs, or change percentages.
  Do not assign company-level consolidated totals to a segment or subsidiary — the target must
  come from a segment- or subsidiary-specific row, not a total or sum row of the table.
  Only assign a metric to a subsidiary if that subsidiary's name appears explicitly in the same
  row header, column header, or sentence as the metric value. If you cannot find the subsidiary's
  name adjacent to the value in the text, do not create the relationship.
  Store metric name and period in property: e.g. "cna revenue fy2024".

════════════════════════════════════════════
RULES
════════════════════════════════════════════
- Only extract what is explicitly stated — never infer
- Write property values in lowercase with fiscal period at the end:
  e.g. "xyz financial revenue fy2024", "abc hotels income fy2023"

Return ONLY valid JSON:
{{"relationships": [{{"source": "name", "target": "name", "type": "GENERATED", "property": "value or null"}}]}}
If none found, return: {{"relationships": []}}"""


def find_relationships(client: OpenAI, text: str, filing_company: str,
                       entities: list[dict], run_timestamp: str, chunk_id: int) -> list[dict]:
    non_fi  = [e for e in entities if e["type"] != "Financial Item"]
    fi_only = [e for e in entities if e["type"] == "Financial Item"]

    def _call_segment(prompt: str, entity_batch: list[dict]) -> list[dict]:
        if not entity_batch:
            return []
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": (
                    f"Validated entities:\n{json.dumps(entity_batch)}\n\n"
                    f"Original text:\n{text}"
                )}
            ],
            temperature=0,
            max_tokens=8192,
            response_format={"type": "json_object"}
        )
        try:
            rels = json.loads(response.choices[0].message.content).get("relationships", [])
        except json.JSONDecodeError:
            return []
        return [r for r in rels if r.get("source") != r.get("target")]

    def _call_with_fi_batching(prompt: str, segment_label: str) -> list[dict]:
        results = []
        if len(fi_only) <= MAX_FI_PER_AGENT3:
            try:
                results = _call_segment(prompt, non_fi + fi_only)
            except Exception as err:
                warn(f"Agent 3 [{segment_label}] failed: {err}")
                log_error(run_timestamp, chunk_id, f"Agent 3 [{segment_label}] error: {err}")
        else:
            batch_size  = max(MAX_FI_PER_AGENT3 - len(non_fi), 1)
            num_batches = (len(fi_only) + batch_size - 1) // batch_size
            info(f"Batching Agent 3 [{segment_label}]: {len(fi_only)} FIs -> {num_batches} call(s)")
            for b_idx in range(0, len(fi_only), batch_size):
                try:
                    batch_rels = _call_segment(prompt, non_fi + fi_only[b_idx:b_idx + batch_size])
                    results.extend(batch_rels)
                except Exception as err:
                    warn(f"Agent 3 [{segment_label}] batch failed  -  skipping batch: {err}")
                    log_error(run_timestamp, chunk_id, f"Agent 3 [{segment_label}] batch error: {err}")
                time.sleep(SLEEP_BETWEEN_AGENTS)
        return results

    relationships = []

    # ── Segment 1: structural (PARENT_OF, OPERATES_IN) ───────────────────────
    has_structural_targets = any(
        e["type"] in ("Subsidiary", "Geography", "Business Segment")
        for e in non_fi
    )
    if has_structural_targets:
        try:
            seg1 = _call_segment(build_structural_prompt(filing_company), non_fi)
            info(f"Agent 3 [structural]  : {len(seg1)} relationship(s)")
            relationships.extend(seg1)
        except Exception as err:
            warn(f"Agent 3 [structural] failed: {err}")
            log_error(run_timestamp, chunk_id, f"Agent 3 [structural] error: {err}")
        time.sleep(SLEEP_BETWEEN_AGENTS)
    else:
        info("Agent 3 [structural]  : skipped (no Subsidiary / Geography / Business Segment)")

    # ── Segment 2: GENERATED (segment/subsidiary revenue/income) ─────────────
    has_generated_sources = any(
        e["type"] in ("Business Segment", "Subsidiary")
        for e in non_fi
    )
    if has_generated_sources and fi_only:
        seg2 = _call_with_fi_batching(build_generated_prompt(filing_company), "generated")
        info(f"Agent 3 [generated]   : {len(seg2)} relationship(s)")
        relationships.extend(seg2)
    else:
        info("Agent 3 [generated]   : skipped (no segment/subsidiary or no Financial Items)")

    # ── Segment 3: BOARD_MEMBER_OF (governance) ──────────────────────────────
    # Only runs when the chunk has at least one Person entity AND a company entity
    # to attach them to — avoids wasted API calls on chunks with no governance data.
    has_persons   = any(e["type"] == "Person" for e in non_fi)
    has_companies = any(e["type"] in ("Parent", "Subsidiary") for e in non_fi)
    if has_persons and has_companies:
        try:
            seg3 = _call_segment(build_governance_prompt(filing_company), non_fi)
            info(f"Agent 3 [governance]  : {len(seg3)} relationship(s)")
            relationships.extend(seg3)
        except Exception as err:
            warn(f"Agent 3 [governance] failed: {err}")
            log_error(run_timestamp, chunk_id, f"Agent 3 [governance] error: {err}")
        time.sleep(SLEEP_BETWEEN_AGENTS)
    else:
        info("Agent 3 [governance]  : skipped (no Person or no Parent/Subsidiary)")

    # ── Deduplicate ───────────────────────────────────────────────────────────
    seen_rels: set[tuple] = set()
    deduped: list[dict]   = []
    for r in relationships:
        key = (r.get("source"), r.get("target"), r.get("type"), r.get("property"))
        if key not in seen_rels:
            seen_rels.add(key)
            deduped.append(r)
    if len(deduped) < len(relationships):
        info(f"Deduped {len(relationships) - len(deduped)} duplicate relationship(s)")

    return deduped
