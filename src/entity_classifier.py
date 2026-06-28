from openai import OpenAI
import json

MODEL = "gpt-4.1-mini"


def build_classifier_prompt(filing_company: str, known_entities: list[dict] | None = None) -> str:
    known_lines = "\n".join(
        f"  - {e['name']} ({e['type']})"
        for e in (known_entities or [])
    ) or "  None yet."

    return f"""You are classifying entity candidates from a financial document.
The filing company is: {filing_company}
"we", "our", "the Company" refer to {filing_company}.

Previously confirmed entities in this document:
{known_lines}
If a candidate refers to the same entity as one above — even by abbreviation or short form —
you MUST output the EXACT confirmed name from the list above.
Never output a shortened form, partial name, or abbreviation if the full canonical name
is already confirmed. For example: if "XYZ Financial Corporation" is confirmed,
output "XYZ Financial Corporation" — never "XYZ" or "XYZ Financial".

Assign exactly one type below — or DISCARD. Omit discarded candidates entirely.

Parent:
  Only the exact legal name "{filing_company}" — character for character. At most one per document.
  If you are not certain the candidate is exactly "{filing_company}", discard it.
  Discard generic references ("the Company", "the Registrant") and any other company.

Subsidiary:
  A company currently owned or controlled by {filing_company}, with explicit ownership
  language nearby: "owns", "wholly-owned subsidiary of", "controlled by", "acquired".
  Discard suppliers, foundries, partners, terminated acquisition targets, and
  peer/competitor companies listed in stock performance comparison tables
  (e.g. "the following graph compares... against a peer group of: Chubb, Travelers...").

  IMPORTANT EXCEPTION — already-confirmed subsidiaries:
  If the candidate matches — even by short form, abbreviation, or operating name —
  an entity already listed under "Previously confirmed entities" above, you MUST
  classify it as Subsidiary using the EXACT confirmed name from that list.
  No ownership language is required for already-confirmed subsidiaries.
  Examples (generic — apply the same logic to any confirmed name):
    "ABC Hotels"     → confirmed "ABC Hotels Holding Corporation" → output "ABC Hotels Holding Corporation"
    "XYZ"            → confirmed "XYZ Financial Corporation"      → output "XYZ Financial Corporation"
    "DEF Offshore"   → confirmed "DEF Offshore Drilling, Inc."    → output "DEF Offshore Drilling, Inc."
    "GHI Pipelines"  → confirmed "GHI Pipeline Partners, LP"      → output "GHI Pipeline Partners, LP"

Geography:
  A specific country, state, province, or city — nothing broader.
  Valid: United States, France, Illinois, London, Brazil, Malaysia, Canada, Texas.
  Discard everything that is not a country/state/province/city:
  — regions, sub-regions, and bodies of water (Gulf Coast, Midwest, Northeast,
    South America, Southeast Asia, Gulf of Mexico, Mediterranean)
  — trade blocs and unions (European Union, E.U., EMEA, APAC)
  — regulatory bodies and institutions — any name containing the words Department,
    Authority, Commission, Association, Commissioners, Superintendent, Monetary,
    Institute, Board, Bureau, Agency, Committee, Council, Office of — even if it
    contains a geographic word. Extract "Bermuda" only if the text separately
    mentions Bermuda as a place of operations, not as part of an institution name.
  — currencies, addresses, postal codes, and generic terms (Other, Worldwide, Rest of World)
  — names with legal or governmental modifiers (law, regulation, court, jurisdiction)

Business Segment:
  A reporting unit that {filing_company} discloses as a segment in its financial statements.
  Trigger on ANY of these signals:
    - The word "segment" appears near the name in the text
      (e.g. "X segment", "our X segment", "X segment results", "segment revenues by X")
    - A table header contains "Segment", "Business Segment", or "Reportable Segment"
      and the name appears as a row label in that table
    - The text discusses "segment income", "segment loss", or "segment operating results"
      and names the reporting units

  In a conglomerate 10-K, a subsidiary and a business segment can refer to the same
  underlying business. A confirmed subsidiary (listed above) may also appear as a
  Business Segment when mentioned in a segment reporting context — classify it as
  Business Segment using its SHORT operational name (drop legal suffixes like
  "Corporation", "LLC", "LP", "Inc.", "Drilling", "Holding").
  Apply this rule generically to any confirmed subsidiary name — do not hardcode names.
  Example of the rule: "XYZ Financial Corporation" in segment context → "XYZ Financial"

  Discard: product names, end-market categories, and segments of OTHER companies.
  Output the segment name only — never append descriptive words like "segment", "revenue", "income".
  WRONG: "X segment", "X revenue", "X income"
  CORRECT: "X"  (the bare operational name only)

Person:
  A named individual disclosed as a board member, director, executive officer, or
  similar governance role for {filing_company} or one of its subsidiaries.
  Trigger when ALL of the following are true:
    - The candidate is a full personal name (first + last, e.g. "Jane R. Doe", "John Smith")
    - A role/title is stated in the same paragraph or adjacent table row that contains
      ANY of: "Director", "Chairman", "Vice Chairman", "Board", "Chief Executive Officer",
      "Chief Financial Officer", "President", "Senior Vice President", "Vice President",
      "General Counsel", "Secretary", "Treasurer", "Chief Investment Officer",
      "Chief Operating Officer", "Officer"
    - The role is held at {filing_company} or one of its subsidiaries (not at a peer,
      vendor, regulator, or external organisation)

  Output the EXACT personal name as it appears in the text — preserve middle initials,
  suffixes (Jr., Sr., III), and capitalization. Do NOT include the title in the name.
  Examples (generic — apply to any name):
    "Mr. John A. Smith"                        → "John A. Smith"
    "Jane R. Doe, our Chief Financial Officer" → "Jane R. Doe"
    "Robert K. Lee III, Director"              → "Robert K. Lee III"

  Discard:
    - References to people in third-party contexts (analysts, regulators, journalists,
      authors of cited works, plaintiffs, defendants)
    - Generic role mentions with no personal name attached
      ("our Chief Executive Officer" with no name → discard)
    - Historical references with no current role
      ("the late John Smith, who served as..." → discard)
    - Mentions in litigation, lawsuit, or court-filing contexts
    - People disclosed only as employees of competitors, peers, or unrelated organisations

Financial Item:
  A specific numeric value from a financial reporting context — must contain a digit.
  If the candidate contains no digit whatsoever, it is a label — discard it immediately.
  Examples of labels to discard: "Research and development", "Basic", "Diluted", "CCPA", "GDPR".
  Extract from financial statements, their accompanying notes, and business description
  sections where specific monetary values are stated — backlog figures, credit facility
  amounts, contract values, and debt amounts qualify even outside formal statements.
  Extract exactly as they appear (do not convert or reformat).
  If a table header says "(In millions)" and a cell shows "8,589", extract "8,589".

  Discard:
  - Values in rows labeled "per share", "Basic", "Diluted", or "EPS"
  - Non-dollar share quantities: values with no $ sign where the row or column unit is
    shares, units, options, RSUs, or PSUs. Dollar amounts in the same table must still
    be extracted — a $ value is always a financial figure regardless of adjacent share columns.
  - Dates and calendar references
  - Values from a stock return comparison chart (multi-year cumulative return tables
    where a hypothetical investment is indexed to a base value at the start of the period)
  - Change columns ($Change, %Change) — period columns only
  - Any value appearing exclusively in a non-financial context: workforce headcount,
    diversity statistics, sustainability disclosures, or ESG targets. If the same value
    also appears in a financial table or note, keep it.
  - Non-monetary business metrics: counts of products, users, systems, or customers
  - Any candidate that is a label, identifier, or reference rather than an actual
    numeric measurement — a Financial Item must represent an amount, rate, or quantity.
    If it does not start with a digit, $, or ( and contains no % or scale word
    (billion/million/trillion/thousand), discard it.

Return ONLY valid JSON:
{{"entities": [{{"name": "entity name", "type": "entity type"}}]}}
If all discarded, return: {{"entities": []}}"""


def classify_entities(client: OpenAI, candidates: list, text: str,
                      filing_company: str, known_entities: list[dict] | None = None) -> list[dict]:
    seen_names: set[str] = set()
    deduped_known: list[dict] = []
    for e in (known_entities or []):
        if e["type"] != "Financial Item" and e["name"] not in seen_names:
            seen_names.add(e["name"])
            deduped_known.append(e)

    user_content = (
        f"Candidates to classify:\n{json.dumps(candidates)}\n\n"
        f"Original text for context:\n{text}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": build_classifier_prompt(filing_company, deduped_known)},
            {"role": "user",   "content": user_content}
        ],
        temperature=0,
        max_tokens=8192,
        response_format={"type": "json_object"}
    )
    try:
        return json.loads(response.choices[0].message.content).get("entities", [])
    except json.JSONDecodeError:
        return []
