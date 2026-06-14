from openai import OpenAI
import json

MODEL = "gpt-4.1-mini"

STATEMENT_EXTRACTOR_PROMPT = """You are reading a chunk of a financial 10-K document.

Identify whether this chunk contains data from one or more of these three financial statements:
  BALANCE_SHEET     — Consolidated Balance Sheet (assets, liabilities, equity)
  INCOME_STATEMENT  — Consolidated Statement of Income / Operations (revenues, expenses, net income)
  CASH_FLOW         — Consolidated Statement of Cash Flows (operating, investing, financing)

You can recognise them by their section headers OR by distinctive body lines.

BALANCE_SHEET headers/lines:
  "CONSOLIDATED BALANCE SHEETS"
  "Total assets", "Total liabilities", "Stockholders' equity", "Total equity"

INCOME_STATEMENT headers/lines:
  "CONSOLIDATED STATEMENTS OF INCOME"
  "CONSOLIDATED STATEMENTS OF OPERATIONS"
  "Total revenues", "Net income", "Net income attributable to"

CASH_FLOW headers/lines — trigger on ANY of these, even if the header is missing:
  "CONSOLIDATED STATEMENTS OF CASH FLOWS"
  "Cash Flows from Operating Activities"
  "Cash Flows from Investing Activities"
  "Cash Flows from Financing Activities"
  "Net cash provided by operating activities"
  "Net cash used in investing activities"
  "Net cash used in financing activities"
  "Net increase in cash", "Net decrease in cash"
  "Cash and cash equivalents at beginning of year"
  "Cash and cash equivalents at end of year"
  If you see ANY of these lines, the chunk is from a Cash Flow statement — extract it.

If you find statement data:
- Extract ALL numeric line items visible in the chunk for each fiscal year column present.
- Use snake_case keys  (e.g. "total_investments", "net_income", "insurance_premiums").
- Copy values verbatim from the text  (e.g. "51,250", "(49)", "7,428").
- Note the unit if stated in the header  (e.g. "millions", "billions").
- If multiple fiscal years appear as separate columns, return each year as a separate statement object.
- If the chunk shows only part of a statement (e.g. only the Assets section), extract what is visible — do not wait for the rest.

DO NOT extract:
- Per-share values (EPS, diluted / basic earnings per share, dividends per share)
- Share counts, option counts, or any unit measured in shares
- Ratios or computed metrics (P/E, ROE, etc.)
- Data from footnotes, supplementary notes, or non-statement narrative sections

Return ONLY valid JSON — no explanation, no markdown:
{
  "statements": [
    {
      "type": "BALANCE_SHEET",
      "fiscal_year": "2019",
      "unit": "millions",
      "items": {
        "total_investments": "51,250",
        "cash": "336",
        "receivables": "7,675",
        "claim_and_claim_adjustment_expense": "21,720"
      }
    }
  ]
}
If no recognised financial statement section is found, return: {"statements": []}"""


def extract_statements(client: OpenAI, text: str,
                       filing_company: str) -> tuple[list[dict], list[dict]]:
    """
    Detect financial statement sections in a text chunk and return structured nodes.
    Returns (statement_entities, statement_relationships).

    statement_entities  : [{name, type, properties}]
    statement_relationships: [{source, target, type, property}]
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": STATEMENT_EXTRACTOR_PROMPT},
            {"role": "user",   "content": text}
        ],
        temperature=0,
        max_tokens=4096,
        response_format={"type": "json_object"}
    )
    try:
        statements = json.loads(response.choices[0].message.content).get("statements", [])
    except json.JSONDecodeError:
        statements = []

    VALID_TYPES = {"BALANCE_SHEET", "INCOME_STATEMENT", "CASH_FLOW"}
    # node_label maps relationship type → Neo4j-safe entity type (no underscores)
    NODE_LABEL  = {
        "BALANCE_SHEET":    "BalanceSheet",
        "INCOME_STATEMENT": "IncomeStatement",
        "CASH_FLOW":        "CashFlow",
    }

    statement_entities:      list[dict] = []
    statement_relationships: list[dict] = []
    seen_nodes: set[str] = set()

    for stmt in statements:
        stmt_type   = str(stmt.get("type", "")).upper().strip()
        fiscal_year = str(stmt.get("fiscal_year", "")).strip()
        items       = stmt.get("items") or {}

        if stmt_type not in VALID_TYPES or not fiscal_year or not items:
            continue

        node_label = NODE_LABEL[stmt_type]
        node_name  = f"{node_label}_FY{fiscal_year}"

        if node_name in seen_nodes:
            continue
        seen_nodes.add(node_name)

        properties = {k: str(v) for k, v in items.items() if v is not None and str(v).strip()}
        properties["fiscal_year"] = fiscal_year
        if stmt.get("unit"):
            properties["unit"] = str(stmt["unit"])

        statement_entities.append({
            "name":       node_name,
            "type":       node_label,
            "properties": properties,
        })
        statement_relationships.append({
            "source":   filing_company,
            "target":   node_name,
            "type":     stmt_type,
            "property": fiscal_year,
        })

    return statement_entities, statement_relationships
