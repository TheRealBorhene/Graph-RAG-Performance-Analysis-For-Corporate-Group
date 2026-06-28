"""
RAGAS evaluation for the Graph RAG system — 20 questions, 3 metrics.

Metrics
-------
  - Faithfulness        : fraction of claims in the answer supported by the
                          retrieved context. Catches hallucination.
  - Answer Relevancy    : semantic match between the question and the answer.
                          Catches off-topic / vague responses.
  - Answer Correctness  : factual match between the answer and a written
                          ground truth. The headline number for factual quality.

Pipeline per question
---------------------
  1. Run the hybrid retrieval (graph → vector fallback) via query1.py.
  2. Collect (question, answer, contexts, ground_truth).
  3. RAGAS scores all three metrics using OpenAI as the judge LLM.
  4. Aggregate + per-question scores are written to .md and .csv.

Install once:
    pip install ragas datasets langchain_openai
"""

import csv
import math
import os
from datetime import datetime

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from query1 import (
    connect_neo4j,
    fetch_entity_catalog,
    _graph_fetch,
    _vector_fetch,
    _synthesize,
)

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EVAL_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "evals")
JUDGE_MODEL  = "gpt-4.1-mini"
EMBED_MODEL  = "text-embedding-3-small"

# The three RAGAS metrics — single source of truth used everywhere
# (score attach, aggregation, console table, .md report, .csv report).
METRICS = ("faithfulness", "answer_relevancy", "answer_correctness")
METRIC_LABELS = {
    "faithfulness":       "Faithfulness",
    "answer_relevancy":   "Answer Relevancy",
    "answer_correctness": "Answer Correctness",
}


# ─────────────────────────────────────────────
# BENCHMARK — 10 questions chosen to cover the full architecture:
#   * every edge type (PARENT_OF, GENERATED, OPERATES_IN, INCOME_STATEMENT)
#   * both retrieval paths (graph + vector fallback)
#   * single-fact, multi-fact, comparison, and geography-list answers
#   * one known-hard case (Q9 — Cypher mapping limitation) kept for honesty
#
# Ground truths reflect what the current Neo4j graph actually contains
# (see data/graph_output.json).
# ─────────────────────────────────────────────
BENCHMARK = [
    # ── Q1 — PARENT_OF, list query, multi-entity answer ──────────────────
    {
        "question": "What subsidiaries does Loews Corporation own?",
        "ground_truth": (
            "Loews Corporation owns six subsidiaries: CNA Financial Corporation, "
            "Diamond Offshore Drilling Inc., Boardwalk Pipeline Partners LP, "
            "Loews Hotels Holding Corporation, Altium Packaging LLC, and BPHC "
            "(Boardwalk Pipelines Holding Corp)."
        ),
    },
    # ── Q2 — INCOME_STATEMENT, consolidated metric, single value ─────────
    {
        "question": "How much net income did CNA Financial Corporation generate in 2019?",
        "ground_truth": (
            "CNA Financial Corporation generated net income of $894 million "
            "attributable to Loews Corporation in fiscal year 2019."
        ),
    },
    # ── Q3 — INCOME_STATEMENT, historical year, value continuity test ────
    {
        "question": "What was CNA Financial Corporation's net income in 2018?",
        "ground_truth": (
            "CNA Financial Corporation generated net income of $726 million "
            "attributable to Loews Corporation in fiscal year 2018."
        ),
    },
    # ── Q4 — OPERATES_IN, geography list answer ──────────────────────────
    {
        "question": "Where does Boardwalk Pipeline Partners operate?",
        "ground_truth": (
            "Boardwalk Pipeline Partners operates pipelines and storage facilities "
            "across nine U.S. states: Oklahoma, Arkansas, Tennessee, Kentucky, "
            "Illinois, Indiana, Ohio, Louisiana, and Texas."
        ),
    },
    # ── Q5 — VECTOR FALLBACK, narrative-only fact (graph has nothing) ────
    {
        "question": "What was the contract backlog of Diamond Offshore in 2020?",
        "ground_truth": (
            "Diamond Offshore Drilling reported a contract backlog of "
            "$1.6 billion at the start of fiscal year 2020."
        ),
    },
    # ── Q6 — GENERATED edge, comparison query, negative-value handling ───
    {
        "question": "Which Loews subsidiaries reported a net loss in 2019?",
        "ground_truth": (
            "Two Loews subsidiaries reported a net loss in 2019: Diamond "
            "Offshore Drilling (-$175 million) and Loews Hotels Holding "
            "Corporation (-$31 million)."
        ),
    },
    # ── Q7 — PARENT_OF, single-entity ownership lookup ───────────────────
    {
        "question": "Who is the parent company of Altium Packaging LLC?",
        "ground_truth": (
            "Altium Packaging LLC is owned by Loews Corporation, its parent company."
        ),
    },
    # ── Q8 — INCOME_STATEMENT, dual-source metric (standalone vs consol.)─
    {
        "question": "What were Diamond Offshore Drilling's contract drilling revenues in 2019?",
        "ground_truth": (
            "Diamond Offshore Drilling reported contract drilling revenues of "
            "approximately $935 million on a standalone basis, or $981 million "
            "as reported in the Loews consolidated income statement for fiscal "
            "year 2019."
        ),
    },
    # ── Q9 — INCOME_STATEMENT, known Cypher mapping limitation (HONESTY) ─
    # Kept in the benchmark even though it scores low. The graph holds the value
    # ($7,428M net earned premiums), but the Cypher generator maps "insurance
    # premiums" to a property key that does not exist. Documented limitation.
    {
        "question": "What were CNA's insurance premiums in 2019?",
        "ground_truth": (
            "CNA Financial reported net earned insurance premiums of "
            "$7,428 million in fiscal year 2019."
        ),
    },
    # ── Q10 — GENERATED edge, negative value, attribution chain ──────────
    {
        "question": "What was Diamond Offshore Drilling's net loss in 2018?",
        "ground_truth": (
            "Diamond Offshore Drilling reported a net loss of $112 million "
            "attributable to Loews Corporation in fiscal year 2018."
        ),
    },
    # ─────────────────────────────────────────────────────────────────────
    # SECOND TIER — 15 additional questions probing more metrics, more
    # years, balance sheet, cash flow, and narrative/vector-fallback paths.
    # Ground truths are kept conservative (only facts the graph holds) to
    # avoid faithfulness regression from over-specified expected answers.
    # ─────────────────────────────────────────────────────────────────────

    # ── Q11 — GENERATED, Loews Hotels net loss ───────────────────────────
    {
        "question": "What was Loews Hotels Holding Corporation's net loss in 2019?",
        "ground_truth": (
            "Loews Hotels Holding Corporation reported a net loss of $31 million "
            "attributable to Loews Corporation in 2019."
        ),
    },
    # ── Q12 — INCOME_STATEMENT, segment revenue ──────────────────────────
    {
        "question": "What was Boardwalk Pipeline Partners' net income in 2019?",
        "ground_truth": (
            "Boardwalk Pipeline Partners generated net income of $209 million "
            "attributable to Loews Corporation in 2019."
        ),
    },
    # ── Q13 — Altium revenue 2018 ────────────────────────────────────────
    {
        "question": "What was Altium Packaging LLC's revenue in 2018?",
        "ground_truth": (
            "Altium Packaging LLC reported revenue of approximately "
            "$867 million in fiscal year 2018."
        ),
    },
    # ── Q14 — GENERATED, gross premiums (different from Q9 metric) ───────
    {
        "question": "What were CNA Financial's gross written premiums in 2019?",
        "ground_truth": (
            "CNA Financial reported gross written premiums of $11,704 million "
            "in fiscal year 2019."
        ),
    },
    # ── Q15 — OPERATES_IN, CNA international ─────────────────────────────
    {
        "question": "Where does CNA Financial Corporation operate internationally?",
        "ground_truth": (
            "CNA Financial Corporation operates internationally in the United "
            "Kingdom, Canada, Luxembourg, Bermuda, and through Lloyd's of London."
        ),
    },
    # ── Q16 — INCOME_STATEMENT, equity income line item ──────────────────
    {
        "question": "What was Loews Hotels' equity income from joint ventures in 2019?",
        "ground_truth": (
            "Loews Hotels Holding Corporation reported equity income from "
            "joint ventures of $69 million in fiscal year 2019."
        ),
    },
    # ── Q17 — OPERATES_IN, Diamond Offshore international ────────────────
    {
        "question": "Where does Diamond Offshore Drilling operate?",
        "ground_truth": (
            "Diamond Offshore Drilling operates internationally with offshore "
            "drilling rigs and offices in the United States, the United Kingdom, "
            "Brazil, Mexico, Malaysia, Singapore, and Australia."
        ),
    },
    # ── Q18 — OPERATES_IN, Loews Hotels city list ────────────────────────
    {
        "question": "Where does Loews Hotels Holding Corporation operate?",
        "ground_truth": (
            "Loews Hotels Holding Corporation operates hotels in multiple U.S. "
            "cities and in Canada (Montreal and Toronto)."
        ),
    },
    # ── Q19 — PARENT_OF, CNA ownership lookup ────────────────────────────
    {
        "question": "Who is the parent company of CNA Financial Corporation?",
        "ground_truth": (
            "CNA Financial Corporation is owned by Loews Corporation, its parent company."
        ),
    },
    # ── Q20 — PARENT_OF, Diamond Offshore ownership lookup ───────────────
    {
        "question": "Who is the parent company of Diamond Offshore Drilling?",
        "ground_truth": (
            "Diamond Offshore Drilling is owned by Loews Corporation, its parent company."
        ),
    },
    # ── Q21 — INCOME_STATEMENT, Boardwalk segment revenue dual-source ────
    {
        "question": "What were Boardwalk Pipeline Partners' operating revenues in 2019?",
        "ground_truth": (
            "Boardwalk Pipeline Partners reported operating revenues of "
            "approximately $1,300 million as a standalone segment in fiscal "
            "year 2019, with the consolidated income statement reporting "
            "$1,266 million for transportation and storage from Boardwalk Pipelines."
        ),
    },
    # ── Q22 — INCOME_STATEMENT, Diamond drilling revenue 2018 ────────────
    {
        "question": "What were Diamond Offshore Drilling's contract drilling revenues in 2018?",
        "ground_truth": (
            "Diamond Offshore Drilling reported contract drilling revenues of "
            "approximately $939 million in fiscal year 2018."
        ),
    },
    # ── Q23 — VECTOR FALLBACK, narrative segment description ─────────────
    {
        "question": "What types of insurance products does CNA Financial offer?",
        "ground_truth": (
            "CNA Financial offers commercial property and casualty insurance products, "
            "including specialty insurance, surety, and warranty products, "
            "and operates through several underwriting business units."
        ),
    },
    # ── Q24 — VECTOR FALLBACK, business description ──────────────────────
    {
        "question": "What does Altium Packaging LLC produce?",
        "ground_truth": (
            "Altium Packaging LLC produces rigid plastic packaging products "
            "for consumer goods, food and beverage, and industrial markets, "
            "serving North American customers."
        ),
    },
    # ── Q25 — INCOME_STATEMENT, total revenues consolidated ──────────────
    {
        "question": "What were the consolidated total revenues of Loews Corporation in 2019?",
        "ground_truth": (
            "Loews Corporation reported consolidated total revenues of approximately "
            "$14,931 million in fiscal year 2019."
        ),
    },
]


# ─────────────────────────────────────────────
# CONTEXT PROSE-IFICATION FOR RAGAS
# Converts raw Cypher records into natural-language sentences so the judge LLM
# can verify the answer's prose claims against prose evidence. No new facts are
# introduced — only the same evidence presented in a format RAGAS was designed for.
# ─────────────────────────────────────────────

def _parse_kv(line: str) -> dict:
    """Parse 'k1: v1 | k2: v2 | ...' into a dict."""
    out: dict[str, str] = {}
    for pair in line.split(" | "):
        if ": " in pair:
            k, v = pair.split(": ", 1)
            out[k.strip()] = v.strip()
    return out


def _record_to_prose(record: str) -> str:
    """
    Convert one Cypher record (single or multi-line) into a verifiable sentence.

    Patterns handled (derived from the four edge types in GRAPH_SCHEMA):
      PARENT_OF       — p.name + s.name              → "<parent> owns subsidiary <sub>."
      GENERATED       — s.name + f.name + r.property → "<sub> reported <metric> in <year> (page <p>)."
      OPERATES_IN     — s.name + g.name              → "<sub> operates in <location>."
      Statement node  — s.name starts with statement → header + line items in one sentence.

    Records that don't match any pattern fall back to the original string.
    """
    record = record.strip()
    if not record:
        return record

    # Multi-line statement record: "header\n  key: value\n  key: value..."
    if "\n  " in record:
        header_line, *item_lines = record.split("\n")
        h = _parse_kv(header_line)
        stmt_name = h.get("s.name", "statement")
        fy   = h.get("s.fiscal_year", "")
        unit = h.get("s.unit", "")
        items = [ln.strip() for ln in item_lines if ln.strip()]
        unit_str = f" (unit: {unit})" if unit else ""
        fy_str   = f" for fiscal year {fy}" if fy else ""
        return (
            f"{stmt_name}{fy_str}{unit_str} reports the following line items: "
            + "; ".join(items) + "."
        )

    d = _parse_kv(record)

    # GENERATED edge with FinancialItem — the f.name carries the metric label and value.
    if "s.name" in d and "f.name" in d:
        year = d.get("r.property") or d.get("property") or ""
        page = d.get("r.page_number") or d.get("f.page_number") or ""
        year_str = f" in {year}" if year else ""
        page_str = f" (page {page})" if page else ""
        return f"{d['s.name']} reported {d['f.name']}{year_str}{page_str}."

    # PARENT_OF: parent + subsidiary (no financial item)
    if "p.name" in d and "s.name" in d:
        return f"{d['p.name']} owns subsidiary {d['s.name']}."

    # OPERATES_IN: subsidiary + geography
    if "s.name" in d and "g.name" in d:
        return f"{d['s.name']} operates in {d['g.name']}."

    # Standalone statement-node header (no items in this row)
    if "s.name" in d and d["s.name"].startswith(("IncomeStatement", "BalanceSheet", "CashFlow")):
        fy = d.get("s.fiscal_year", "")
        return f"Statement node {d['s.name']} for fiscal year {fy}."

    # Geography-only row
    if "g.name" in d and len(d) == 1:
        return f"Location: {d['g.name']}."

    # Parent-only row (e.g. PARENT_OF lookup returning just the owner)
    if "p.name" in d and len(d) == 1:
        return f"Parent company: {d['p.name']}."

    # Subsidiary-only row
    if "s.name" in d and len(d) == 1:
        return f"Subsidiary: {d['s.name']}."

    # Fallback — original record kept as-is so no information is lost
    return record


# ─────────────────────────────────────────────
# RETRIEVAL — hybrid pipeline reused from query1.py
# ─────────────────────────────────────────────

def _retrieve(question: str, driver, catalog: str) -> tuple[str, list[str], str]:
    """
    Run the hybrid pipeline for one question.

    Returns
    -------
    answer   : final synthesized answer string
    contexts : list of context strings (graph records OR vector passages)
    source   : 'graph' | 'vector' | 'empty'

    Graph contexts passed to RAGAS are prose-ified per-record so the faithfulness
    judge can verify the answer's natural-language claims against natural-language
    evidence. The synthesizer still receives the raw graph_ctx, so the production
    behaviour is unchanged — only the evaluation-time view of the same evidence is
    reformatted.
    """
    graph_ctx  = _graph_fetch(question, driver, catalog)
    vector_ctx = ""
    if not graph_ctx:
        vector_ctx = _vector_fetch(question)

    if graph_ctx:
        source = "graph"
        raw_records = [r for r in graph_ctx.split("\n\n") if r.strip()]
        contexts    = [_record_to_prose(r) for r in raw_records]
    elif vector_ctx:
        source   = "vector"
        contexts = [p.strip() for p in vector_ctx.split("\n---\n") if p.strip()]
    else:
        source   = "empty"
        contexts = ["No context retrieved."]

    answer = _synthesize(question, graph_ctx, vector_ctx)
    return answer, contexts, source


# ─────────────────────────────────────────────
# REPORT WRITERS
# ─────────────────────────────────────────────

def _write_md(path: str, rows: list[dict], scores: dict, timestamp: str) -> None:
    """Human-readable Markdown report with per-question breakdown + aggregates."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines: list[str] = []
    lines.append("# RAGAS Evaluation Report\n")
    lines.append(f"**Run timestamp:** {timestamp}\n")
    lines.append(f"**Judge model:** {JUDGE_MODEL}\n")
    lines.append(f"**Embedding model:** {EMBED_MODEL}\n")
    lines.append(f"**Questions evaluated:** {len(rows)}\n\n")

    overall = sum(scores.get(m, 0) for m in METRICS) / len(METRICS)
    lines.append("## Aggregate scores\n")
    lines.append("| Metric | Score (0-1) |")
    lines.append("|---|---:|")
    for m in METRICS:
        lines.append(f"| {METRIC_LABELS[m]:<19}| {scores.get(m, 0):.3f} |")
    lines.append(f"| **Average**         | **{overall:.3f}** |\n")

    lines.append("## Per-question scores\n")
    header_cells = " | ".join(METRIC_LABELS[m] for m in METRICS)
    lines.append(f"| # | Source | {header_cells} | Question |")
    lines.append("|---:|:---:|" + "---:|" * len(METRICS) + "---|")
    for i, r in enumerate(rows, 1):
        q     = r["question"].replace("|", "\\|")
        cells = " | ".join(_fmt_score(r.get(m)) for m in METRICS)
        lines.append(f"| {i} | {r['source']} | {cells} | {q} |")
    lines.append("")

    lines.append("## Detail\n")
    for i, r in enumerate(rows, 1):
        scores_inline = "  |  ".join(
            f"**{METRIC_LABELS[m]}:** {r.get(m, '—')!r}" for m in METRICS
        )
        lines.append(f"### Q{i}: {r['question']}\n")
        lines.append(f"**Source:** `{r['source']}`  |  {scores_inline}\n")
        lines.append(f"**Ground truth:**\n> {r['ground_truth']}\n")
        lines.append(f"**Generated answer:**\n> {r['answer']}\n")
        lines.append(f"**Retrieved contexts ({len(r['contexts'])} items, first 3 shown):**")
        for ctx in r["contexts"][:3]:
            short = ctx[:300] + ("..." if len(ctx) > 300 else "")
            lines.append(f"- `{short}`")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_csv(path: str, rows: list[dict], timestamp: str) -> None:
    """One row per question — question, source, answer, scores, ground truth."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "question", "source", *METRICS, "answer", "ground_truth"])
        for r in rows:
            w.writerow([
                timestamp, r["question"], r["source"],
                *[r.get(m, "") for m in METRICS],
                r["answer"], r["ground_truth"],
            ])


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _is_nan(x) -> bool:
    """True for None or NaN floats (RAGAS sometimes returns NaN on short answers)."""
    if x is None:
        return True
    return isinstance(x, float) and math.isnan(x)


def _fmt_score(x, dash: str = "—") -> str:
    """Format a metric score as 3-decimal string, or `dash` if missing/NaN."""
    return f"{x:.3f}" if isinstance(x, float) else dash


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_evaluation():
    print("═" * 60)
    print("  RAGAS EVALUATION  —  Faithfulness + Answer Relevancy + Answer Correctness")
    print("═" * 60)

    driver  = connect_neo4j()
    catalog = fetch_entity_catalog(driver)

    judge_llm  = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBED_MODEL))

    rows: list[dict] = []
    try:
        # ── Phase 1: run hybrid retrieval for every benchmark question ──────
        print(f"\nRetrieving answers for {len(BENCHMARK)} question(s)...\n")
        for i, entry in enumerate(BENCHMARK, 1):
            print(f"  [{i:>2}/{len(BENCHMARK)}] {entry['question'][:70]}")
            try:
                answer, contexts, source = _retrieve(entry["question"], driver, catalog)
            except Exception as exc:
                print(f"        retrieval failed: {exc}")
                answer, contexts, source = f"(retrieval failed: {exc})", ["(none)"], "empty"
            rows.append({
                "question":     entry["question"],
                "answer":       answer,
                "contexts":     contexts,
                "ground_truth": entry["ground_truth"],
                "source":       source,
            })
            print(f"        source={source}  answer={answer[:80].strip()}...")
    finally:
        driver.close()

    # ── Phase 2: RAGAS evaluation ───────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Running RAGAS metrics (faithfulness + answer_relevancy + answer_correctness)...")
    print(f"{'─' * 60}")

    ds = Dataset.from_dict({
        "question":     [r["question"]     for r in rows],
        "answer":       [r["answer"]       for r in rows],
        "contexts":     [r["contexts"]     for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })

    try:
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, answer_correctness],
            llm=judge_llm,
            embeddings=embeddings,
        )
    except Exception as exc:
        print(f"RAGAS evaluation failed: {exc}")
        return

    # ── Phase 3: attach per-question scores ─────────────────────────────────
    df = result.to_pandas()
    for i, row in df.iterrows():
        for m in METRICS:
            rows[i][m] = float(row[m]) if not _is_nan(row.get(m)) else None

    scores: dict[str, float] = {}
    counts: dict[str, int]   = {}
    for m in METRICS:
        vals = [r[m] for r in rows if isinstance(r.get(m), float)]
        scores[m] = sum(vals) / len(vals) if vals else 0.0
        counts[m] = len(vals)
    overall = sum(scores.values()) / len(scores)

    # ── Phase 4: console summary ────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  RAGAS RESULTS")
    print(f"{'─' * 60}")
    for m in METRICS:
        print(f"  {METRIC_LABELS[m]:<19} (avg over {counts[m]:>2} q): {scores[m]:.3f}")
    print(f"  {'Overall':<19} (mean of all three metrics): {overall:.3f}")
    print(f"{'═' * 60}")

    print("\n  Per-question scores:")
    short_labels = {"faithfulness": "Faith.", "answer_relevancy": "Relev.", "answer_correctness": "Corr."}
    header_cells = "  ".join(f"{short_labels[m]:>7}" for m in METRICS)
    print(f"  {'#':>3}  {'Source':<8}  {header_cells}  Question")
    print(f"  {'-'*3}  {'-'*8}  " + "  ".join('-'*7 for _ in METRICS) + f"  {'-'*60}")
    for i, r in enumerate(rows, 1):
        cells = "  ".join(f"{_fmt_score(r.get(m), '  —  '):>7}" for m in METRICS)
        print(f"  {i:>3}  {r['source']:<8}  {cells}  {r['question'][:60]}")

    # ── Phase 5: write report files ─────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    md_path  = os.path.join(EVAL_OUT_DIR, f"ragas_eval_{timestamp}.md")
    csv_path = os.path.join(EVAL_OUT_DIR, f"ragas_eval_{timestamp}.csv")
    _write_md(md_path, rows, scores, timestamp)
    _write_csv(csv_path, rows, timestamp)
    print(f"\n  Report saved: {md_path}")
    print(f"  Report saved: {csv_path}\n")


if __name__ == "__main__":
    run_evaluation()
