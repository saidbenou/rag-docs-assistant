"""Evaluation on the question set.

    python -m scripts.evaluate --retrieval-only      # retrieval only, fast, no LLM
    python -m scripts.evaluate                       # + LLM answers
    python -m scripts.evaluate --ragas               # + Ragas metrics (judge LLM, slow locally)
    python -m scripts.evaluate --ragas --metrics faithfulness,context_recall --limit 5

Measures:
- retrieval     : is the expected page in the top-k, and at which rank (Hit@k, MRR);
- answer        : do the numbers of the expected answer appear in the generated answer;
- out of corpus : does the assistant decline to answer instead of making things up;
- Ragas (option): faithfulness, answer_accuracy, context_recall... scored by a judge LLM.
Writes a Markdown report and the raw results as JSON in eval/. When Langfuse is
configured, each question is traced and its scores are attached to its trace.
"""
import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from app import observability
from app.config import get_settings
from app.retrieval.retriever import retrieve

console = Console()

# Refusal phrasing, in French (the corpus language) and in English.
REFUSAL = re.compile(
    r"ne (permet|permettent|contien|mentionn|précis|fourni|trouv)|pas d'information|aucune information"
    r"|n'est pas (mentionn|précis|indiqu)|ne sais pas|impossible de répondre|pas en mesure"
    r"|(do|does) not (contain|mention|specify|provide)|no information|not (mentioned|specified)"
    r"|cannot answer|unable to answer|don't know",
    re.IGNORECASE,
)

SHORT = {"faithfulness": "Faith.", "answer_accuracy": "Accur.", "context_recall": "Recall",
         "context_precision": "Prec.", "answer_relevancy": "Relev.", "factual_correctness": "Fact."}


def numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in re.findall(r"\d+(?:[.,]\d+)?", text)}


def answer_ok(expected: str, answer: str) -> bool:
    expected_nums = numbers(expected)
    if expected_nums:
        return expected_nums <= numbers(answer)
    return expected.split()[0].lower().strip(",.") in answer.lower()  # e.g. "Oui, ..." (French for "yes")


def evaluate(questions: list[dict], retrieval_only: bool, metrics: list[str]) -> list[dict]:
    from app.generation.chain import answer

    scorers = {}
    if metrics:
        from app.evaluation.ragas_metrics import build_scorers

        scorers = build_scorers(metrics)

    rows = []
    for q in questions:
        start = time.perf_counter()
        in_corpus = q["source"] is not None
        row = {"id": q["id"], "question": q["question"], "expected": q["expected_answer"],
               "out_of_corpus": not in_corpus}
        if retrieval_only:
            docs = retrieve(q["question"])
            found = [(d.metadata["source"], d.metadata["page"]) for d in docs]
            top_score = docs[0].metadata.get("score") if docs else None
        else:
            result = answer(q["question"])
            found = [(s["source"], s["page"]) for s in result["sources"]]
            top_score = result["sources"][0]["score"] if result["sources"] else None
            row.update(answer=result["answer"], contexts=result["contexts"], trace_id=result["trace_id"])
            row["correct"] = (bool(REFUSAL.search(row["answer"])) if not in_corpus
                              else answer_ok(q["expected_answer"], row["answer"]))
        row["rank"] = next((i for i, f in enumerate(found, 1) if f == (q["source"], q["page"])), None)
        row["top1"] = f"{found[0][0]} p.{found[0][1]}" if found else None
        row["top1_score"] = top_score

        if scorers and in_corpus:
            from app.evaluation.ragas_metrics import score_row

            with console.status(f"[cyan]{q['id']}: Ragas scoring ({', '.join(scorers)})…"):
                row["ragas"] = score_row(row, scorers)
            failed = [n for n, v in row["ragas"].items() if v is None]
            if failed:
                console.print(f"[yellow]{q['id']}: the judge gave no valid score for "
                              f"{', '.join(failed)} (judge model too weak or unreachable?)[/]")
        if row.get("trace_id"):
            push_scores(row)
        row["seconds"] = round(time.perf_counter() - start, 1)
        rows.append(row)
        console.log(f"{q['id']} done ({row['seconds']} s)")
    observability.flush()
    return rows


def push_scores(row: dict) -> None:
    """Attach the scores computed here to the question's Langfuse trace."""
    tid = row["trace_id"]
    observability.score(tid, "eval-correct", int(row["correct"]), comment=f"expected: {row['expected']}")
    if not row["out_of_corpus"]:
        observability.score(tid, "eval-retrieval-rank", row["rank"] or 0)
    for name, value in (row.get("ragas") or {}).items():
        if value is not None:
            observability.score(tid, f"ragas-{name}", value)


def summary(rows: list[dict], retrieval_only: bool, metrics: list[str]) -> dict:
    from app.evaluation.ragas_metrics import mean

    k = get_settings().top_k
    inside = [r for r in rows if not r["out_of_corpus"]]
    out = {
        "Questions": len(rows),
        "Hit@1": sum(r["rank"] == 1 for r in inside) / len(inside),
        f"Hit@{k}": sum(r["rank"] is not None for r in inside) / len(inside),
        "MRR": sum(1 / r["rank"] for r in inside if r["rank"]) / len(inside),
    }
    if not retrieval_only:
        outside = [r for r in rows if r["out_of_corpus"]]
        out["Answer accuracy"] = sum(r["correct"] for r in inside) / len(inside)
        if outside:
            out["Out-of-corpus refusals"] = sum(r["correct"] for r in outside) / len(outside)
    for name in metrics:
        out[f"Ragas {name}"] = mean([r["ragas"][name] for r in inside if "ragas" in r])
    out["Mean time (s)"] = sum(r["seconds"] for r in rows) / len(rows)
    return out


def fmt(name: str, value: float | int | None) -> str:
    if value is None:
        return "—"
    if name == "Questions":
        return str(value)
    if "time" in name:
        return f"{value:.1f}"
    if name == "MRR" or name.startswith("Ragas"):
        return f"{value:.2f}"
    return f"{value:.0%}"


def colored(value: float | None) -> str:
    if value is None:
        return "[dim]—[/]"
    color = "green" if value >= 0.8 else "yellow" if value >= 0.5 else "red"
    return f"[{color}]{value:.2f}[/]"


def print_report(rows: list[dict], stats: dict, retrieval_only: bool, metrics: list[str]) -> None:
    table = Table(title="Results per question", title_justify="left", header_style="bold", expand=True)
    table.add_column("ID", width=4)
    table.add_column("Question", ratio=4, overflow="fold")
    table.add_column("Rank", justify="center", width=5)
    table.add_column("Top 1", ratio=3, overflow="fold")
    if not retrieval_only:
        table.add_column("Answer", justify="center", width=7)
    for name in metrics:
        table.add_column(SHORT.get(name, name), justify="center", width=7)
    for r in rows:
        rank = "[dim]n/a[/]" if r["out_of_corpus"] else (
            "[green]1[/]" if r["rank"] == 1 else f"[yellow]{r['rank']}[/]" if r["rank"] else "[red]✘[/]")
        cells = [r["id"], r["question"] + (" [dim](out of corpus)[/]" if r["out_of_corpus"] else ""), rank, r["top1"]]
        if not retrieval_only:
            cells.append("[green]✔[/]" if r["correct"] else "[red]✘[/]")
        cells += [colored((r.get("ragas") or {}).get(name)) for name in metrics]
        table.add_row(*cells)
    console.print(table)

    synth = Table(title="Summary", title_justify="left", show_header=False)
    synth.add_column(style="bold")
    synth.add_column(justify="right")
    for name, value in stats.items():
        synth.add_row(name, fmt(name, value))
    console.print(synth)


def write_markdown(path: Path, rows: list[dict], stats: dict, retrieval_only: bool, metrics: list[str]) -> None:
    s = get_settings()
    model = s.ollama_model if s.llm_provider == "ollama" else s.openai_model
    lines = [
        "# Evaluation report", "",
        f"- Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Embeddings: `{s.embedding_model}` · chunks {s.chunk_size}/{s.chunk_overlap} · top-k {s.top_k}",
        f"- LLM: {'not evaluated (--retrieval-only)' if retrieval_only else f'`{model}`'}",
    ]
    if metrics:
        from app.evaluation.ragas_metrics import judge_model

        lines.append(f"- Ragas: judge `{judge_model()}` · metrics {', '.join(metrics)}")
    lines += ["", "## Summary", "", "| Metric | Value |", "|---|---:|",
              *[f"| {n} | {fmt(n, v)} |" for n, v in stats.items()], "", "## Details", ""]
    for r in rows:
        status = "out of corpus" if r["out_of_corpus"] else f"rank {r['rank'] or '✘'}"
        if not retrieval_only:
            status += " · " + ("✔ correct" if r["correct"] else "✘ incorrect")
        lines += [f"### {r['id']} — {r['question']}", "", f"*{status} · top 1: {r['top1']}*", "",
                  f"- **Expected:** {r['expected']}"]
        if not retrieval_only:
            lines.append(f"- **Got:** {' '.join(r['answer'].split())}")
        if r.get("ragas"):
            lines.append("- **Ragas:** " + " · ".join(f"{n} {fmt('Ragas', v)}" for n, v in r["ragas"].items()))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.disable(logging.WARNING)
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    parser = argparse.ArgumentParser(description="Evaluate the RAG on a question set.")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--retrieval-only", action="store_true", help="Evaluate retrieval only (no LLM)")
    parser.add_argument("--ragas", action="store_true", help="Add the Ragas metrics (judge LLM)")
    parser.add_argument("--metrics", default=None, help="Comma-separated Ragas metrics")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")
    parser.add_argument("--out", default="eval", help="Output folder for the reports")
    args = parser.parse_args()
    if args.ragas and args.retrieval_only:
        parser.error("--ragas needs the LLM answers: remove --retrieval-only")

    metrics: list[str] = []
    if args.ragas:
        from app.evaluation.ragas_metrics import ALL_METRICS, DEFAULT_METRICS

        metrics = args.metrics.split(",") if args.metrics else DEFAULT_METRICS
        unknown = set(metrics) - set(ALL_METRICS)
        if unknown:
            parser.error(f"unknown metrics: {', '.join(unknown)} (available: {', '.join(ALL_METRICS)})")

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))[: args.limit]
    rows = evaluate(questions, args.retrieval_only, metrics)
    stats = summary(rows, args.retrieval_only, metrics)
    print_report(rows, stats, args.retrieval_only, metrics)

    out = Path(args.out)
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    raw = [{k: v for k, v in r.items() if k != "contexts"} for r in rows]
    (out / f"results_{stamp}.json").write_text(
        json.dumps({"summary": stats, "questions": raw}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out / f"report_{stamp}.md", rows, stats, args.retrieval_only, metrics)
    console.print(f"\n[dim]Report written to {out / f'report_{stamp}.md'}[/]")
    if observability.get_langfuse() is not None and not args.retrieval_only:
        console.print(f"[dim]Traces and scores sent to Langfuse ({get_settings().langfuse_host})[/]")


if __name__ == "__main__":
    main()
