"""CLI: python -m scripts.ask "Your question"   (no argument: interactive mode)"""
import logging
import re
import sys
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app import observability
from app.config import get_settings

console = Console()


def model_label() -> str:
    s = get_settings()
    return s.ollama_model if s.llm_provider == "ollama" else s.openai_model


def score_bar(score: float | None, width: int = 10) -> Text:
    if score is None:
        return Text("—", style="dim")
    filled = round(max(0.0, min(1.0, score)) * width)
    color = "green" if score >= 0.6 else "yellow" if score >= 0.45 else "red"
    return Text("█" * filled + "░" * (width - filled), style=color) + Text(f" {score:.2f}")


def highlight_citations(answer: str) -> str:
    """Render the [1] references in bold in the Markdown output."""
    return re.sub(r"\[(\d+)\]", r"**[\1]**", answer)


def render(question: str, result: dict, elapsed: float) -> None:
    sources = result["sources"]
    cited = [s for s in sources if s["cited"]]

    console.print()
    console.print(Panel(Text(question, style="bold"), title="Question", title_align="left", border_style="cyan"))

    status = (
        "[green]● sourced answer[/]" if cited
        else "[yellow]● no source cited — check it[/]" if sources
        else "[red]● no passage found[/]"
    )
    console.print(
        Panel(
            Markdown(highlight_citations(result["answer"])),
            title="Answer",
            title_align="left",
            subtitle=f"{status}  [dim]· {model_label()} · {elapsed:.1f} s[/]",
            subtitle_align="right",
            border_style="green" if cited else "yellow",
            padding=(1, 2),
        )
    )

    if not sources:
        return
    table = Table(title="Retrieved sources", title_justify="left", header_style="bold", expand=True)
    table.add_column("#", justify="right", width=3)
    table.add_column("Cited", justify="center", width=5)
    table.add_column("Document", overflow="fold", ratio=2)
    table.add_column("Page", justify="right", width=4)
    table.add_column("Relevance", width=16)
    table.add_column("Excerpt", overflow="fold", ratio=4)
    for s in sources:
        excerpt = " ".join(s["excerpt"].split())
        table.add_row(
            str(s["id"]),
            "[green]✔[/]" if s["cited"] else "[dim]·[/]",
            s["source"],
            str(s["page"]),
            score_bar(s["score"]),
            excerpt[:160] + ("…" if len(excerpt) > 160 else ""),
            style=None if s["cited"] else "dim",
        )
    console.print(table)


def ask_once(question: str) -> None:
    from app.generation.chain import answer

    start = time.perf_counter()
    try:
        with console.status("[cyan]Retrieving passages and generating the answer…"):
            result = answer(question)
    except Exception as exc:
        console.print(Panel(f"[red]{exc}[/]", title="Error", title_align="left", border_style="red"))
        return
    render(question, result, time.perf_counter() - start)
    url = observability.trace_url(result.get("trace_id"))
    if url:
        console.print(f"[dim]Langfuse trace: {url}[/]")


def main() -> None:
    logging.disable(logging.WARNING)
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    try:
        run(" ".join(sys.argv[1:]).strip())
    finally:
        observability.flush()


def run(question: str) -> None:
    if question:
        ask_once(question)
        return
    console.print(Panel("Ask questions about the indexed documents.\nType [bold]q[/] to quit.",
                        title="RAG Docs Assistant", border_style="cyan"))
    while True:
        try:
            question = console.input("\n[bold cyan]❯ [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"q", "quit", "exit"}:
            break
        if question:
            ask_once(question)


if __name__ == "__main__":
    main()
