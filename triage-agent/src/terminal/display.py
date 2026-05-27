from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from typing import List, Dict

console = Console()

def show_ticket(ticket: Dict):
    console.rule(f"Ticket: {ticket.get('ticket_id','-')}")
    body = Text(ticket.get('text',''))
    console.print(Panel(body, title="Incoming Ticket", border_style="cyan"))


def show_classification(summary: Dict):
    table = Table(show_header=False, box=None)
    table.add_row("Domain", summary.get("domain", "unknown"))
    table.add_row("Intent", summary.get("intent", "other"))
    table.add_row("Urgency", summary.get("urgency", "low"))
    table.add_row("Risk", f"{summary.get('risk', 0.0):.2f}")
    table.add_row("Retrieval confidence", f"{summary.get('retrieval_confidence', 0.0):.2f}")
    table.add_row("Action", summary.get("action", "answer"))
    console.print(Panel(table, title="Triage Summary", border_style="green"))

def show_retrieved(chunks: List[Dict]):
    table = Table(show_header=True, header_style="bold magenta", title="Retrieved Sources")
    table.add_column("Score")
    table.add_column("Domain")
    table.add_column("Title")
    table.add_column("Snippet")
    for idx, c in enumerate(chunks, start=1):
        s = f"{c['score']:.3f}"
        meta = c['meta']
        snippet = meta.get('text', '')[:120].replace('\n', ' ')
        table.add_row(f"#{idx} {s}", meta.get('domain',''), meta.get('title',''), snippet)
    console.print(table)

def show_decision(decision: Dict):
    action = decision.get('action', 'answer')
    confidence = decision.get('score', 0.0)
    body = f"Action: {action}\nReason: {decision.get('reason')}\nConfidence: {confidence:.2f}"
    style = "red" if action in {"refuse", "escalate"} else "yellow"
    console.print(Panel(body, title="Decision", border_style=style))
