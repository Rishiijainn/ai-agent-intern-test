import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from src.agent import AsterRowAgent

console = Console()

def run_chat_cli():
    console.print(Panel.fit(
        "[bold cyan]Aster & Row AI Support Agent[/bold cyan]\n"
        "[dim]Ask policy questions, check order status (e.g. ORD-1001), or test follow-ups.\n"
        "Type [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red] to end the session.[/dim]",
        border_style="cyan"
    ))

    agent = AsterRowAgent()
    session_id = "cli_user_session"

    while True:
        try:
            user_input = console.input("\n[bold green]You:[/bold green] ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Ending chat session. Goodbye![/yellow]")
                break

            with console.status("[bold blue]Thinking & Retrieving...[/bold blue]"):
                response = agent.process_message(user_input, session_id=session_id)

            console.print(Panel(
                Markdown(response.answer),
                title="[bold blue]Aster & Row Agent[/bold blue]",
                border_style="blue"
            ))

            if response.citations:
                cite_text = " | ".join([f"[cyan]{c.filename}[/cyan] > {c.heading}" for c in response.citations])
                console.print(f"[dim]📚 Sources Cited: {cite_text}[/dim]")

            if response.tool_called:
                console.print(f"[bold magenta]🔧 Tool Executed:[/bold magenta] {response.tool_called} with args {response.sanitized_tool_args}")

            if response.human_handoff_recommended:
                console.print("[bold red]⚠️ Human Support Escalation Recommended[/bold red]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting...[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    run_chat_cli()