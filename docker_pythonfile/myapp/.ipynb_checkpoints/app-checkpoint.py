# app.py - v2
import os
import sys
import statistics as stats
import typer
from rich.console import Console
from rich.table import Table
from rich import box

app = typer.Typer(help="Demo CLI inside Docker using Typer + Rich")
console = Console()

@app.command()
def hello(name: str = typer.Option("world", help="Who to greet")):
    console.rule("[bold cyan]Hello")
    console.print(f":wave:  Hello, [bold green]{name}[/]!", highlight=False)

@app.command()
def summarize(nums: list[float] = typer.Argument(..., help="Numbers to summarize")):
    total = sum(nums)
    mean = stats.fmean(nums)
    t = Table(title="Summary", box=box.SIMPLE)
    t.add_column("count"), t.add_column("sum"), t.add_column("mean")
    t.add_row(str(len(nums)), f"{total:.3f}", f"{mean:.3f}")
    console.print(t)

@app.command()
def env(var: str = typer.Option("APP_ENV", help="Env var to read")):
    val = os.getenv(var, "<unset>")
    console.print(f"[bold]{var}[/] = {val}")

@app.command()
def echo():
    console.print(f"argv: {sys.argv[1:]}")

if __name__ == "__main__":
    app()
# -------------------------------------------------------------
# app.py - v1
# import sys
# print("Hello from inside Docker, args:", sys.argv[1:])