"""CLI entrypoints.

Usage:
    triplet-rag run --config experiment/fixture_smoke
    triplet-rag run --config experiment/squad_vanilla_pilot \\
        --override budget.total_context_items=20
    triplet-rag inspect <experiment_id>
    triplet-rag list --filter dataset=squad
    triplet-rag report --filter experiment_name=squad_*_pilot

We use Hydra under the hood for config composition and overrides, but expose a
typer CLI rather than the hydra @main decorator, because the latter is awkward
when you want a multi-subcommand tool.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import ExperimentConfig
from .experiment import list_experiments, run_experiment
from .settings import get_settings
from .utils.io import read_json
from .utils.logging import setup_logging

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def _load_config(config: str, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load and compose a Hydra config from configs/<config>.yaml plus overrides."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[2]
    cfg_root = repo_root / "configs"
    if not cfg_root.exists():
        cfg_root = Path.cwd() / "configs"
    if not cfg_root.exists():
        typer.echo(f"Could not find configs/ at {cfg_root}", err=True)
        raise typer.Exit(2)

    overrides = overrides or []

    # Strip ".yaml" if user passed it
    cfg_name = config
    if cfg_name.endswith(".yaml"):
        cfg_name = cfg_name[:-5]

    with initialize_config_dir(version_base=None, config_dir=str(cfg_root.resolve())):
        cfg = compose(config_name=cfg_name, overrides=overrides)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError("Config did not resolve to a dict")
    # Drop hydra/internal keys
    cfg_dict.pop("hydra", None)
    return ExperimentConfig(**cfg_dict)


@app.command()
def run(
    config: str = typer.Option(
        ..., "--config", "-c", help="Config name, e.g. experiment/fixture_smoke"
    ),
    override: list[str] = typer.Option(
        [],
        "--override",
        "-o",
        help="Hydra-style override, e.g. budget.num_triplets=3 (repeatable)",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-run all phases"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print resolved config and exit"),
) -> None:
    """Run an experiment end-to-end."""
    s = get_settings()
    setup_logging(s.log_level)
    cfg = _load_config(config, override)

    console.print(f"[bold]Experiment:[/bold] {cfg.experiment_name}")
    console.print(f"[bold]preprocessing_hash:[/bold] {cfg.preprocessing_hash}")
    console.print(f"[bold]index_hash:[/bold] {cfg.index_hash}")
    console.print(f"[bold]experiment_id:[/bold] {cfg.experiment_id}")

    if dry_run:
        console.print_json(json.dumps(cfg.model_dump(), default=str))
        return

    exp_dir = run_experiment(cfg, force=force)
    console.print(f"[green]done[/green] -> {exp_dir}")
    headline = exp_dir / "metrics" / "headline.json"
    if headline.exists():
        console.print("\n[bold]Headline metrics:[/bold]")
        console.print_json(json.dumps(read_json(headline), default=str))


@app.command()
def inspect(experiment_id: str) -> None:
    """Print config, status, and headline metrics for an experiment."""
    s = get_settings()
    exp_dir = s.experiments_dir / experiment_id
    if not exp_dir.exists():
        # try fuzzy: any dir starting with this id
        candidates = sorted(s.experiments_dir.glob(f"*{experiment_id}*"))
        if not candidates:
            typer.echo(f"No experiment found matching: {experiment_id}", err=True)
            raise typer.Exit(2)
        exp_dir = candidates[-1]
        console.print(f"[dim]Resolved -> {exp_dir.name}[/dim]")

    cfg_path = exp_dir / "config.yaml.json"
    status_path = exp_dir / "status.json"
    headline = exp_dir / "metrics" / "headline.json"
    aggregate = exp_dir / "metrics" / "aggregate.json"

    if status_path.exists():
        console.print("[bold]Status:[/bold]")
        console.print_json(json.dumps(read_json(status_path), default=str))
    if cfg_path.exists():
        console.print("\n[bold]Config:[/bold]")
        console.print_json(json.dumps(read_json(cfg_path), default=str))
    if headline.exists():
        console.print("\n[bold]Headline metrics:[/bold]")
        console.print_json(json.dumps(read_json(headline), default=str))
    if aggregate.exists():
        console.print("\n[bold]Aggregate (with CIs):[/bold]")
        console.print_json(json.dumps(read_json(aggregate), default=str))


def _filters_from_strs(filter_strs: list[str]) -> dict[str, str]:
    """Parse 'k=v,k2=v2' style filter args."""
    out: dict[str, str] = {}
    for s in filter_strs:
        for piece in s.split(","):
            piece = piece.strip()
            if not piece:
                continue
            if "=" not in piece:
                continue
            k, v = piece.split("=", 1)
            out[k.strip()] = v.strip()
    return out


@app.command(name="list")
def list_cmd(
    filter: list[str] = typer.Option([], "--filter", help="key=value filters"),
) -> None:
    """List registered experiments."""
    s = get_settings()
    df = list_experiments(s.storage_dir, _filters_from_strs(filter))
    if df.empty:
        console.print("[dim]No experiments registered[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    show_cols = [
        c
        for c in [
            "experiment_id",
            "experiment_name",
            "dataset",
            "strategy",
            "indexing_strategy",
            "student_model",
            "status",
            "em",
            "f1",
            "rouge_l",
            "faithfulness",
            "answer_correctness",
        ]
        if c in df.columns
    ]
    for c in show_cols:
        table.add_column(c)
    for _, row in df.iterrows():
        vals = []
        for c in show_cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}" if not (v != v) else "—")
            else:
                vals.append(str(v) if v is not None else "—")
        table.add_row(*vals)
    console.print(table)


@app.command()
def report(
    filter: list[str] = typer.Option([], "--filter", help="key=value filters"),
    metrics: str = typer.Option(
        "em,f1,rouge_l,faithfulness,answer_correctness",
        "--metrics",
        help="comma-separated metric column names",
    ),
) -> None:
    """Side-by-side comparison of experiments matching the filter."""
    s = get_settings()
    df = list_experiments(s.storage_dir, _filters_from_strs(filter))
    if df.empty:
        console.print("[dim]No matching experiments[/dim]")
        return
    cols = ["experiment_name", "strategy", "indexing_strategy", "student_model"]
    for m in metrics.split(","):
        m = m.strip()
        if m and m in df.columns:
            cols.append(m)
    cols = [c for c in cols if c in df.columns]
    table = Table(show_header=True, header_style="bold")
    for c in cols:
        table.add_column(c)
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}" if not (v != v) else "—")
            else:
                vals.append(str(v) if v is not None else "—")
        table.add_row(*vals)
    console.print(table)


@app.command()
def grid(
    grid_file: str = typer.Argument(..., help="Path to a YAML grid spec"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Run a grid of experiments specified in a YAML file.

    Grid file format:

        base: experiment/squad_vanilla_pilot
        sweep:
          - student.model_name: gpt-4o-mini
            inference.strategy: vanilla_rag
          - student.model_name: gpt-4o-mini
            inference.strategy: triplet_rag
            indexer.indexing_strategy: triplets

    Each sweep entry overrides the base config; one experiment per entry.
    """
    import yaml  # type: ignore

    s = get_settings()
    setup_logging(s.log_level)

    p = Path(grid_file)
    if not p.exists():
        typer.echo(f"Grid file not found: {p}", err=True)
        raise typer.Exit(2)
    grid_spec = yaml.safe_load(p.read_text())
    base = grid_spec["base"]
    sweep = grid_spec.get("sweep", [])

    n = len(sweep)
    console.print(f"[bold]Grid:[/bold] base={base}, n={n}")

    results = []
    for i, entry in enumerate(sweep, 1):
        overrides = [f"{k}={v}" for k, v in entry.items()]
        console.print(f"\n[bold]({i}/{n})[/bold] overrides={overrides}")
        cfg = _load_config(base, overrides)
        try:
            exp_dir = run_experiment(cfg, force=force)
            results.append((cfg.experiment_id, "DONE", exp_dir))
        except Exception as e:
            console.print(f"[red]FAILED[/red]: {e}")
            results.append((cfg.experiment_id, "FAILED", None))

    console.print("\n[bold]Grid complete[/bold]")
    for eid, status, _ in results:
        console.print(f"  {status}\t{eid}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
