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
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import ExperimentConfig, LLMConfig
from .evaluate import run_ragas_on_experiment, sanitize_judge_tag
from .experiment import list_experiments, run_experiment
from .settings import get_settings
from .utils.io import read_json
from .utils.logging import setup_logging

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

CONFIG_OPTION = typer.Option(
    ..., "--config", "-c", help="Config name, e.g. experiment/fixture_smoke"
)
OVERRIDE_OPTION = typer.Option(
    [],
    "--override",
    "-o",
    help="Hydra-style override, e.g. budget.num_triplets=3 (repeatable)",
)
LIST_FILTER_OPTION = typer.Option([], "--filter", help="key=value filters")
REPORT_FILTER_OPTION = typer.Option([], "--filter", help="key=value filters")
REPORT_METRICS_OPTION = typer.Option(
    "em,f1,rouge_l,faithfulness,answer_correctness",
    "--metrics",
    help="comma-separated metric column names",
)


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
    config: str = CONFIG_OPTION,
    override: list[str] = OVERRIDE_OPTION,
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
    filter: list[str] = LIST_FILTER_OPTION,
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
    filter: list[str] = REPORT_FILTER_OPTION,
    metrics: str = REPORT_METRICS_OPTION,
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


def _resolve_exp_dir(experiment_id: str) -> Path:
    s = get_settings()
    exp_dir = s.experiments_dir / experiment_id
    if exp_dir.exists():
        return exp_dir
    candidates = sorted(s.experiments_dir.glob(f"*{experiment_id}*"))
    if not candidates:
        typer.echo(f"No experiment found matching: {experiment_id}", err=True)
        raise typer.Exit(2)
    chosen = candidates[-1]
    console.print(f"[dim]Resolved -> {chosen.name}[/dim]")
    return chosen


@app.command(name="eval-ragas")
def eval_ragas_cmd(
    experiment_id: str = typer.Argument(..., help="Experiment id (or fuzzy substring)"),
    judge_model: str = typer.Option(
        "openai:gpt-4o",
        "--judge-model",
        "-j",
        help="<kind>:<model_name>; kind in {openai, anthropic, vllm, local_hf}",
    ),
    metrics: str = typer.Option(
        "faithfulness,answer_relevancy,answer_correctness",
        "--metrics",
        "-m",
        help="Comma-separated RAGAS metric names. "
        "Supported: faithfulness, answer_relevancy, answer_correctness, "
        "context_precision, context_recall, nv_accuracy, "
        "nv_response_groundedness, nv_context_relevance, factual_correctness, "
        "rouge_score, bleu_score, non_llm_string_similarity, string_present, exact_match.",
    ),
    judge_tag: str | None = typer.Option(
        None,
        "--judge-tag",
        help="Folder label for this run; defaults to the sanitized model name.",
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="OpenAI-compatible endpoint for the judge "
        "(e.g. http://localhost:7114/v1 for a self-hosted vLLM). "
        "Required when kind=vllm and the host differs from VLLM_BASE_URL.",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="API key for the judge endpoint. "
        "Defaults to VLLM_API_KEY (vllm) or the matching provider env var.",
    ),
    temperature: float = typer.Option(0.0, "--temperature"),
    max_tokens: int = typer.Option(1024, "--max-tokens"),
    force: bool = typer.Option(False, "--force", "-f"),
    max_workers: int | None = typer.Option(
        None,
        "--max-workers",
        help="Parallel RAGAS workers (ragas RunConfig.max_workers). Default: ragas built-in (16).",
    ),
    timeout: int | None = typer.Option(
        None,
        "--timeout",
        help="Per-call timeout in seconds (ragas RunConfig.timeout). "
        "Default: ragas built-in (180s).",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Print judge prompts via langchain_core.globals.set_debug(True).",
    ),
    dump_inputs: bool = typer.Option(
        False,
        "--dump-inputs",
        help="Dump the rows fed to ragas.evaluate as JSONL under "
        "metrics/ragas/<tag>/inputs/. Default single-evaluate mode writes "
        "ragas_inputs.jsonl.",
    ),
    ks: str | None = typer.Option(
        None,
        "--ks",
        help="Comma-separated retrieval ks at which to replicate "
        "context-dependent RAGAS metrics (e.g. '5,10,20'). "
        "Default: auto-derive from the experiment's metrics.retrieval_metrics "
        "@k suffixes; pass '' to force a single un-suffixed pass.",
    ),
    separate_scopes: bool = typer.Option(
        False,
        "--separate-scopes",
        help="Run each context-dependent k scope as a separate RAGAS call. "
        "Default merges all k rows for context-dependent metrics while "
        "evaluating context-free metrics once.",
    ),
) -> None:
    """Run RAGAS on a completed experiment's predictions.

    Results land in `experiments/<id>/metrics/ragas/<tag>/`. Existing metrics
    files are not modified, so you can compare across judge models.
    """
    s = get_settings()
    setup_logging(s.log_level)
    exp_dir = _resolve_exp_dir(experiment_id)

    if ":" not in judge_model:
        typer.echo(
            f"--judge-model must be '<kind>:<model_name>', got {judge_model!r}",
            err=True,
        )
        raise typer.Exit(2)
    kind, model_name = judge_model.split(":", 1)
    if kind not in ("openai", "anthropic", "vllm", "local_hf"):
        typer.echo(f"Unsupported judge kind: {kind}", err=True)
        raise typer.Exit(2)

    judge_cfg = LLMConfig(
        kind=kind,  # type: ignore[arg-type]
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    metric_names = [m.strip() for m in metrics.split(",") if m.strip()]
    tag = judge_tag or sanitize_judge_tag(model_name)

    # --ks: None  → auto-derive from experiment config
    # --ks ""     → explicit empty list (single un-suffixed pass)
    # --ks "5,10" → parse
    ks_list: list[int] | None = None
    if ks is not None:
        ks_list = []
        for piece in ks.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                ks_list.append(int(piece))
            except ValueError:
                typer.echo(f"--ks: '{piece}' is not an integer", err=True)
                raise typer.Exit(2) from None

    endpoint_str = f" @ {base_url}" if base_url else ""
    console.print(f"[bold]Experiment:[/bold] {exp_dir.name}")
    console.print(f"[bold]Judge:[/bold] {kind}:{model_name}{endpoint_str} (tag={tag})")
    console.print(f"[bold]Metrics:[/bold] {metric_names}")
    if ks_list is not None:
        console.print(f"[bold]ks (override):[/bold] {ks_list or 'single-pass'}")
    if max_workers is not None or timeout is not None:
        console.print(f"[bold]RunConfig:[/bold] max_workers={max_workers}, timeout={timeout}")
    if debug:
        console.print("[yellow]debug=True (judge prompts will be printed)[/yellow]")
    console.print(
        f"[bold]RAGAS mode:[/bold] {'separate k scopes' if separate_scopes else 'merged k scopes'}"
    )

    try:
        agg, out_dir = run_ragas_on_experiment(
            exp_dir=exp_dir,
            judge_cfg=judge_cfg,
            metric_names=metric_names,
            judge_tag=tag,
            judge_base_url=base_url,
            judge_api_key=api_key,
            force=force,
            max_workers=max_workers,
            timeout=timeout,
            debug=debug,
            dump_inputs=dump_inputs,
            ks=ks_list,
            single_evaluate=not separate_scopes,
        )
    except FileExistsError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from None

    console.print(f"\n[green]done[/green] -> {out_dir}")
    console.print("\n[bold]RAGAS aggregate (means):[/bold]")
    console.print_json(json.dumps(agg, default=str))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
