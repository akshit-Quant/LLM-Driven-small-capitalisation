#!/usr/bin/env python3
"""Tyche CLI + pipeline entrypoint — agentic FinBERT news-sentiment extraction.

    uv run python -m tyche.news.sentiment_pipeline run  [--input PATH] [--output PATH] [--limit N]
    uv run python -m tyche.news.sentiment_pipeline audit-a
    uv run python -m tyche.news.sentiment_pipeline audit-b [--input PATH] [--limit N]
    uv run python -m tyche.news.sentiment_pipeline audit-c [--input PATH] [--limit N]

Also runnable as the console script installed by ``[project.scripts]``: ``uv run tyche ...``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tyche.common.logging import configure_logging, get_logger
from tyche.news.agents import (
    auditor,
    ingest,
    neutralizer,
    qwen_enricher,
    scorer,
    summarizer,
)
from tyche.news.config import settings
from tyche.news.graph import build_graph
from tyche.news.records import OUTPUT_COLUMNS, backend_score_columns

log = get_logger("tyche.main")


def _ingest_limited(input_path: str | None, limit: int | None) -> pd.DataFrame:
    # Cap the source read so the multi-GB feed is never fully loaded; limit is the
    # number of source articles (a row explodes into one row per ticker).
    return ingest.ingest(input_path, nrows=limit)


def _write_contract(neutralized: pd.DataFrame, output_path: str | None) -> pd.DataFrame:
    # Every configured sentiment backend gets its own <backend>_-prefixed columns
    # (see tyche.news.agents.scorer); the primary backend's copy already rides along
    # via OUTPUT_COLUMNS's canonical agg_p_pos/raw_score/... columns.
    backend_columns = [
        c
        for backend in settings.sentiment_backends.active
        for c in backend_score_columns(backend)
    ]
    columns = list(dict.fromkeys([*OUTPUT_COLUMNS, *backend_columns]))
    contract = neutralized[[c for c in columns if c in neutralized.columns]].copy()
    out_path = Path(str(output_path or settings.paths.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    contract.to_parquet(out_path, index=False)
    log.info("wrote %d rows to %s", len(contract), out_path)
    return contract


def run(
    input_path: str | None = None,
    output_path: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Score a news file end-to-end. Returns the contract frame and writes parquet."""
    auditor.audit_a()  # startup guard — halts before scoring if the model is wrong

    if limit:
        # Bounded run: ingest a slice directly, then push it through the same
        # agents the graph would use (skips re-reading the full feed via state).
        ingested = _ingest_limited(input_path, limit)
        enriched = qwen_enricher.enrich(summarizer.summarize(ingested))
        scored = scorer.score(enriched)
        neutralized = neutralizer.neutralize(scored)
        auditor.audit_d(neutralized)
    else:
        graph = build_graph()
        final_state = graph.invoke({"input_path": input_path})
        neutralized = final_state["neutralized"]

    return _write_contract(neutralized, output_path)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Tyche FinBERT sentiment pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="score a news file end-to-end")
    p_run.add_argument("--input", default=None)
    p_run.add_argument("--output", default=None)
    p_run.add_argument("--limit", type=int, default=None)

    sub.add_parser("audit-a", help="verify label order + sanity sentences")

    for name, helptext in [
        ("audit-b", "build entity_prior artifact"),
        ("audit-c", "causality verification"),
    ]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--input", default=None)
        p.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    if args.command == "run":
        run(args.input, args.output, args.limit)
    elif args.command == "audit-a":
        auditor.audit_a()
        log.info("audit-a OK")
    elif args.command == "audit-b":
        auditor.audit_b(_ingest_limited(args.input, args.limit))
    elif args.command == "audit-c":
        ingested = _ingest_limited(args.input, args.limit)
        scored = scorer.score(qwen_enricher.enrich(summarizer.summarize(ingested)))
        auditor.audit_c(scored)
        log.info("audit-c OK")


if __name__ == "__main__":
    main()
