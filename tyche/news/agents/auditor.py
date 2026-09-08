"""Agent 6 — Auditor. Verifies the pipeline is honest. Four modes.

A · label-order + sanity sentences (startup, halts on failure)
B · entity-bias measurement → produces the entity_prior artifact (offline/quarterly)
C · causality verification → future rows must not change past scores
D · score-distribution health check (each run, in the DAG)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tyche.common.logging import get_logger
from tyche.news.agents import scorer, summarizer
from tyche.news.config import settings
from tyche.news.records import (
    Aggregate,
    Article,
    Neutralize,
    SanityDirection,
)

log = get_logger(__name__)


class AuditError(RuntimeError):
    """A hard audit failure — the pipeline must not emit scores."""


# --- Audit A ------------------------------------------------------------------
def audit_a() -> None:
    """Verify sanity-sentence directions. HALT on any failure.

    The Azure OpenAI scorer returns a labeled positive/negative/neutral distribution
    directly (no positional/label-order assumption), so this is a directional sanity
    check: each sentence must be dominated by its expected class with prob > 0.5.
    """
    revision = scorer.get_model_revision()  # warms the client + resolves revision
    sentences = list(settings.auditor.sanity_sentences)
    log.info(
        "Audit A START: checking %d sanity sentences on %s (rev=%s); "
        "each sentence must be dominated by its expected class with prob > 0.5",
        len(sentences),
        settings.azure.deployment,
        revision,
    )
    probs = scorer.score_texts([s["text"] for s in sentences])  # (n, 3) pos/neg/neu
    for row, (p_pos, p_neg, p_neu) in zip(sentences, probs):
        expect = SanityDirection(row["expect"])
        dominant = {
            0: SanityDirection.POS,
            1: SanityDirection.NEG,
            2: SanityDirection.NEU,
        }[int(np.argmax([p_pos, p_neg, p_neu]))]
        top = max(p_pos, p_neg, p_neu)
        status = "OK" if (dominant is expect and top > 0.5) else "FAIL"
        log.info(
            "Audit A check [%s]: %r expect=%s got pos=%.2f neg=%.2f neu=%.2f "
            "(dominant=%s, top=%.2f)",
            status,
            row["text"],
            expect.value,
            p_pos,
            p_neg,
            p_neu,
            dominant.value,
            top,
        )
        if status == "FAIL":
            raise AuditError(
                f"Audit A failed: {row['text']!r} expected {expect.value} but got "
                f"pos={p_pos:.2f} neg={p_neg:.2f} neu={p_neu:.2f}"
            )
    log.info("Audit A PASSED: label mapping + %d sanity sentences OK", len(sentences))


# --- Audit B ------------------------------------------------------------------
def _mask(text: str, ticker: str, name: str) -> str:
    placeholder = str(settings.ingest.masked_placeholder)
    masked = text
    for token in filter(None, [name, ticker]):
        masked = pd.Series([masked]).str.replace(
            token, placeholder, case=False, regex=False
        )[0]
    return masked


def audit_b(ingested: pd.DataFrame) -> dict:
    """Measure entity bias and persist the entity_prior artifact.

    Scores each article NAMED and MASKED through Summarizer→Scorer; the per-ticker
    prior is ``mean(raw_named − raw_masked)``, the per-group prior the mean across its
    tickers. This feeds Neutralizer step 0."""
    log.info(
        "Audit B START: measuring entity bias on %d (article, ticker) rows — scoring "
        "each article NAMED vs MASKED (placeholder=%r) and taking the raw_score gap",
        len(ingested),
        str(settings.ingest.masked_placeholder),
    )
    log.info("Audit B: scoring NAMED articles through Summarizer→Scorer")
    named = _score_through(ingested)
    masked_src = ingested.copy()
    masked_src[Article.full_text] = [
        _mask(r[Article.full_text], r[Article.ticker], r.get(Article.name, ""))
        for _, r in masked_src.iterrows()
    ]
    log.info("Audit B: scoring MASKED articles through Summarizer→Scorer")
    masked = _score_through(masked_src)

    merged = named.merge(
        masked[[Article.id, Article.ticker, Aggregate.raw_score]],
        on=[Article.id, Article.ticker],
        suffixes=("_named", "_masked"),
    )
    merged["gap"] = (
        merged[f"{Aggregate.raw_score}_named"] - merged[f"{Aggregate.raw_score}_masked"]
    )
    by_ticker = merged.groupby(Article.ticker)["gap"].mean()
    group_map = named.drop_duplicates(Article.ticker).set_index(Article.ticker)[
        Article.group_key
    ]
    merged[Article.group_key] = merged[Article.ticker].map(group_map)
    by_group = merged.groupby(Article.group_key)["gap"].mean()

    log.info(
        "Audit B: gap (named − masked) over %d rows → mean=%.4f std=%.4f "
        "min=%.4f max=%.4f across %d tickers / %d groups",
        len(merged),
        float(merged["gap"].mean()),
        float(merged["gap"].std(ddof=0)),
        float(merged["gap"].min()),
        float(merged["gap"].max()),
        by_ticker.size,
        by_group.size,
    )
    top20 = by_ticker.abs().sort_values(ascending=False).head(20)
    log.info("Audit B top-biased tickers (|prior|):\n%s", top20.to_string())

    artifact = {
        "model_revision": scorer.get_model_revision(),
        "by_ticker": {k: float(v) for k, v in by_ticker.items()},
        "by_group": {k: float(v) for k, v in by_group.items()},
    }
    path = Path(str(settings.neutralizer.entity_prior_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2))
    log.info(
        "Audit B PASSED: wrote entity_prior for %d tickers / %d groups to %s",
        len(by_ticker),
        by_group.size,
        path,
    )
    return artifact


def _score_through(ingested: pd.DataFrame) -> pd.DataFrame:
    """Summarize then score — so the raw_score reflects only the LLM's read of each
    summary. Audit B compares the same article named vs masked, so each paired row is
    scored directly."""
    return scorer.score(summarizer.summarize(ingested))


# --- Audit C ------------------------------------------------------------------
def audit_c(aggregated: pd.DataFrame) -> None:
    """Future rows must never change past scores. Neutralize the full set, then
    neutralize only the earliest 80% and assert those ``sentiment_final`` values are
    bit-for-bit identical."""
    from tyche.news.agents import neutralizer

    log.info(
        "Audit C START: causality check on %d rows — neutralize the FULL set vs the "
        "earliest days only, then assert past sentiment_final is bit-for-bit identical",
        len(aggregated),
    )
    ordered = aggregated.sort_values(Article.valid_time).reset_index(drop=True)
    # Cut on a TRADING-DAY boundary, never mid-day: step 2's z-score is a
    # same-day cross-sectional demean by design, so slicing inside a day would
    # change kept rows' scores for a reason unrelated to the trailing window
    # this audit is meant to test. Dropping whole future days leaves every kept
    # day's cross-sectional bucket intact, isolating step 1's causality.
    days = pd.to_datetime(ordered[Article.valid_time], utc=True).dt.normalize()
    unique_days = np.sort(days.unique())
    if len(unique_days) < 2:
        log.warning(
            "Audit C skipped: need ≥2 distinct trading days to test causality, got %d",
            len(unique_days),
        )
        return
    cutoff = unique_days[int(len(unique_days) * 0.8)]  # keep days strictly before it
    past = ordered[days < cutoff]
    log.info(
        "Audit C: %d distinct trading days; cutoff=%s → comparing %d past rows against "
        "their values in the full %d-row neutralization",
        len(unique_days),
        pd.Timestamp(cutoff).date(),
        len(past),
        len(ordered),
    )
    full = (
        neutralizer.neutralize(ordered)
        .sort_values([Article.id, Article.ticker])
        .reset_index(drop=True)
    )
    past_only = (
        neutralizer.neutralize(past)
        .sort_values([Article.id, Article.ticker])
        .reset_index(drop=True)
    )
    ref = full.merge(
        past_only, on=[Article.id, Article.ticker], suffixes=("_full", "_past")
    )
    delta = (
        (
            ref[f"{Neutralize.sentiment_final}_full"]
            - ref[f"{Neutralize.sentiment_final}_past"]
        )
        .abs()
        .max()
    )
    if pd.notna(delta) and delta > 0:
        raise AuditError(
            f"Audit C FAILED: past sentiment_final changed by up to {delta:.3e} when "
            "future rows were removed — the rolling window is leaking future data."
        )
    log.info(
        "Audit C PASSED: past scores invariant to future rows "
        "(n_past=%d, max|Δ sentiment_final|=%.3e)",
        len(past_only),
        float(delta) if pd.notna(delta) else 0.0,
    )


# --- Audit D ------------------------------------------------------------------
def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / max(len(expected), 1) + 1e-6
    a = np.histogram(actual, edges)[0] / max(len(actual), 1) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def audit_d(final: pd.DataFrame) -> dict:
    """Distribution health check; logs stats and raises alerts (non-fatal)."""
    log.info(
        "Audit D START: distribution health on %d final rows — checking raw_score & "
        "sentiment_final moments, tercile balance, PSI drift vs baseline, and "
        "per-group same-sign concentration",
        len(final),
    )

    def describe(x: pd.Series) -> dict:
        v = x.to_numpy()
        return {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "skew": float(stats.skew(v)) if len(v) > 2 else 0.0,
            "kurtosis": float(stats.kurtosis(v)) if len(v) > 3 else 0.0,
        }

    report = {
        "n": len(final),
        "raw_score": describe(final[Aggregate.raw_score]),
        "sentiment_final": describe(final[Neutralize.sentiment_final]),
    }
    dominant = (
        final[[Aggregate.p_pos, Aggregate.p_neg, Aggregate.p_neu]]
        .to_numpy()
        .argmax(axis=1)
    )
    report["terciles"] = {
        "positive": float(np.mean(dominant == 0)),
        "negative": float(np.mean(dominant == 1)),
        "neutral": float(np.mean(dominant == 2)),
    }
    log.info(
        "Audit D: raw_score %s; sentiment_final %s",
        report["raw_score"],
        report["sentiment_final"],
    )
    log.info(
        "Audit D: tercile balance pos=%.1f%% neg=%.1f%% neu=%.1f%%",
        report["terciles"]["positive"] * 100,
        report["terciles"]["negative"] * 100,
        report["terciles"]["neutral"] * 100,
    )

    # PSI vs baseline (if present); persist current as the next baseline.
    baseline_path = Path(str(settings.auditor.baseline_path))
    alerts: list[str] = []
    if baseline_path.exists():
        baseline = np.asarray(json.loads(baseline_path.read_text())["sentiment_final"])
        psi = _psi(baseline, final[Neutralize.sentiment_final].to_numpy())
        report["psi"] = psi
        log.info(
            "Audit D: PSI vs baseline (%s) = %.4f (threshold %.2f)",
            baseline_path,
            psi,
            float(settings.auditor.psi_threshold),
        )
        if psi > float(settings.auditor.psi_threshold):
            alerts.append(f"PSI {psi:.3f} exceeds {settings.auditor.psi_threshold}")
    else:
        log.info(
            "Audit D: no baseline at %s — skipping PSI, writing current as baseline",
            baseline_path,
        )
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        json.dumps({"sentiment_final": final[Neutralize.sentiment_final].tolist()})
    )

    # Same-sign concentration per group.
    thresh = float(settings.auditor.same_sign_alert)
    for group, sub in final.groupby(Article.group_key):
        signs = np.sign(sub[Neutralize.sentiment_final].to_numpy())
        signs = signs[signs != 0]
        if len(signs) and max(np.mean(signs > 0), np.mean(signs < 0)) > thresh:
            alerts.append(f"group {group!r} has >{thresh:.0%} same-sign scores")

    report["alerts"] = alerts
    for a in alerts:
        log.warning("Audit D alert: %s", a)
    log.info(
        "Audit D %s: %d rows, %d alert(s)",
        "PASSED (no alerts)" if not alerts else "COMPLETED WITH ALERTS",
        len(final),
        len(alerts),
    )
    return report
