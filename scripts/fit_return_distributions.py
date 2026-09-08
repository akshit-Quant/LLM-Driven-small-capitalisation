#!/usr/bin/env python3
"""Test whether daily OHLCV returns are better described by heavy-tailed Student-t.

The script calculates adjusted-close log returns, standardizes them *within each
ticker*, pools the standardized returns, and compares a Normal distribution with a
Student-t distribution.  Standardizing per ticker avoids mistaking cross-ticker
volatility differences for heavy tails.

Because fitting parameters on the same sample invalidates a vanilla KS p-value,
the reported p-values use a parametric bootstrap: simulated samples are re-fitted
before their KS statistics are compared with the observed statistic.

Examples:
    uv run python scripts/fit_return_distributions.py
    uv run python scripts/fit_return_distributions.py --ticker AAPL --bootstrap 2_000
    uv run python scripts/fit_return_distributions.py --output benchmark/return_fits.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

Distribution = stats.rv_continuous


def _standardized_log_returns(
    path: Path, tickers: list[str], min_observations: int
) -> tuple[np.ndarray, float, int]:
    """Return standardized returns, raw-return variance, and ticker count."""
    frame = pd.read_parquet(path, columns=["symbol", "date", "adj_close"])
    if tickers:
        wanted = {ticker.upper() for ticker in tickers}
        frame = frame[frame["symbol"].str.upper().isin(wanted)]

    frame = frame.dropna(subset=["symbol", "date", "adj_close"])
    frame = frame[frame["adj_close"] > 0].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.sort_values(["symbol", "date"])
    frame["log_return"] = frame.groupby("symbol", sort=False)["adj_close"].transform(
        lambda price: np.log(price).diff()
    )

    chunks: list[np.ndarray] = []
    raw_chunks: list[np.ndarray] = []
    used_tickers = 0
    for _, group in frame.groupby("symbol", sort=False):
        returns = group["log_return"].dropna().to_numpy(dtype=float)
        if len(returns) < min_observations:
            continue
        scale = returns.std(ddof=1)
        if not np.isfinite(scale) or scale <= 0:
            continue
        raw_chunks.append(returns)
        chunks.append((returns - returns.mean()) / scale)
        used_tickers += 1

    if not chunks:
        raise ValueError(
            "no ticker has enough valid adjusted-close observations; lower "
            "--min-observations or check the input file"
        )
    raw_returns = np.concatenate(raw_chunks)
    return np.concatenate(chunks), float(np.var(raw_returns, ddof=1)), used_tickers


def _fit_and_ks(data: np.ndarray, distribution: Distribution) -> tuple[tuple, float]:
    parameters = distribution.fit(data)
    statistic = float(stats.kstest(data, distribution.cdf, args=parameters).statistic)
    return parameters, statistic


def _bootstrap_ks_pvalue(
    data: np.ndarray,
    distribution: Distribution,
    parameters: tuple,
    observed_statistic: float,
    draws: int,
    rng: np.random.Generator,
) -> float:
    """Parametric-bootstrap KS p-value that accounts for fitted parameters."""
    if draws == 0:
        return float("nan")

    simulated_statistics = np.empty(draws)
    for i in range(draws):
        sample = distribution.rvs(*parameters, size=len(data), random_state=rng)
        _, simulated_statistics[i] = _fit_and_ks(sample, distribution)
    return float(
        (1 + np.count_nonzero(simulated_statistics >= observed_statistic)) / (draws + 1)
    )


def _fit_row(
    name: str,
    distribution: Distribution,
    data: np.ndarray,
    bootstrap_draws: int,
    rng: np.random.Generator,
) -> dict[str, float | str]:
    parameters, ks_statistic = _fit_and_ks(data, distribution)
    data_variance = float(np.var(data, ddof=1))
    row: dict[str, float | str] = {
        "distribution": name,
        "ks_statistic": ks_statistic,
        "ks_bootstrap_pvalue": _bootstrap_ks_pvalue(
            data, distribution, parameters, ks_statistic, bootstrap_draws, rng
        ),
        "data_variance": data_variance,
        "loc": float(parameters[-2]),
        "scale": float(parameters[-1]),
    }
    if name == "student_t":
        degrees_of_freedom = float(parameters[0])
        row["degrees_of_freedom"] = degrees_of_freedom
        row["fitted_variance"] = (
            float(parameters[-1] ** 2 * degrees_of_freedom / (degrees_of_freedom - 2))
            if degrees_of_freedom > 2
            else float("inf")
        )
    else:
        row["fitted_variance"] = float(parameters[-1] ** 2)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/rl2k/ohlcv.parquet"),
        help="OHLCV parquet path",
    )
    parser.add_argument(
        "--ticker",
        action="append",
        default=[],
        help="ticker to include; repeat to select several",
    )
    parser.add_argument("--min-observations", type=int, default=60)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=25_000,
        help="maximum pooled returns to fit; use 0 to use every return",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="parametric-bootstrap draws; use 0 to skip",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, help="optional CSV report path")
    args = parser.parse_args()

    if args.bootstrap < 0 or args.max_samples < 0 or args.min_observations < 3:
        parser.error(
            "--bootstrap and --max-samples must be non-negative; "
            "--min-observations must be at least 3"
        )
    if not args.input.is_file():
        parser.error(f"OHLCV input does not exist: {args.input}")

    returns, raw_return_variance, n_tickers = _standardized_log_returns(
        args.input, args.ticker, args.min_observations
    )
    rng = np.random.default_rng(args.seed)
    available_returns = len(returns)
    if args.max_samples and available_returns > args.max_samples:
        returns = rng.choice(returns, size=args.max_samples, replace=False)
    report = pd.DataFrame(
        [
            _fit_row("normal", stats.norm, returns, args.bootstrap, rng),
            _fit_row("student_t", stats.t, returns, args.bootstrap, rng),
        ]
    )

    print(
        "Pooled standardized daily log returns: "
        f"n={len(returns):,} of {available_returns:,}, tickers={n_tickers:,}"
    )
    print(f"Pooled raw daily-log-return variance: {raw_return_variance:.6g}")
    print(
        f"Empirical excess kurtosis: {stats.kurtosis(returns, fisher=True, bias=False):.3f}"
    )
    print(report.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(
        "\nInterpretation: a smaller KS statistic and a non-rejected bootstrap p-value "
        "for Student-t, especially with finite degrees_of_freedom, support heavy tails."
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
