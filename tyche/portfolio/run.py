"""Portfolio backtest runner for EW, BL, Bayesian_BL, MVO, RP, and HRP.

The runner prepares the aligned data and chronological split, trains the return
distribution model for each holding period, predicts mean/covariance forecasts for the
out-of-sample rebalance dates, then passes those forecasts into the allocation models.
Transaction-cost scenarios reuse the same predictions for a holding period.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from tyche.common.logging import get_logger
from tyche.portfolio.allocation.backtest import run_backtest
from tyche.portfolio.allocation.mask import log_mask_coverage, masked_strategy
from tyche.portfolio.allocation.moments import log_to_simple
from tyche.portfolio.allocation.strategies import bayesian_bl, bl, ew, hrp, mvo, rp
from tyche.portfolio.config import Config, default_config
from tyche.portfolio.data.assemble import AlignedData, assemble
from tyche.portfolio.data.preprocessing import apply_standardizer, fit_standardizer
from tyche.portfolio.data.windows import (
    Sample,
    SplitIndex,
    build_splits,
    train_day_indices,
)
from tyche.portfolio.evaluation.model_metrics import evaluate_model
from tyche.portfolio.evaluation.portfolio_metrics import (
    evaluate_portfolio,
    walk_forward_metrics,
)
from tyche.portfolio.model.predict import Predictions, predict
from tyche.portfolio.model.train import train_model

log = get_logger(__name__)
DEFAULT_HOLDINGS: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 40, 60)


MomentForecasts = dict[int, tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class PreparedExperiment:
    data: AlignedData
    splits: SplitIndex
    oos: list[Sample]
    predictions: Predictions
    forecasts: MomentForecasts
    oos_label: str


def _regularize_covariance(cov: np.ndarray, eps: float) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    min_eval = float(np.linalg.eigvalsh(cov).min())
    if min_eval < eps:
        cov = cov + (eps - min_eval) * np.eye(cov.shape[0])
    return cov


def _build_forecasts(predictions: Predictions, cfg: Config) -> MomentForecasts:
    forecasts: MomentForecasts = {}
    for t, mu, cov in zip(
        predictions.decision_t, predictions.mu, predictions.cov, strict=True
    ):
        alloc_mu, alloc_cov = np.asarray(mu, dtype=float), np.asarray(cov, dtype=float)
        if cfg.portfolio.convert_to_simple_returns:
            alloc_mu, alloc_cov = log_to_simple(alloc_mu, alloc_cov)
        forecasts[int(t)] = (
            alloc_mu,
            _regularize_covariance(alloc_cov, cfg.model.cov_eps),
        )
    return forecasts


def prepare(cfg: Config) -> PreparedExperiment:
    raw = assemble(cfg)
    splits = build_splits(raw, cfg)
    oos = splits.test if splits.test else splits.val
    oos_label = "test" if splits.test else "val (no test samples)"
    if not splits.train:
        raise RuntimeError("no training samples available for the configured split")
    if not oos:
        raise RuntimeError("no out-of-sample samples available for portfolio backtest")

    standardizer = fit_standardizer(raw, train_day_indices(splits, cfg))
    data = apply_standardizer(raw, standardizer)

    log.info(
        "universe=%s | train=%d val=%d oos=%d [%s] | H=%d T=%d | "
        "target_distribution=%s",
        data.assets,
        len(splits.train),
        len(splits.val),
        len(oos),
        oos_label,
        cfg.window.holding,
        cfg.window.lookback,
        cfg.train.target_distribution,
    )

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    train_result = train_model(
        data,
        splits.train,
        splits.val,
        cfg,
        tag=f"H{cfg.window.holding}",
    )
    predictions = predict(
        train_result.model,
        data,
        oos,
        batch_size=cfg.train.batch_size,
        mc_dropout_samples=cfg.train.mc_dropout_samples,
        device=cfg.train.device,
    )
    pred_path = cfg.artifacts_dir / f"predictions_H{cfg.window.holding}.npz"
    predictions.save(pred_path)

    model_metrics = evaluate_model(
        predictions,
        cfg.train.target_distribution,
        cfg.train.student_t_df,
    )
    pd.DataFrame([model_metrics]).to_csv(
        cfg.artifacts_dir / f"model_metrics_H{cfg.window.holding}.csv",
        index=False,
    )
    _log_table("Predictive model metrics", pd.DataFrame([model_metrics]))
    log.info(
        "saved %d prediction rows for H=%d to %s",
        len(predictions.decision_t),
        cfg.window.holding,
        pred_path,
    )

    return PreparedExperiment(
        data=data,
        splits=splits,
        oos=oos,
        predictions=predictions,
        forecasts=_build_forecasts(predictions, cfg),
        oos_label=oos_label,
    )


def _rebalance_days(oos, cfg: Config) -> list[int]:
    ts = sorted(s.t for s in oos)
    return ts[:: cfg.window.holding]


def _backtest(data, days, rebal_t, strategy, cfg: Config):
    net = run_backtest(data.adj_close, days, rebal_t, strategy, cfg)
    zero_cost = replace(
        cfg,
        portfolio=replace(cfg.portfolio, transaction_cost_bps=0.0, slippage_bps=0.0),
    )
    gross = run_backtest(data.adj_close, days, rebal_t, strategy, zero_cost)
    metrics = evaluate_portfolio(net, gross.value)
    metrics["walk_forward_folds"] = int(cfg.portfolio.walk_forward_folds)
    return net, metrics, walk_forward_metrics(
        net, gross.value, cfg.portfolio.walk_forward_folds
    )


def _run_portfolios(prepared: PreparedExperiment, cfg: Config) -> dict:
    data = prepared.data
    log.info(
        "portfolio backtest | H=%d | cost_model=%s | transaction_cost_bps=%.4f "
        "| stock_filter=%s",
        cfg.window.holding,
        cfg.portfolio.cost_model,
        cfg.portfolio.transaction_cost_bps,
        cfg.filter_indicator if cfg.filter_enabled else "off",
    )

    rebal_t = _rebalance_days(prepared.oos, cfg)
    strategies = {
        "EW": ew(data.adj_close),
        "BL": bl(prepared.forecasts, cfg),
        "Bayesian_BL": bayesian_bl(prepared.forecasts, cfg),
        "MVO": mvo(prepared.forecasts, cfg),
        "RP": rp(prepared.forecasts, cfg),
        "HRP": hrp(prepared.forecasts, cfg),
    }

    # Stock filter: the model forecasts returns/covariances from OHLCV + news, then
    # the allocator sizes only names whose signal passes the gate. Applying the mask
    # after forecasting keeps the model's N x N covariance shape fixed while letting
    # the filter decide tradable membership at each rebalance.
    if cfg.filter_enabled:
        if data.alpha_signal is None:
            raise RuntimeError(
                "the stock filter is enabled but the aligned data carries no "
                "alpha_signal — rebuild it through data.assemble.assemble"
            )
        log_mask_coverage(data.alpha_signal, rebal_t, cfg.alpha_filter)
        strategies = {
            name: masked_strategy(strat, data.alpha_signal, cfg.alpha_filter)
            for name, strat in strategies.items()
        }

    port_metrics, curves, walk_forward = {}, {}, {}
    for name, strat in strategies.items():
        net, metrics, folds = _backtest(data, data.days, rebal_t, strat, cfg)
        port_metrics[name] = metrics
        walk_forward[name] = folds
        curves[name] = pd.Series(net.value, index=net.dates)

    return {
        "portfolio_metrics": port_metrics,
        "curves": curves,
        "walk_forward": walk_forward,
    }


def run_experiment(cfg: Config) -> dict:
    return _run_portfolios(prepare(cfg), cfg)


def config_for_holding(cfg: Config, holding: int) -> Config:
    """Return a config for one holding period with a matching split embargo."""
    return replace(
        cfg,
        window=replace(
            cfg.window, holding=holding, embargo=max(cfg.window.embargo, holding)
        ),
    )


def config_for_transaction_cost(cfg: Config, cost_bps: float) -> Config:
    """Return a config with only the explicit transaction-cost component changed."""
    return replace(
        cfg,
        portfolio=replace(cfg.portfolio, transaction_cost_bps=float(cost_bps)),
    )


def _cost_label(cost_bps: float) -> str:
    return f"{cost_bps:g}".replace(".", "p")


def _tidy(
    metrics: dict[str, dict[str, float]], holding: int, cost_bps: float | None = None
) -> pd.DataFrame:
    df = pd.DataFrame(metrics).T
    df.index.name = "model"
    df = df.reset_index().assign(holding=holding)
    index_cols = ["holding", "model"]
    if cost_bps is not None:
        df = df.assign(transaction_cost_bps=float(cost_bps))
        index_cols = ["transaction_cost_bps", *index_cols]
    return df.set_index(index_cols)


def run_grid(
    cfg: Config,
    holdings: tuple[int, ...] = DEFAULT_HOLDINGS,
    transaction_cost_bps: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Run the requested holding/cost grid and write portfolio artifacts."""
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    failed: dict[tuple[float, int], str] = {}
    costs = transaction_cost_bps or (cfg.portfolio.transaction_cost_bps,)
    multi_cost = len(costs) > 1

    for holding in holdings:
        holding_cfg = config_for_holding(cfg, holding)
        try:
            prepared = prepare(holding_cfg)
        except Exception as exc:
            for cost_bps in costs:
                failed[(float(cost_bps), holding)] = str(exc)
            log.exception("failed model training/prediction for H=%d", holding)
            continue

        for cost_bps in costs:
            cost_cfg = config_for_transaction_cost(holding_cfg, cost_bps)
            log.info("transaction cost %g bps | H=%d", cost_bps, holding)
            try:
                results = _run_portfolios(prepared, cost_cfg)
            except Exception as exc:
                failed[(float(cost_bps), holding)] = str(exc)
                log.exception("failed cost=%g H=%d", cost_bps, holding)
                continue

            portfolio = _tidy(
                results["portfolio_metrics"],
                holding,
                cost_bps if multi_cost else None,
            )
            frames.append(portfolio)

            suffix = (
                f"_C{_cost_label(cost_bps)}_H{holding}"
                if multi_cost
                else f"_H{holding}"
            )
            portfolio.to_csv(cfg.artifacts_dir / f"portfolio_metrics{suffix}.csv")
            pd.DataFrame(results["curves"]).to_csv(
                cfg.artifacts_dir / f"equity_curves{suffix}.csv"
            )
            wf_rows = [
                {"model": name, **row}
                for name, rows in results["walk_forward"].items()
                for row in rows
            ]
            if wf_rows:
                pd.DataFrame(wf_rows).to_csv(
                    cfg.artifacts_dir / f"walk_forward_metrics{suffix}.csv",
                    index=False,
                )

    if not frames:
        raise RuntimeError(f"every portfolio run failed: {failed}")

    combined = pd.concat(frames).sort_index()
    combined_name = (
        "cost_portfolio_metrics.csv" if multi_cost else "portfolio_metrics.csv"
    )
    combined.to_csv(cfg.artifacts_dir / combined_name)
    if failed:
        log.warning("failed runs: %s", sorted(failed))
    return combined


def _format_df(df: pd.DataFrame, width: int) -> str:
    with pd.option_context(
        "display.width",
        width,
        "display.max_columns",
        None,
        "display.float_format",
        lambda x: f"{x:.4f}",
    ):
        return df.to_string()


def _log_table(title: str, df: pd.DataFrame) -> None:
    log.info("%s\n%s", title, _format_df(df, 160))


def _print_pivot(df: pd.DataFrame, metric: str) -> None:
    if metric not in df.columns:
        return
    table = df[metric].unstack("model")
    log.info("%s by holding period\n%s", metric, _format_df(table, 200))


def main() -> None:
    ap = argparse.ArgumentParser(description="Multimodal portfolio pipeline")
    ap.add_argument("--holding", type=int, default=None, help="single H")
    ap.add_argument(
        "--holdings",
        type=int,
        nargs="+",
        default=None,
        help="holding periods to run (default: 1 2 3 5 10 20 40 60)",
    )
    ap.add_argument("--lookback", type=int, default=None)
    ap.add_argument(
        "--transaction-cost-bps",
        type=float,
        nargs="+",
        default=None,
        help="transaction-cost scenarios in bps (slippage is left unchanged)",
    )
    args = ap.parse_args()

    cfg = default_config()
    if args.lookback is not None:
        cfg = replace(cfg, window=replace(cfg.window, lookback=args.lookback))
    if args.holding is not None and args.holdings is not None:
        raise SystemExit("use either --holding or --holdings, not both")

    if args.holding is not None:
        cfg = config_for_holding(cfg, args.holding)
        if args.transaction_cost_bps:
            if len(args.transaction_cost_bps) != 1:
                raise SystemExit("--holding accepts at most one transaction cost")
            cfg = config_for_transaction_cost(cfg, args.transaction_cost_bps[0])
        results = run_experiment(cfg)
        _log_table("Portfolio backtest", pd.DataFrame(results["portfolio_metrics"]).T)
        return

    holdings = tuple(args.holdings) if args.holdings else DEFAULT_HOLDINGS
    combined = run_grid(cfg, holdings, args.transaction_cost_bps)
    for metric in ("cum_return_net", "sharpe", "max_drawdown", "avg_turnover"):
        _print_pivot(combined, metric)
    log.info("artifacts written to %s", cfg.artifacts_dir)


if __name__ == "__main__":
    main()
