"""Generate a dependency-free PDF analysis of the Tyche quant strategy."""

from __future__ import annotations

import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "Tyche_Quant_Strategy_Report.pdf"

SECTIONS = [
    ("Tyche Quant Strategy and Backtest Analysis", [
        "Prepared from the current Tyche source tree. This report explains how the research backtest, predictive model, portfolio allocation layer, and autonomous paper executor work together, what they contribute to quantitative finance, and which changes would most improve reliability.",
        "Important scope note: this is a technical and quantitative analysis, not investment advice. The current broker integration is paper-only unless credentials and an explicitly enabled live adapter are configured.",
    ]),
    ("1. Executive summary", [
        "Tyche is a multimodal quantitative research and paper-trading platform. Its research path combines daily market features and clustered news sentiment, trains a distributional neural network, estimates predictive uncertainty, converts forecasts into portfolio weights, and evaluates the result with costs, turnover, exposure, drawdown, and walk-forward metrics.",
        "The autonomous paper path monitors live or fallback news, uses the newest persisted portfolio forecast when it is fresh, falls back to sentiment when no current research artifact is available, sizes each trade against a one-percent risk budget, and applies stop-loss and take-profit rules.",
        "The strongest research decisions are the causal feature construction, chronological purged and embargoed splits, uncertainty-aware forecasts, explicit transaction costs, and exposure reporting. The largest remaining risk is that historical research and live execution still operate on different time scales and data-quality assumptions.",
    ]),
    ("2. What quantitative trading is", [
        "Quantitative trading translates data into explicit, repeatable decisions. A complete quant system has six layers: data ingestion, feature engineering, signal generation, portfolio construction, execution, and evaluation.",
        "The quant advantage is not that a model predicts every price correctly. It is that assumptions become measurable. A researcher can compare a price-only model with a news-only model, measure whether news adds incremental rank information, include realistic costs, and reject a strategy that only works in one historical interval.",
        "A valid quant process separates research from evaluation. The model must not see future prices, future universe membership, or future information timestamps while features, scaling, training, or allocation rules are being selected.",
    ]),
    ("3. End-to-end architecture", [
        "Historical research flow:",
        "1. Load daily OHLCV, news sentiment, and optional macro data.",
        "2. Resolve a tradable universe using symbols supported by both price and news data, with in-sample liquidity selection.",
        "3. Build causal daily features and trailing clustered news features.",
        "4. Align both branches into arrays shaped as assets x trading days x features.",
        "5. Build lookback windows and forward-return targets.",
        "6. Train the multimodal return-distribution model on chronological samples.",
        "7. Generate out-of-sample mean and covariance forecasts with MC dropout.",
        "8. Allocate weights using EW, BL, Bayesian BL, MVO, RP, or HRP.",
        "9. Simulate drift, rebalancing, turnover, transaction costs, and slippage.",
        "10. Report performance, risk, exposure, and walk-forward folds.",
        "Live paper flow:",
        "1. Load the newest predictions_H*.npz artifact if it is within the configured freshness window.",
        "2. Convert predicted return and covariance into a volatility-adjusted signal and MVO portfolio weight.",
        "3. Rank eligible symbols and process multiple symbols per cycle.",
        "4. Size quantity from account equity, risk-per-trade, price, stop distance, and position limits.",
        "5. Fill a paper order, monitor its stop and target, record P&L, and audit the lifecycle.",
    ]),
    ("4. Data and universe selection", [
        "The central alignment code is in tyche/portfolio/data/assemble.py. It creates one fixed asset order and one shared trading-day index. This prevents the daily and news branches from describing different assets or different dates in the same model input.",
        "Universe resolution in tyche/portfolio/data/universe.py keeps names with both prices and news, optionally requires a complete history, applies liquidity filters, and ranks by median dollar volume. Selection can be restricted to data on or before the in-sample boundary, reducing look-ahead and survivorship bias.",
        "The remaining data risk is upstream. If the source dataset already excludes delisted names, contains revised timestamps, or only represents successful companies, Tyche cannot fully remove that bias. A stronger research dataset should preserve historical membership and publication-time availability.",
    ]),
    ("5. Feature engineering", [
        "Daily market features include returns, overnight and intraday movement, range, volatility, momentum, RSI, ATR, moving-average ratios, return z-scores, volume z-scores, relative volume, and liquidity diagnostics. These features describe direction, trend persistence, movement scale, participation, and trading friction.",
        "News features use a trailing calendar window, embedding-based deduplication, representative articles, mean sentiment, and article-count information. Deduplication is important because ten copies of one headline should not automatically count as ten independent signals.",
        "The I-MACD alpha filter attempts to remove market-explained movement. It estimates a market relationship, works with residual movement, and applies direction, purity, and persistence gates. This makes the signal more stock-specific than ordinary price MACD, although it can delay reactions to abrupt events.",
        "Recommended additions are sentiment freshness decay, positive/negative disagreement, event type, surprise, source quality, spread, quote age, and a live version of the market-relative filter.",
    ]),
    ("6. Windowing and leakage control", [
        "The window builder in tyche/portfolio/data/windows.py uses a lookback T and forward holding horizon H. A sample at decision day t contains the synchronized T-day history and targets the forward log return log(price[t+H] / price[t]).",
        "Splits are chronological rather than random. A target horizon is purged when it crosses a split boundary, and an embargo of at least H trading days is applied after boundaries. Standardization is fitted only on days touched by training samples.",
        "These protections materially improve the credibility of results. Without them, overlapping forward targets and future normalization statistics can make a backtest look profitable for the wrong reason.",
    ]),
    ("7. Predictive model and uncertainty", [
        "The model in tyche/portfolio/model/network.py has separate sequence-convolution encoders for daily and news inputs. Their embeddings are fused and passed through an LSTM or attention encoder.",
        "The model produces a predicted mean return for every asset and a positive covariance using a low-rank factor plus diagonal parameterization. This allows the allocator to reason about cross-asset relationships rather than treating every forecast as independent.",
        "At inference, MC dropout produces repeated mean forecasts. The variation between those forecasts is epistemic uncertainty, while the model covariance is aleatoric uncertainty. Their sum is the total predictive covariance used by allocation.",
        "Uncertainty is valuable for position sizing and diversification, but it is only useful when calibrated. Coverage, rank information coefficient, covariance quality, and performance by uncertainty bucket should be monitored rather than assuming a neural variance estimate is automatically reliable.",
    ]),
    ("8. Portfolio allocation", [
        "Equal Weight is the benchmark. MVO uses predicted return and covariance under long-only and per-asset constraints. Risk parity and HRP emphasize balanced risk contributions. Black-Litterman combines a prior with model views; direct BL can produce large gross exposure unless constrained.",
        "The allocator is more appropriate for a portfolio strategy than simply buying the strongest ticker. It can reduce concentration, incorporate covariance, and express a coherent target book. However, forecast error in expected returns can dominate optimization, so equal weight and risk-based baselines are essential.",
        "A production allocator should apply explicit gross leverage, net exposure, short exposure, sector, single-name, turnover, and correlated-risk limits after optimization, not only rely on the optimizer's nominal constraints.",
    ]),
    ("9. Backtest mechanics and costs", [
        "The event-driven engine in tyche/portfolio/allocation/backtest.py rebalances every H trading days. Between rebalances, weights drift with realized returns. At the end of each segment, the engine applies turnover-based transaction costs and slippage.",
        "The engine records daily value, target weights, turnover, gross segment return, net segment return, transaction costs, and slippage. This is materially better than a signal chart that ignores execution friction.",
        "A remaining modelling limitation is that costs are booked at a discrete holding-period boundary. A stronger execution simulator would model bid-ask spreads, order timing, partial fills, market impact, stop gaps, and publication-to-trade latency.",
    ]),
    ("10. Evaluation and what the metrics mean", [
        "Portfolio metrics include cumulative gross and net return, annualized return, volatility, Sharpe, Sortino, Calmar, maximum drawdown, average turnover, total costs, hit rate, gross exposure, net exposure, short exposure, and worst leverage.",
        "Net return must be evaluated alongside turnover and cost drag. Sharpe must be read alongside drawdown, leverage, and sample length. A high return from hidden gross exposure is not equivalent to a fully invested strategy.",
        "Walk-forward folds divide the out-of-sample path into contiguous sections. Stable performance across folds is more informative than one attractive aggregate number. Add confidence intervals, bootstrap analysis, probability of backtest overfitting, deflated Sharpe, and regime breakdowns before trusting a result.",
    ]),
    ("11. Autonomous paper executor", [
        "tyche/execution/paper.py maintains an auditable order lifecycle and persists orders, execution audit records, and learning state. It supports long and short paper trades, multiple symbols per cycle, stale-price duplicate suppression, stop-loss and take-profit exits, and configurable position limits.",
        "The default risk budget is capped at one percent of equity per trade. Quantity is approximately risk budget divided by price multiplied by stop distance. This is preferable to fixed share quantities because the dollar risk scales with price and account size.",
        "The bot's learning layer is a capped per-symbol bias adjustment. It is not full reinforcement learning: it currently records mark-to-market outcomes and nudges future signals. It should only learn from completed round trips with realized P&L, fees, slippage, and a clearly defined trade episode.",
        "The current account must also have portfolio-level controls. Ten trades at one percent risk each can expose ten percent of equity to stop losses, and correlated semiconductor positions can lose together.",
    ]),
    ("12. Impact on quantitative research", [
        "Tyche provides a useful experimental platform for multimodal alpha research. It can test whether news adds information beyond market features, whether residual momentum improves stock selection, whether uncertainty improves allocation, and whether portfolio optimizers survive costs.",
        "Its biggest research contribution is traceability: each layer has a measurable contract. Data becomes aligned arrays, windows become targets, the model becomes predictive moments, moments become weights, and weights become a costed equity curve.",
        "This structure helps prevent a common quant failure: optimizing a model metric while ignoring whether the final portfolio is tradable. A model can improve RMSE while worsening rank IC, turnover, drawdown, or net return. Tyche makes those trade-offs visible.",
    ]),
    ("13. Main weaknesses and risks", [
        "Research/live divergence: the historical model predicts daily forward returns while the live bot can react to intraday news and fallback quotes. A successful backtest does not automatically validate the live sentiment executor.",
        "Data freshness: a last close is not a live quote. New trades should be blocked when quote age, market session, or provider status is unacceptable, while existing positions can still be marked with a stale quote.",
        "P&L accounting: per-order mark-to-market P&L can double-count movements. A proper lot ledger should separate realized P&L, unrealized P&L, fees, slippage, and matched entry/exit trades.",
        "Risk aggregation: per-trade risk is not portfolio risk. Add maximum total open risk, sector limits, correlation limits, gross and net exposure limits, and a daily drawdown kill switch.",
        "Operational reliability: model loading, external providers, and background server lifetime can delay readiness. The UI should expose data source, quote age, artifact age, and provider errors.",
    ]),
    ("14. Highest-impact improvement roadmap", [
        "Phase 1 - correctness: add a lot-based trade ledger; block stale new entries; show target risk versus actual risk; add maximum total open risk; persist signal timestamp, quote timestamp, and artifact timestamp.",
        "Phase 2 - research/live consistency: share feature builders between backtest and live mode; use the same trained model and allocator online; align news publication time with the trading decision; build a live-style simulator.",
        "Phase 3 - signal quality: combine model forecast, sentiment, momentum, volatility, liquidity, market regime, and expected cost. Trade only when expected return exceeds estimated costs and a confidence threshold.",
        "Phase 4 - validation: compare price-only, news-only, combined, and filtered ablations; test multiple holding periods and cost scenarios; evaluate bull, bear, and sideways regimes; report confidence intervals and robustness statistics.",
    ]),
    ("15. Final assessment", [
        "Tyche is best described as a multimodal cross-sectional alpha and portfolio-construction research system with a connected paper execution layer. It is substantially more rigorous than a fixed-rule trading demo because it includes causal features, uncertainty, allocation choices, costs, exposure metrics, and walk-forward evaluation.",
        "The strategy should not yet be judged by live paper P&L alone. The next meaningful milestone is a reproducible experiment showing that the same signal and allocation path performs consistently out of sample, after costs, under realistic execution assumptions and portfolio-level risk constraints.",
        "If those tests remain positive, Tyche will have a stronger claim as a quantitative research framework. Until then, treat live results as engineering validation and hypothesis generation, not evidence of a durable trading edge.",
    ]),
]


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pages() -> list[list[tuple[str, int]]]:
    pages: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    lines_left = 46
    for heading, paragraphs in SECTIONS:
        needed = 2
        for paragraph in paragraphs:
            needed += len(textwrap.wrap(paragraph, 96)) + 1
        if current and needed > lines_left:
            pages.append(current)
            current, lines_left = [], 46
        current.append((heading, 16))
        lines_left -= 2
        for paragraph in paragraphs:
            wrapped = textwrap.wrap(paragraph, 96) or [""]
            for line in wrapped:
                current.append((line, 10))
                lines_left -= 1
            current.append(("", 10))
            lines_left -= 1
    if current:
        pages.append(current)
    return pages


def write_pdf() -> None:
    pages = build_pages()
    objects: list[bytes] = []

    def add(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    catalog_id = add(b"")
    pages_id = add(b"")
    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []

    for page_lines in pages:
        commands = ["BT", "/F1 16 Tf", "50 760 Td"]
        first = True
        for line, size in page_lines:
            if first:
                first = False
            else:
                commands.append("0 -15 Td" if size == 10 else "0 -19 Td")
            commands.append(f"/F1 {size} Tf ({pdf_escape(line)}) Tj")
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1", "replace")
        content_id = add(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
        page_ids.append(add(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents "
            + str(content_id).encode() + b" 0 R >>"
        ))

    kids = b" ".join(str(pid).encode() + b" 0 R" for pid in page_ids)
    objects[pages_id - 1] = (
        b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(len(page_ids)).encode() + b" >>"
    )
    objects[catalog_id - 1] = b"<< /Type /Catalog /Pages 2 0 R >>"

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode())
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode()
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(output)
    print(f"wrote {OUT} ({len(output)} bytes, {len(pages)} pages)")


if __name__ == "__main__":
    write_pdf()
