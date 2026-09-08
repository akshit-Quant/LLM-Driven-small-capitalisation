"""News feature branch — exact-window, centroid-representative story sentiment.

For each ``(stock, trading-day)`` this looks back over a configurable one-month
selection window, builds a cosine-similarity graph over local text embeddings, and
treats every connected component as one news story. Each component is represented by
the article closest to its centroid. The cosine graph is evaluated on CUDA or MPS when
available, while the selection window itself is never partitioned: boundary articles
cannot be omitted because a cluster was fitted on an incomplete time bucket. Daily
features then use only those representative article scores: the mean representative
sentiment and the log number of unique story components in the trailing selection
window.

Every feature is built from articles mapped to the prior 30 calendar days; the
decision day's articles are deliberately excluded. Days with no news are zeros —
never forward-filled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from tyche.common.device import resolve_device
from tyche.common.logging import get_logger
from tyche.news.service.embedder import embed_texts
from tyche.portfolio.config import Config

log = get_logger(__name__)

NEWS_FEATURES: list[str] = [
    "mean_sent",
    "log_n_articles",
]


def _unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x


class _DisjointSet:
    """Small union-find used after GPU cosine-neighbor discovery."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _cluster_labels(
    embeddings: np.ndarray,
    similarity_threshold: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Cluster one complete selection window from its cosine-neighbor graph.

    This is single-linkage clustering at ``similarity_threshold``. Unlike fitting
    independent calendar buckets, every article currently in the trailing window is
    considered together. GPU work is limited to dense, batched cosine products.
    """
    size = len(embeddings)
    if size <= 1:
        return np.zeros(size, dtype=int)

    vectors = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
    sets = _DisjointSet(size)
    batch_size = max(1, int(batch_size))
    threshold = float(similarity_threshold)

    with torch.inference_mode():
        for start in range(0, size, batch_size):
            stop = min(start + batch_size, size)
            similarities = vectors[start:stop] @ vectors.T
            # Retain one orientation of every edge and remove self-links.
            row_ids, col_ids = torch.where(similarities >= threshold)
            row_ids = row_ids + start
            keep = row_ids < col_ids
            if not bool(keep.any()):
                continue
            pairs = torch.stack((row_ids[keep], col_ids[keep]), dim=1).cpu().numpy()
            for left, right in pairs:
                sets.union(int(left), int(right))

    roots = [sets.find(i) for i in range(size)]
    label_map: dict[int, int] = {}
    return np.asarray(
        [label_map.setdefault(root, len(label_map)) for root in roots], dtype=int
    )


def _centroid_representatives(embeddings: np.ndarray, labels: np.ndarray) -> list[int]:
    representatives: list[int] = []
    for label in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == label)
        cluster = embeddings[members]
        centroid = _unit(cluster.mean(axis=0))
        scores = cluster @ centroid
        representatives.append(int(members[int(np.argmax(scores))]))
    return representatives


def _representative_indices(
    group: pd.DataFrame,
    embedding_by_text: dict[str, np.ndarray],
    similarity_threshold: float,
    device: torch.device,
    batch_size: int,
) -> list[int]:
    summary = group["summary_text"].fillna("").astype(str).str.strip()
    embed_rows = [idx for idx, text in summary.items() if text in embedding_by_text]
    keep_rows = [idx for idx, text in summary.items() if text not in embedding_by_text]
    if len(embed_rows) <= 1:
        return [*keep_rows, *embed_rows]

    embeddings = np.vstack(
        [embedding_by_text[str(summary.loc[idx])] for idx in embed_rows]
    )
    labels = _cluster_labels(embeddings, similarity_threshold, device, batch_size)
    reps = _centroid_representatives(embeddings, labels)
    return [*keep_rows, *[embed_rows[i] for i in reps]]


def _load_embedding_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    try:
        cache = np.load(path, allow_pickle=True)
        texts = [str(t) for t in cache["texts"].tolist()]
        embeddings = cache["embeddings"].astype(np.float32)
    except Exception as exc:  # noqa: BLE001 - a corrupt or truncated cache can fail
        # in numpy, pickle or codec territory; every one of them is recoverable by
        # rebuilding the cache from scratch.
        log.warning("could not load news embedding cache %s: %s", path, exc)
        return {}
    if len(texts) != len(embeddings):
        log.warning(
            "ignoring invalid news embedding cache %s: %d texts for %d embeddings",
            path,
            len(texts),
            len(embeddings),
        )
        return {}
    return {text: embeddings[i] for i, text in enumerate(texts)}


def _save_embedding_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    texts = list(cache)
    embeddings = np.vstack([cache[text] for text in texts]).astype(np.float32)
    np.savez_compressed(
        path,
        texts=np.asarray(texts, dtype=object),
        embeddings=embeddings,
    )


def _embedding_lookup(df: pd.DataFrame, cfg: Config) -> dict[str, np.ndarray]:
    text = df["summary_text"].fillna("").astype(str).str.strip()
    unique_texts = list(dict.fromkeys(t for t in text if t))
    if not unique_texts:
        return {}

    cache_path = cfg.paths.news_embedding_cache
    cache = _load_embedding_cache(Path(cache_path))
    missing = [text for text in unique_texts if text not in cache]
    log.info(
        "portfolio news feature extraction found %d unique summaries from %d "
        "article rows | cached=%d missing=%d",
        len(unique_texts),
        len(df),
        len(unique_texts) - len(missing),
        len(missing),
    )
    if missing:
        log.info("embedding %d missing portfolio news summaries", len(missing))
        embeddings = embed_texts(missing)
        for i, text in enumerate(missing):
            cache[text] = embeddings[i].astype(np.float32)
        _save_embedding_cache(Path(cache_path), cache)
        log.info("saved news embedding cache to %s", cache_path)
    return {text: cache[text] for text in unique_texts if text in cache}


def _representative_window_features(
    window: pd.DataFrame,
    embedding_by_text: dict[str, np.ndarray],
    cfg: Config,
    device: torch.device,
) -> tuple[float, int]:
    keep = _representative_indices(
        window,
        embedding_by_text,
        cfg.news.dedup_similarity_threshold,
        device,
        cfg.news.dedup_similarity_batch_size,
    )
    scores = window.loc[keep, "sentiment_final"].astype(float)
    return float(scores.mean()), len(scores)


def _effective_publication_day(ts: pd.Series, cfg: Config) -> pd.DatetimeIndex:
    """The calendar day from which an article is tradable.

    Articles published after the session close cannot inform a decision taken at
    that close, so they are pushed to the next calendar day before being snapped
    onto the trading calendar. ``news_cutoff_utc_hour`` is the close expressed in
    UTC (16:00 America/New_York is 20:00 UTC in winter, 21:00 in summer; the fixed
    default of 20:00 is the conservative choice, since being an hour early only ever
    delays an article by one session — it can never pull one forward).
    """
    stamps = pd.to_datetime(ts, utc=True)
    cutoff = float(cfg.news.cutoff_utc_hour)
    hour = stamps.dt.hour + stamps.dt.minute / 60.0 + stamps.dt.second / 3600.0
    day = stamps.dt.normalize()
    after_close = hour > cutoff
    day = day + pd.to_timedelta(after_close.astype(int), unit="D")
    return pd.DatetimeIndex(day)


def _empty_aggregate(trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """An empty aggregate whose ``date`` dtype matches the trading-day index.

    ``pd.DataFrame(columns=[...])`` would give every column ``object`` dtype, and
    merging an object ``date`` against the tz-aware datetimes in the dense grid
    raises rather than yielding an empty join. Typing the columns here keeps the
    no-news case on the normal path, where it lands as all-zero news features.
    """
    return pd.DataFrame(
        {
            "asset": pd.Series(dtype="object"),
            "date": pd.Series(dtype=trading_days.dtype),
            "mean_sent": pd.Series(dtype="float64"),
            "n_articles": pd.Series(dtype="int64"),
        }
    )


def _aggregate_selection_windows(
    df: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    cfg: Config,
) -> pd.DataFrame:
    """Aggregate representative stories from the trailing selection window."""
    if df.empty:
        return _empty_aggregate(trading_days)

    embedding_by_text = _embedding_lookup(df, cfg) if cfg.news.dedup_enabled else {}
    lookback = pd.Timedelta(days=int(cfg.news.dedup_lookback_days))
    device = resolve_device(cfg.news.dedup_device) if cfg.news.dedup_enabled else None
    if cfg.news.dedup_enabled:
        log.info(
            "clustering complete rolling news windows on device=%s "
            "(similarity batch size=%d)",
            device,
            cfg.news.dedup_similarity_batch_size,
        )
    rows: list[dict] = []
    total_representatives = 0
    grouped = list(df.groupby("asset", sort=True))
    total_windows = len(grouped) * len(trading_days)

    with tqdm(
        total=total_windows,
        desc="news feature windows",
        unit="window",
    ) as pbar:
        for asset, asset_df in grouped:
            asset_df = asset_df.sort_values(["date", "ts"])
            for date in trading_days:
                pbar.update(1)
                pbar.set_postfix_str(f"asset={asset}")
                start = date - lookback
                # Strictly lag the branch: at decision day t, only use effective
                # news dates t-lookback through t-1.  In particular, no score from
                # t can enter the model, even if upstream same-day normalization
                # touched a story published after the market close.
                window = asset_df[
                    (asset_df["date"] >= start) & (asset_df["date"] < date)
                ]
                if window.empty:
                    continue
                if cfg.news.dedup_enabled:
                    mean_sent, n_articles = _representative_window_features(
                        window,
                        embedding_by_text,
                        cfg,
                        device,
                    )
                else:
                    scores = window["sentiment_final"].astype(float)
                    mean_sent, n_articles = float(scores.mean()), len(scores)
                total_representatives += n_articles
                rows.append(
                    {
                        "asset": asset,
                        "date": date,
                        "mean_sent": mean_sent,
                        "n_articles": n_articles,
                    }
                )

    selection_mode = (
        "centroid representatives" if cfg.news.dedup_enabled else "articles"
    )
    log.info(
        "news selection used %d %s across %d strictly lagged asset-day windows "
        "from %d articles (lookback=%d days)",
        total_representatives,
        selection_mode,
        len(rows),
        len(df),
        cfg.news.dedup_lookback_days,
    )
    if not rows:
        # Every selection window came back empty; same dtype care as above.
        return _empty_aggregate(trading_days)
    return pd.DataFrame(rows, columns=["asset", "date", "mean_sent", "n_articles"])


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (asset, date) summarizing representative story articles."""
    grouped = df.groupby(["asset", "date"])
    agg = grouped.agg(
        mean_sent=("sentiment_final", "mean"),
        n_articles=("sentiment_final", "size"),
    ).reset_index()
    return agg


def build_news_features(
    news: pd.DataFrame, trading_days: pd.DatetimeIndex, cfg: Config
) -> pd.DataFrame:
    """Return a dense long frame ``[asset, date, *NEWS_FEATURES]`` covering every
    (asset, trading_day) — including no-news days.

    An article is attributed to the first trading session that could actually have
    *acted* on it. Decisions are taken at the close, so an article published at or
    before that session's close lands on that day; one published after it — an
    evening wire, an overnight release, anything on a weekend or holiday — rolls
    forward to the next session. The feature window then excludes that landing day,
    using only the preceding calendar-month window. Nothing is dropped, and nothing
    leaks backward.

    The cutoff matters: simply truncating the timestamp to its calendar date, as an
    earlier version did, attributes a 9pm earnings release to a session that closed
    five hours earlier. That is a genuine look-ahead — the model trains on
    information the strategy could not have traded — and on this feed it affects
    roughly one article in seven."""
    df = news.copy()
    if "summary_text" not in df.columns:
        df["summary_text"] = ""
    if df.empty:
        # Not fatal — the branch degrades to all-zero news features — but it is
        # almost always a universe/date mismatch between the sentiment file and the
        # price data rather than a genuinely news-free period, and silently training
        # on a dead branch is worse than a noisy log.
        log.warning(
            "no news rows for the resolved universe — every news feature will be "
            "zero. Check that the sentiment file's symbols overlap the price file's "
            "and that its publication dates overlap the trading calendar."
        )
    cal_day = _effective_publication_day(df["ts"], cfg)
    # Snap each article to the first trading day >= the day it was actionable.
    pos = trading_days.searchsorted(cal_day, side="left")
    valid = pos < len(trading_days)
    if len(df) and not valid.any():
        log.warning(
            "all %d news rows fall after the last trading day (%s) — every news "
            "feature will be zero",
            len(df),
            trading_days[-1].date() if len(trading_days) else "n/a",
        )
    df = df[valid].copy()
    df["date"] = trading_days[pos[valid]]

    # Dense grid over (assets seen in news) x trading_days, so no-news days exist.
    assets = sorted(df["asset"].unique())
    grid = pd.MultiIndex.from_product(
        [assets, trading_days], names=["asset", "date"]
    ).to_frame(index=False)
    agg = _aggregate_selection_windows(df, trading_days, cfg)
    out = grid.merge(agg, on=["asset", "date"], how="left")

    for col in ("mean_sent", "n_articles"):
        out[col] = out[col].fillna(0.0)

    # Story counts are heavily right-skewed, so they enter on a log scale.
    out["log_n_articles"] = np.log1p(out["n_articles"])

    return out[["asset", "date", *NEWS_FEATURES]]
