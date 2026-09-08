"""Train the predictive model; select the checkpoint by validation NLL.

Pure model training — no portfolio objective ever enters here (per the spec, the
network optimizes only the return-distribution likelihood). Returns the best model
(lowest validation NLL) plus the per-epoch history.

Each epoch logs the NLL alongside diagnostics of the two predicted
objects the allocator actually consumes:

* **mean** — its level, and its *cross-sectional dispersion*. Dispersion is the one
  that matters: the allocator only ever sees relative views, and the reported IC and
  rank IC are cross-sectional. A model whose ``mu`` collapses to the same value for
  every asset can still post a falling NLL while carrying no allocation signal at all,
  and that failure is invisible in the loss.
* **aleatoric covariance** — mean predicted variance, mean off-diagonal correlation, and the
  spread between the largest and smallest variance. Together these catch the covariance
  head drifting toward a degenerate solution, which shows up downstream as unstable
  ``Sigma^-1 mu`` in the direct-BL allocation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from tyche.common.logging import get_logger
from tyche.portfolio.config import Config
from tyche.portfolio.data.assemble import AlignedData
from tyche.portfolio.data.windows import Sample
from tyche.portfolio.model.dataset import WindowDataset
from tyche.portfolio.model.losses import distribution_nll
from tyche.portfolio.model.network import MultimodalReturnModel, Prediction

log = get_logger(__name__)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainResult:
    model: MultimodalReturnModel
    best_val_nll: float
    history: list[dict]


def build_model(data: AlignedData, cfg: Config, use_news=True):
    return MultimodalReturnModel(
        n_assets=data.n_assets,
        daily_features=len(data.daily_names),
        news_features=len(data.news_names),
        cfg=cfg.model,
        use_news=use_news,
    )


def _move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def moment_stats(pred: Prediction) -> dict[str, float]:
    """Diagnostics of the predicted mean and aleatoric covariance for one batch."""
    mu, cov = pred.mu, pred.aleatoric_cov
    n = mu.shape[-1]
    var = torch.diagonal(cov, dim1=-2, dim2=-1)
    sd = var.clamp_min(1e-12).sqrt()
    corr = cov / (sd.unsqueeze(-1) * sd.unsqueeze(-2))
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=mu.device)
    return {
        "mu_mean": mu.mean().item(),
        # Std across assets within a date — the signal the allocator can act on.
        "mu_dispersion": mu.std(dim=-1).mean().item(),
        "cov_var": var.mean().item(),
        "cov_corr": corr[..., off_diagonal].mean().item(),
        "cov_var_spread": (var.max(dim=-1).values / var.min(dim=-1).values)
        .mean()
        .item(),
    }


def _accumulate(running: dict[str, float], parts: dict[str, float]) -> None:
    for k, v in parts.items():
        running[k] = running.get(k, 0.0) + v


@torch.no_grad()
def _evaluate(model, loader, device, cfg: Config) -> tuple[float, dict[str, float]]:
    """Validation NLL plus the same moment diagnostics, sample-weighted."""
    model.eval()
    total, count = 0.0, 0
    running: dict[str, float] = {}
    for batch in loader:
        batch = _move(batch, device)
        pred = model(batch["daily"], batch["news"])
        size = len(batch["target"])
        total += (
            distribution_nll(
                pred,
                batch["target"],
                cfg.train.target_distribution,
                cfg.train.student_t_df,
            ).item()
            * size
        )
        _accumulate(running, {k: v * size for k, v in moment_stats(pred).items()})
        count += size
    denom = max(count, 1)
    return total / denom, {k: v / denom for k, v in running.items()}


def _log_epoch(tag: str, epoch: int, record: dict) -> None:
    log.info(
        "[%s] epoch %02d | val_nll %+.4f | train nll %+.4f | "
        "mu mean %+.5f disp %.5f | cov var %.5f corr %+.3f spread %.1f",
        tag,
        epoch,
        record["val_nll"],
        record["nll"],
        record["mu_mean"],
        record["mu_dispersion"],
        record["cov_var"],
        record["cov_corr"],
        record["cov_var_spread"],
    )


def train_model(
    data: AlignedData,
    train_samples: list[Sample],
    val_samples: list[Sample],
    cfg: Config,
    use_news=True,
    tag: str = "model",
) -> TrainResult:
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    model = build_model(data, cfg, use_news).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    train_loader = DataLoader(
        WindowDataset(data, train_samples),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        WindowDataset(data, val_samples), batch_size=cfg.train.batch_size
    )

    log.info(
        "[%s] training on %d samples (val %d) | device=%s | params=%d | "
        "target_distribution=%s df=%.2f",
        tag,
        len(train_samples),
        len(val_samples),
        device,
        sum(p.numel() for p in model.parameters()),
        cfg.train.target_distribution,
        cfg.train.student_t_df,
    )

    best_state, best_nll, history, stale = None, float("inf"), [], 0
    for epoch in range(cfg.train.epochs):
        model.train()
        running: dict[str, float] = {}
        for batch in train_loader:
            batch = _move(batch, device)
            opt.zero_grad()
            pred = model(batch["daily"], batch["news"])
            loss = distribution_nll(
                pred,
                batch["target"],
                cfg.train.target_distribution,
                cfg.train.student_t_df,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            _accumulate(running, {"nll": loss.item()})
            _accumulate(running, moment_stats(pred))

        n_batches = max(len(train_loader), 1)
        record = {"epoch": epoch, **{k: v / n_batches for k, v in running.items()}}

        if val_samples:
            val_nll, val_moments = _evaluate(model, val_loader, device, cfg)
            record["val_nll"] = val_nll
            record.update({f"val_{k}": v for k, v in val_moments.items()})
        else:
            record["val_nll"] = float("nan")

        history.append(record)
        _log_epoch(tag, epoch, record)

        if val_samples and record["val_nll"] < best_nll - 1e-4:
            best_nll, best_state, stale = record["val_nll"], _clone(model), 0
        else:
            stale += 1
            if stale >= cfg.train.patience:
                log.info(
                    "[%s] early stop at epoch %d (best val_nll %+.4f)",
                    tag,
                    epoch,
                    best_nll,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    return TrainResult(model=model, best_val_nll=best_nll, history=history)


def _clone(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
