"""Generate and persist out-of-sample predictions for every rebalance date.

For each sample the model emits a MC-dropout predictive mean, aleatoric covariance,
epistemic covariance, and their sum (the total predictive covariance). The total
moments are the interface between the predictive and portfolio-construction stages;
all are saved to ``.npz`` so portfolio tuning never re-runs the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tyche.common.device import resolve_device
from tyche.common.logging import get_logger
from tyche.portfolio.data.assemble import AlignedData
from tyche.portfolio.data.windows import Sample
from tyche.portfolio.model.dataset import WindowDataset
from tyche.portfolio.model.network import MultimodalReturnModel

log = get_logger(__name__)


@dataclass
class Predictions:
    decision_t: np.ndarray  # [S] trading-day index of each rebalance decision
    mu: np.ndarray  # [S, N]
    cov: np.ndarray  # [S, N, N] total predictive covariance = aleatoric + epistemic
    aleatoric_cov: np.ndarray  # [S, N, N]
    epistemic_cov: np.ndarray  # [S, N, N]
    target: np.ndarray  # [S, N] realized forward H-day return (for eval)
    assets: list[str]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            decision_t=self.decision_t,
            mu=self.mu,
            cov=self.cov,
            aleatoric_cov=self.aleatoric_cov,
            epistemic_cov=self.epistemic_cov,
            target=self.target,
            assets=np.array(self.assets),
        )

    @staticmethod
    def load(path: Path) -> Predictions:
        z = np.load(path, allow_pickle=True)
        cov = z["cov"]
        return Predictions(
            decision_t=z["decision_t"],
            mu=z["mu"],
            cov=cov,
            aleatoric_cov=(z.get("aleatoric_cov", cov)),
            epistemic_cov=(
                z["epistemic_cov"] if "epistemic_cov" in z else np.zeros_like(cov)
            ),
            target=z["target"],
            assets=list(z["assets"]),
        )


def _enable_mc_dropout(model: nn.Module) -> None:
    """Activate dropout while preserving eval-mode BatchNorm statistics."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def _mc_predict_batch(
    model: MultimodalReturnModel,
    batch: dict[str, torch.Tensor],
    samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the law of total covariance to one batch.

    This is valid for both configured likelihoods: ``aleatoric_cov`` is the Gaussian
    covariance or the finite Student-t covariance (nu > 2), and the covariance of the
    stochastic conditional means is epistemic uncertainty.
    """
    if samples < 1:
        raise ValueError("mc_dropout_samples must be at least 1")
    stochastic = [model(batch["daily"], batch["news"]) for _ in range(samples)]
    mus = torch.stack([pred.mu for pred in stochastic], dim=0)
    aleatoric = torch.stack([pred.aleatoric_cov for pred in stochastic], dim=0).mean(
        dim=0
    )
    mu = mus.mean(dim=0)
    if samples == 1:
        epistemic = torch.zeros_like(aleatoric)
    else:
        centered = mus - mu.unsqueeze(0)
        epistemic = torch.einsum("sbn,sbm->bnm", centered, centered) / (samples - 1)
    return mu, aleatoric, aleatoric + epistemic


@torch.no_grad()
def predict(
    model: MultimodalReturnModel,
    data: AlignedData,
    samples: list[Sample],
    batch_size: int = 64,
    mc_dropout_samples: int = 50,
    device: str = "auto",
) -> Predictions:
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    _enable_mc_dropout(model)
    loader = DataLoader(WindowDataset(data, samples), batch_size=batch_size)
    log.info(
        "MC-dropout prediction | samples=%d | device=%s | batches=%d",
        mc_dropout_samples,
        resolved_device,
        len(loader),
    )
    mus, aleatoric_covs, epistemic_covs, covs, tgts, ts = [], [], [], [], [], []
    for batch in loader:
        inputs = {
            key: value.to(resolved_device)
            for key, value in batch.items()
            if key in {"daily", "news"}
        }
        mu, aleatoric_cov, cov = _mc_predict_batch(model, inputs, mc_dropout_samples)
        mus.append(mu.cpu().numpy())
        aleatoric_covs.append(aleatoric_cov.cpu().numpy())
        epistemic_covs.append((cov - aleatoric_cov).cpu().numpy())
        covs.append(cov.cpu().numpy())
        tgts.append(batch["target"].numpy())
        ts.append(batch["t"].numpy())
    model.to("cpu")
    return Predictions(
        decision_t=np.concatenate(ts),
        mu=np.concatenate(mus),
        cov=np.concatenate(covs),
        aleatoric_cov=np.concatenate(aleatoric_covs),
        epistemic_cov=np.concatenate(epistemic_covs),
        target=np.concatenate(tgts),
        assets=data.assets,
    )
