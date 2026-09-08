"""The multimodal return-distribution model.

Per asset, the two branch encoders produce a per-day embedding sequence; these are
concatenated into one multimodal vector per lookback day and run through a one-layer
LSTM (or a light self-attention encoder). The final asset embeddings feed two heads:
a mean head predicting mu in R^N, and an aleatoric-covariance head predicting a valid
Sigma_A = L L^T + diag(d) via a low-rank-plus-diagonal parameterization. At inference,
MC dropout turns repeated mean-head outputs into epistemic covariance Sigma_E.

The ``use_news`` flag drops the news branch for ablations without changing any other
wiring.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from tyche.portfolio.config import ModelConfig
from tyche.portfolio.model.encoders import SequenceConvEncoder


@dataclass
class Prediction:
    mu: torch.Tensor  # [B, N]
    scale_tril: torch.Tensor  # [B, N, N] lower-Cholesky factor of Sigma_A
    aleatoric_cov: torch.Tensor  # [B, N, N] Sigma_A = L L^T + diag(d)

    @property
    def cov(self) -> torch.Tensor:
        """Backward-compatible name for the training-time aleatoric covariance."""
        return self.aleatoric_cov


class MultimodalReturnModel(nn.Module):
    def __init__(
        self,
        n_assets: int,
        daily_features: int,
        news_features: int,
        cfg: ModelConfig,
        use_news: bool = True,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.use_news = use_news
        self.cov_eps = cfg.cov_eps
        self.cov_rank = cfg.cov_rank
        c, k, do = cfg.conv_channels, cfg.kernel_size, cfg.dropout

        self.daily_enc = SequenceConvEncoder(daily_features, c, k, do)
        fused = self.daily_enc.out_dim
        if use_news:
            self.news_enc = SequenceConvEncoder(news_features, c, k, do)
            fused += self.news_enc.out_dim

        h = cfg.hidden_dim
        self.sequence_kind = cfg.sequence_encoder
        if cfg.sequence_encoder == "attention":
            layer = nn.TransformerEncoderLayer(
                d_model=fused,
                nhead=4,
                dim_feedforward=2 * h,
                dropout=do,
                batch_first=True,
            )
            self.sequence = nn.TransformerEncoder(layer, num_layers=1)
            self.project = nn.Linear(fused, h)
        else:
            self.sequence = nn.LSTM(fused, h, batch_first=True)
            self.project = nn.Identity()

        # This dropout is deliberately placed immediately before both prediction heads.
        # It is left active only during MC-dropout inference to sample model uncertainty.
        self.head_dropout = nn.Dropout(do)
        self.mu_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
        self.factor_head = nn.Linear(h, cfg.cov_rank)
        self.diag_head = nn.Linear(h, 1)

    def _asset_embeddings(self, daily, news) -> torch.Tensor:
        b, n, t = daily.shape[0], daily.shape[1], daily.shape[2]
        parts = [self.daily_enc(daily)]  # [B, N, T, C]
        if self.use_news:
            parts.append(self.news_enc(news))
        fused = torch.cat(parts, dim=-1)  # [B, N, T, Cf]

        seq = fused.reshape(b * n, t, fused.shape[-1])
        if self.sequence_kind == "attention":
            enc = self.sequence(seq)
            last = self.project(enc[:, -1])
        else:
            _, (hn, _) = self.sequence(seq)
            last = hn[-1]  # [B*N, H]
        return last.reshape(b, n, -1)  # [B, N, H]

    def forward(self, daily, news) -> Prediction:
        emb = self._asset_embeddings(daily, news)  # [B, N, H]
        b, n = emb.shape[0], emb.shape[1]

        stochastic_emb = self.head_dropout(emb)
        mu = self.mu_head(stochastic_emb).squeeze(-1)  # [B, N]
        factor = self.factor_head(stochastic_emb)  # [B, N, rank]
        diag = F.softplus(self.diag_head(stochastic_emb).squeeze(-1)) + self.cov_eps

        cov = factor @ factor.transpose(-1, -2)  # [B, N, N] low-rank
        cov = cov + torch.diag_embed(diag)
        jitter = self.cov_eps * torch.eye(n, device=emb.device).expand(b, n, n)
        scale_tril = torch.linalg.cholesky(cov + jitter)
        return Prediction(mu=mu, scale_tril=scale_tril, aleatoric_cov=cov)
