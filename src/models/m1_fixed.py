# -*- coding: utf-8 -*-
"""M1-Fixed (g = 0.5 constant; no gate network).
Provenance: byte-identical to M1FixedModel6 in paper_preparation/10_Phase32_Results/01_Model_Implementation/model_phase32.py (only the mlp import is localized).
"""
import torch
import torch.nn as nn
from .m1 import mlp  # originally imported from src/models/experiment5/model5


class M1FixedModel6(nn.Module):
    """M1-Fixed：固定 0.5 融合（无门控网络）。"""

    def __init__(self, cfg, input_transform='log1p'):
        super().__init__()
        self.input_transform = input_transform
        self.shared_enc = mlp([cfg.n_shared, cfg.hidden, cfg.zdim])
        self.spec_enc = mlp([cfg.n_specific, cfg.hidden, cfg.zdim])
        # 无 fusion 模块（仅此一处与 M1 不同）
        self.fine = nn.Linear(cfg.zdim, cfg.n_fine)
        self.coarse = nn.Linear(cfg.zdim, cfg.n_coarse)
        self.use_spec = cfg.use_specific
        self.spec_dropout = getattr(cfg, 'spec_dropout', 0.2)

    def _transform(self, x):
        if self.input_transform == 'log1p':
            return torch.log1p(x)
        return x

    def forward(self, xs, xp, extra=None):
        zs = self.shared_enc(self._transform(xs))
        if self.use_spec:
            if self.training and self.spec_dropout > 0:
                keep = (torch.rand(xs.shape[0], device=xs.device)
                        > self.spec_dropout).float().unsqueeze(-1)
                xp = xp * keep
            zp = self.spec_enc(self._transform(xp))
        else:
            zp = torch.zeros_like(zs)
        g = torch.full((xs.shape[0], 1), 0.5, device=xs.device)  # 固定 0.5
        z = g * zs + (1 - g) * zp
        return dict(z=z, z_shared=zs, z_spec=zp, gate=g,
                    logits_fine=self.fine(z), logits_coarse=self.coarse(z))


