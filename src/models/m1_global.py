# -*- coding: utf-8 -*-
"""M1-Global (one globally learned scalar gate g = sigmoid(alpha)).
Provenance: byte-identical to M1GlobalModel6 in model_phase32.py
(only the mlp import is localized).
"""
import torch
import torch.nn as nn
from .m1 import mlp  # originally imported from src/models/experiment5/model5


class M1GlobalModel6(nn.Module):
    """M1-Global：全局可学习标量门控 g = sigmoid(alpha)，所有细胞共享。"""

    def __init__(self, cfg, input_transform='log1p'):
        super().__init__()
        self.input_transform = input_transform
        self.shared_enc = mlp([cfg.n_shared, cfg.hidden, cfg.zdim])
        self.spec_enc = mlp([cfg.n_specific, cfg.hidden, cfg.zdim])
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 唯一全局门控参数
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
        w = torch.sigmoid(self.alpha)
        g = w.expand(xs.shape[0], 1)
        z = g * zs + (1 - g) * zp
        return dict(z=z, z_shared=zs, z_spec=zp, gate=g,
                    logits_fine=self.fine(z), logits_coarse=self.coarse(z))
