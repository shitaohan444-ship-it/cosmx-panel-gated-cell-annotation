# -*- coding: utf-8 -*-
"""M1 (final manuscript model).

Provenance: this file reproduces byte-identically:
  - mlp() and GateFusion5 from src/models/experiment5/model5.py
  - M1Model6 from src/models/experiment6/model6.py
The only difference from the run code is the import location of mlp/
GateFusion5 (local definition instead of package import).
Architecture: z = g * z_shared + (1-g) * z_specific with
g = sigmoid(MLP([z_sh; z_sp])) in [B,1]; fine/coarse linear heads.
"""
import torch
import torch.nn as nn


def mlp(dims, bn=True):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if bn:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class GateFusion5(nn.Module):
    def __init__(self, zdim=128, extra_dim=0, mode='base'):
        super().__init__()
        self.mode = mode
        self.zdim = zdim
        if mode == 'base':
            self.gate = nn.Sequential(nn.Linear(2 * zdim, zdim), nn.ReLU(),
                                      nn.Linear(zdim, 1), nn.Sigmoid())
        elif mode in ('platform', 'cell_evidence', 'full'):
            self.gate = nn.Sequential(nn.Linear(2 * zdim + extra_dim, zdim), nn.ReLU(),
                                      nn.Linear(zdim, 1), nn.Sigmoid())
        elif mode == 'fixed':
            self.w = nn.Parameter(torch.tensor(0.5))
        elif mode == 'concat':
            pass
        else:
            raise ValueError(mode)

    def forward(self, z_shared, z_spec, extra=None):
        if self.mode == 'base':
            g = self.gate(torch.cat([z_shared, z_spec], dim=-1))
            z = g * z_shared + (1 - g) * z_spec
            return z, g
        if self.mode in ('platform', 'cell_evidence', 'full'):
            g = self.gate(torch.cat([z_shared, z_spec, extra], dim=-1))
            z = g * z_shared + (1 - g) * z_spec
            return z, g
        if self.mode == 'fixed':
            w = torch.sigmoid(self.w)
            z = w * z_shared + (1 - w) * z_spec
            g = w.expand(z_shared.shape[0], 1)
            return z, g
        if self.mode == 'concat':
            return torch.cat([z_shared, z_spec], dim=-1), None
        raise ValueError(self.mode)


# -*- coding: utf-8 -*-
"""实验6：统一 M1 复现模型。

与 src/models/experiment5/model5.py 的 M1（base 门控）在结构、参数构造顺序上
完全一致（保证相同随机种子下初始化逐位相同），仅增加 input_transform 开关：
  - log1p：编码前 log1p 变换（= exp5 协议）
  - raw  ：原始计数直接输入编码器（= 4.3 协议）
"""

import torch
import torch.nn as nn

# (local definition below; originally imported from src/models/experiment5/model5.py — byte-identical code)


class M1Model6(nn.Module):
    def __init__(self, cfg, input_transform='log1p'):
        super().__init__()
        self.input_transform = input_transform
        self.gate_mode = cfg.gate_mode
        self.shared_enc = mlp([cfg.n_shared, cfg.hidden, cfg.zdim])
        self.spec_enc = mlp([cfg.n_specific, cfg.hidden, cfg.zdim])
        mode_extra = {'platform': 2, 'cell_evidence': 3, 'full': 5}
        extra_dim = mode_extra.get(cfg.gate_mode, getattr(cfg, 'gate_extra_dim', 0))
        self.fusion = GateFusion5(cfg.zdim, extra_dim, self.gate_mode)
        out_dim = cfg.zdim * 2 if self.gate_mode == 'concat' else cfg.zdim
        self.fine = nn.Linear(out_dim, cfg.n_fine)
        self.coarse = nn.Linear(out_dim, cfg.n_coarse)
        self.use_spec = cfg.use_specific
        self.spec_dropout = getattr(cfg, 'spec_dropout', 0.2)
        self.zero_shared = getattr(cfg, 'zero_shared', False)
        self.zero_spec = getattr(cfg, 'zero_spec', False)

    def _transform(self, x):
        if self.input_transform == 'log1p':
            return torch.log1p(x)
        return x

    def forward(self, xs, xp, extra=None):
        if self.zero_shared:
            xs = xs * 0.0
        if self.zero_spec:
            xp = xp * 0.0
        zs = self.shared_enc(self._transform(xs))
        if self.use_spec:
            if self.training and self.spec_dropout > 0:
                keep = (torch.rand(xs.shape[0], device=xs.device)
                        > self.spec_dropout).float().unsqueeze(-1)
                xp = xp * keep
            zp = self.spec_enc(self._transform(xp))
        else:
            zp = torch.zeros_like(zs)
        z, g = self.fusion(zs, zp, extra)
        return dict(z=z, z_shared=zs, z_spec=zp, gate=g,
                    logits_fine=self.fine(z), logits_coarse=self.coarse(z))
