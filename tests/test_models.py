# -*- coding: utf-8 -*-
"""CPU smoke tests for the release models (no data, no GPU required).

Verifies the paper's structural claims:
- parameter counts: M1 358,671 / M1-Fixed 325,646 / M1-Global 325,647
- input dims 108/892; outputs z/z_shared/z_spec (128), logits_fine (8),
  logits_coarse (6); gate shape [B,1]
- M1-Fixed gate == 0.5; M1-Global gate is a single broadcast scalar;
  M1 gate in (0,1)
- one training step runs (CE fine + coarse, clip 5.0)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace

from src.models import M1Model6, M1FixedModel6, M1GlobalModel6


def make_cfg():
    return SimpleNamespace(n_shared=108, n_specific=892, hidden=256, zdim=128,
                           n_fine=8, n_coarse=6, spec_dropout=0.2,
                           gate_mode='base', use_specific=True,
                           lr=1e-3, wd=1e-4, batch_size=1024, epochs=15,
                           random_seed=42)


def test_param_counts():
    cfg = make_cfg()
    counts = {}
    for name, cls in [('m1', M1Model6), ('m1_fixed', M1FixedModel6),
                      ('m1_global', M1GlobalModel6)]:
        counts[name] = sum(p.numel() for p in cls(cfg).parameters())
    assert counts['m1'] == 358671, counts
    assert counts['m1_fixed'] == 325646, counts
    assert counts['m1_global'] == 325647, counts
    assert counts['m1'] - counts['m1_fixed'] == 33025
    assert counts['m1'] - counts['m1_global'] == 33024
    print('param counts OK', counts)


def test_shapes_and_gates():
    cfg = make_cfg()
    torch.manual_seed(42)
    xs = torch.rand(16, 108)
    xp = torch.rand(16, 892)
    m1 = M1Model6(cfg, input_transform='log1p').eval()
    mf = M1FixedModel6(cfg, input_transform='log1p').eval()
    mg = M1GlobalModel6(cfg, input_transform='log1p').eval()
    with torch.no_grad():
        for m in (m1, mf, mg):
            o = m(xs, xp)
            assert o['z'].shape == (16, 128)
            assert o['z_shared'].shape == (16, 128)
            assert o['z_spec'].shape == (16, 128)
            assert o['gate'].shape == (16, 1)
            assert o['logits_fine'].shape == (16, 8)
            assert o['logits_coarse'].shape == (16, 6)
    with torch.no_grad():
        of = mf(xs, xp)
        og = mg(xs, xp)
        om = m1(xs, xp)
    assert float(of['gate'].abs().max()) == 0.5
    assert np.allclose(of['gate'].numpy(), 0.5)
    assert og['gate'].std() < 1e-7  # single broadcast scalar
    assert ((om['gate'] > 0) & (om['gate'] < 1)).all()
    print('shapes and gate behavior OK')


def test_training_step():
    cfg = make_cfg()
    torch.manual_seed(7)
    m = M1Model6(cfg, input_transform='log1p')
    opt = torch.optim.Adam(m.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    xs = torch.rand(64, 108)
    xp = torch.rand(64, 892)
    yf = torch.randint(0, 8, (64,))
    yc = torch.randint(0, 6, (64,))
    m.train()
    o = m(xs, xp)
    loss = (torch.nn.functional.cross_entropy(o['logits_fine'], yf)
            + torch.nn.functional.cross_entropy(o['logits_coarse'], yc))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
    opt.step()
    assert np.isfinite(loss.item())
    print('training step OK, loss=%.4f' % loss.item())


def test_metrics():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import metrics as M
    y = np.array([0, 1, 2, 2, 1])
    yh = np.array([0, 1, 2, 1, 2])
    assert abs(M.accuracy(y, yh) - 0.6) < 1e-12
    # perfectly calibrated case: all conf 0.6, accuracy 0.6 -> ECE 0
    p = np.full((5, 3), 0.2)
    p[np.arange(5), yh] = 0.6
    assert M.ece15(p, y)[0] < 1e-9
    assert M.brier_mc(p, y) > 0
    assert abs(M.bh_fdr([0.01, 0.04])[1] - 0.04) < 1e-9
    assert M.perm_pvalue_plus1(1.0, np.array([0.0])) == 0.5
    print('metrics OK')


if __name__ == '__main__':
    test_param_counts()
    test_shapes_and_gates()
    test_training_step()
    test_metrics()
    print('ALL_TESTS_PASS')
