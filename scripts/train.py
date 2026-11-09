# -*- coding: utf-8 -*-
"""Training script for M1, M1-Fixed and M1-Global (Phase 3.6 release).

Adapted from scripts/experiment6/train6.py and
10_Phase32_Results/09_Scripts/train_phase32.py (run code). Differences from
the run code (see docs/DIFFERENCES_FROM_RUN_CODE.md):
- Data comes from a user-provided preprocessed .npz (see src/data.py) instead
  of the in-house h5ad pipeline (build_data5); the npz schema matches
  build_data5's outputs exactly (X_shared/X_specific/y_fine/y_coarse/folds/core_arr).
- Paths are config-driven (configs/experiment_config.example.yaml).
The training loop itself (loss, optimizer, dropout, batch order, val split,
checkpoint format, prediction format) is line-identical to train6.py.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import M1Model6, M1FixedModel6, M1GlobalModel6  # noqa: F401
from src.data import load_preprocessed, make_cfg

MODELS = {'m1': M1Model6, 'm1_fixed': M1FixedModel6, 'm1_global': M1GlobalModel6}
SEED_VAL = 42  # fixed fit/val core split (identical to the run code)


def train_fold(cfg, data, fold, model_cls, out_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr = (data['folds'] != fold) & (data['folds'] > 0)
    idx_tr = np.where(tr)[0]
    core_tr = data['core_arr'][tr]
    uniq = np.unique(core_tr)
    rng = np.random.RandomState(SEED_VAL)
    rng.shuffle(uniq)
    n_val = max(1, int(0.2 * len(uniq)))
    val_cores = set(uniq[:n_val])
    val_mask = np.array([c in val_cores for c in core_tr])
    idx_fit = idx_tr[~val_mask]
    idx_val = idx_tr[val_mask]
    idx_te = np.where(data['folds'] == fold)[0]

    model = model_cls(cfg, input_transform='log1p').to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    n_batch = cfg.batch_size
    epochs = cfg.epochs
    Xs_t = torch.from_numpy(data['X_shared']).float().to(device)
    Xp_t = torch.from_numpy(data['X_specific']).float().to(device)
    yf_t = torch.from_numpy(data['y_fine']).long().to(device)
    yc_t = torch.from_numpy(data['y_coarse']).long().to(device)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, 'model_fold%d.pt' % fold)
    print('fold%d: fit=%d val=%d test=%d' % (fold, len(idx_fit), len(idx_val),
                                             len(idx_te)), flush=True)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        order = np.random.RandomState(cfg.random_seed + ep).permutation(idx_fit)
        tl = {}
        n_b = 0
        for b in range(0, len(order), n_batch):
            bidx = order[b:b + n_batch]
            out = model(Xs_t[bidx], Xp_t[bidx])
            l_cls = torch.nn.functional.cross_entropy(out['logits_fine'],
                                                      yf_t[bidx])
            l_cls_c = torch.nn.functional.cross_entropy(out['logits_coarse'],
                                                        yc_t[bidx])
            total = l_cls + l_cls_c
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl['l_cls'] = tl.get('l_cls', 0) + l_cls.item()
            tl['l_cls_c'] = tl.get('l_cls_c', 0) + l_cls_c.item()
            n_b += 1
        if ep % 2 == 0 or ep == epochs - 1:
            print('  ep%d l_cls=%.4f l_cls_c=%.4f' % (ep, tl['l_cls'] / n_b,
                                                      tl['l_cls_c'] / n_b),
                  flush=True)
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                    epoch=epochs - 1), ckpt)
    model.eval()

    def predict(idx):
        outs = []
        with torch.no_grad():
            for b in range(0, len(idx), 4096):
                bidx = idx[b:b + 4096]
                o = model(Xs_t[bidx], Xp_t[bidx])
                g = o['gate']
                g = g.squeeze(-1).cpu().numpy() if g is not None \
                    else np.zeros(len(bidx))
                outs.append(dict(
                    gate=g,
                    proba_f=torch.softmax(o['logits_fine'], -1).cpu().numpy(),
                    proba_c=torch.softmax(o['logits_coarse'], -1).cpu().numpy(),
                    z=o['z'].cpu().numpy()))
        return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}

    for tag, idx in [('val', idx_val), ('test', idx_te)]:
        d = predict(idx)
        np.savez_compressed(os.path.join(out_dir, 'pred_fold%d_%s.npz'
                                         % (fold, tag)),
                            idx=idx, **{k: d[k] for k in d})
    print('fold%d done in %.0fs' % (fold, time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True,
                    help='path to experiment_config yaml')
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        exp = yaml.safe_load(f)
    data = load_preprocessed(exp['preprocessed_npz'])
    cfg = make_cfg(exp['model_config'] if 'model_config' in exp
                   else None)
    cfg.random_seed = args.seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_fold(cfg, data, args.fold, MODELS[args.model], args.out_dir)
    print('TRAIN_DONE')


if __name__ == '__main__':
    main()
