# -*- coding: utf-8 -*-
"""Frozen-checkpoint gate counterfactuals (Phase 3.1 / manuscript Fig 6-7).

Adapted from paper_preparation/09_Phase31_Results/10_Scripts/
phase31_gate_counterfactual.py. Differences (see docs/DIFFERENCES_FROM_RUN_CODE.md):
paths come from experiment_config.example.yaml; data from preprocessed npz.
Interventions: A original, B g=1, C g=0, D g=0.5, E training-fold mean gate,
F 100 within-fold gate permutations, G 0.25/0.75 sensitivity.
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import M1Model6  # noqa: F401
from src.data import load_preprocessed, make_cfg
from src import metrics as M

INTERVENTIONS = [('A_original', 'learned'), ('B_shared_only', 'g=1'),
                 ('C_spec_only', 'g=0'), ('D_fixed_0.5', 'g=0.5'),
                 ('E_train_mean', 'train-mean'), ('G_0.25', '0.25'),
                 ('G_0.75', '0.75')]
N_PERMS = 100


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--ckpt', required=True, help='model_foldN.pt path')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        exp = yaml.safe_load(f)
    data = load_preprocessed(exp['preprocessed_npz'])
    cfg = make_cfg()
    model = M1Model6(cfg, input_transform='log1p')
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()
    te = np.where(data['folds'] == args.fold)[0]
    tr = np.where(data['folds'] != args.fold)[0]
    y_all = data['y_fine']

    def fwd(rows):
        zs, zp, g = [], [], []
        with torch.no_grad():
            for b in range(0, len(rows), 4096):
                o = model(torch.from_numpy(data['X_shared'][rows[b:b + 4096]]),
                          torch.from_numpy(data['X_specific'][rows[b:b + 4096]]))
                zs.append(o['z_shared'].numpy())
                zp.append(o['z_spec'].numpy())
                g.append(o['gate'].squeeze(-1).numpy())
        return (np.concatenate(zs), np.concatenate(zp), np.concatenate(g))

    zs, zp, g_test = fwd(te)
    _, _, g_train = fwd(tr)
    train_mean = float(g_train.mean())
    W = model.fine.weight.detach().numpy().astype(np.float32)
    b = model.fine.bias.detach().numpy().astype(np.float32)
    A0 = zp @ W.T + b
    D = (zs - zp) @ W.T
    y = y_all[te]
    rows = []
    gates = {'A_original': g_test, 'B_shared_only': np.ones(len(te), np.float32),
             'C_spec_only': np.zeros(len(te), np.float32),
             'D_fixed_0.5': np.full(len(te), 0.5, np.float32),
             'E_train_mean': np.full(len(te), train_mean, np.float32),
             'G_0.25': np.full(len(te), 0.25, np.float32),
             'G_0.75': np.full(len(te), 0.75, np.float32)}
    for key, _ in INTERVENTIONS:
        lg = A0 + gates[key][:, None] * D
        yh = lg.argmax(1)
        rows.append(dict(intervention=key, acc=M.accuracy(y, yh),
                         macro_f1=M.macro_f1(y, yh, 8),
                         bal_acc=M.balanced_accuracy(y, yh, 8),
                         ece=M.ece15(softmax(lg), y)[0],
                         brier=M.brier_mc(softmax(lg), y),
                         train_mean_gate=train_mean))
    # F: within-fold gate permutation (identical rng scheme to run code)
    lg_orig = A0 + g_test[:, None] * D
    acc_orig = M.accuracy(y, lg_orig.argmax(1))
    perm = []
    for p in range(N_PERMS):
        rng = np.random.RandomState(20260331 + args.seed * 1000
                                    + args.fold * 10 + p)
        gp = g_test[rng.permutation(len(g_test))]
        yh = (A0 + gp[:, None] * D).argmax(1)
        perm.append(M.accuracy(y, yh))
    pval = M.perm_pvalue_plus1(acc_orig, np.array(perm), 'greater')
    rows.append(dict(intervention='F_permutation', orig_acc=acc_orig,
                     perm_mean=float(np.mean(perm)),
                     perm_sd=float(np.std(perm, ddof=1)),
                     p_plus1=pval, n_perms=N_PERMS))
    import pandas as pd
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print('COUNTERFACTUAL_DONE')
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
