# -*- coding: utf-8 -*-
"""Calibration analysis (ECE 15-bin / multi-class Brier), adapted from
13_Phase35_Final_Manuscript/05_Tables/phase35_calibration.py. Identical
protocol: proba_f, 15 equal-width bins, no temperature scaling, same cells."""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import metrics as M

SEEDS = [42, 7, 99, 123, 2024]
FOLDS = [1, 2, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        exp = yaml.safe_load(f)
    data = np.load(exp['preprocessed_npz'], allow_pickle=True)
    y_all = data['y_fine']
    rows = []
    for model, base in exp['prediction_dirs'].items():
        for s in SEEDS:
            yl, pl = [], []
            for f in FOLDS:
                d = np.load(os.path.join(base % s, 'pred_fold%d_test.npz' % f))
                yl.append(y_all[d['idx']])
                pl.append(np.asarray(d['proba_f'], dtype=np.float64))
            y = np.concatenate(yl)
            p = np.concatenate(pl)
            assert abs(p.sum(1) - 1).max() < 1e-5
            rows.append(dict(model=model, seed=s, n_cells=len(y),
                             ece=M.ece15(p, y)[0], brier=M.brier_mc(p, y)))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.groupby('model')[['ece', 'brier']].mean().round(6).to_string())


if __name__ == '__main__':
    main()
