# -*- coding: utf-8 -*-
"""Patient-level paired statistics for M1 vs M1-Fixed / M1-Global.

Adapted from 10_Phase32_Results/09_Scripts/phase32_stats.py. Primary estimand:
per-patient accuracy averaged over the five seeds, then equal-weight mean over
37 patients; two-sided Wilcoxon; patient-clustered bootstrap (B=2000, rng
20260331); Cohen's dz; BH over the two comparisons.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import metrics as M  # noqa: F401

SEEDS = [42, 7, 99, 123, 2024]
FOLDS = [1, 2, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        exp = yaml.safe_load(f)
    core = pd.read_csv(exp['core_patient_map'],
                       dtype=str)  # columns: core, patient
    c2p = dict(zip(core.core, core.patient))

    def load(base):
        out = {}
        for s in SEEDS:
            yl, ph, pt = [], [], []
            for f in FOLDS:
                d = np.load(os.path.join(base % s, 'pred_fold%d_test.npz' % f))
                idx = d['idx']
                yl.append(d['y_fine'][idx] if 'y_fine' in d.files else None)
                ph.append(np.asarray(d['proba_f']))
                pt.append(np.array([c2p[c] for c in d['core_arr'][idx]]))
            yh = np.concatenate(ph).argmax(1)
            pat = np.concatenate(pt)
            # y from npz if shipped; else assume idx rows carry labels via data npz
            y = np.concatenate(yl) if yl[0] is not None else None
            out[s] = (y, yh, pat)
        return out

    data = np.load(exp['preprocessed_npz'], allow_pickle=True)
    y_all = data['y_fine']

    def load2(base):
        out = {}
        for s in SEEDS:
            yl, ph, pt = [], [], []
            for f in FOLDS:
                d = np.load(os.path.join(base % s, 'pred_fold%d_test.npz' % f))
                idx = d['idx']
                yl.append(y_all[idx])
                ph.append(np.asarray(d['proba_f']))
                pt.append(np.array([c2p[c] for c in data['core_arr'][idx]]))
            y = np.concatenate(yl)
            yh = np.concatenate(ph).argmax(1)
            pat = np.concatenate(pt)
            out[s] = (y, yh, pat)
        return out

    def patient_seed_avg(model_base):
        per = {}
        for s in SEEDS:
            y, yh, pat = load2(model_base)[s]
            for p in np.unique(pat):
                per.setdefault(p, []).append(M.accuracy(y[pat == p],
                                                        yh[pat == p]))
        return per

    pa = patient_seed_avg(exp['prediction_dirs']['m1'])
    rows = []
    pats = sorted(pa.keys())
    for tag in ['m1_fixed', 'm1_global']:
        pb = patient_seed_avg(exp['prediction_dirs'][tag])
        a = np.array([np.mean(pa[p]) for p in pats])
        b = np.array([np.mean(pb[p]) for p in pats])
        d = a - b
        w = stats.wilcoxon(d[d != 0], zero_method='wilcox',
                           alternative='two-sided')
        dz = M.cohens_dz(d)
        lo, hi, pneg, mean_d, n = M.patient_bootstrap_ci(a, b, 2000, 20260331)
        per_seed = []
        for si, s in enumerate(SEEDS):
            ds = np.array([pa[p][si] - pb[p][si] for p in pats])
            ws = stats.wilcoxon(ds[ds != 0], zero_method='wilcox',
                                alternative='two-sided')
            per_seed.append((float(np.mean(ds)) * 100, float(ws.pvalue)))
        rows.append(dict(comparison='M1 - %s' % tag,
                         patient_seed_avg_diff_pp=mean_d * 100,
                         wilcoxon_p=float(w.pvalue), cohens_dz=dz,
                         ci_lo_pp=lo * 100, ci_hi_pp=hi * 100,
                         per_seed_pp_p=str(per_seed)))
    # BH over the two primary comparisons
    from src.metrics import bh_fdr
    q = bh_fdr([r['wilcoxon_p'] for r in rows])
    for r, qv in zip(rows, q):
        r['bh_q'] = qv
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(rows)


if __name__ == '__main__':
    main()
