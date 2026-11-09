# -*- coding: utf-8 -*-
"""Phase 3.5 校准性能独立重算（07_QC 输入）。

协议（与论文 Methods 4.8 一致，全新独立实现）：
- 概率：pred npz 的 proba_f（softmax 输出，8 类，未温度缩放）
- ECE：15 个等宽置信箱 [0,1]，每箱 |acc - conf| 加权；无温度缩放
- Brier（多类）：每细胞 sum_c (p_c - y_c)^2 的均值
- 分母：与 M1 相同的三折测试细胞（169,283；五种子各自合并）
- 模型：M1（outputs/exp6/m1_s*）、M1-Fixed/M1-Global（10_Phase32_Results）
输出：05_Tables/calibration_comparison.csv + calibration_analysis.md 数据
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = '/home/hst/research/openset_st_cf'
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + '/scripts/experiment5')
from train5 import build_data5

SEEDS = [42, 7, 99, 123, 2024]
FOLDS = [1, 2, 3]
OUT = ROOT + '/paper_preparation/13_Phase35_Final_Manuscript/05_Tables'
os.makedirs(OUT, exist_ok=True)

N_BINS = 15


def ece15(proba, y):
    conf = proba.max(1)
    pred = proba.argmax(1)
    accv = (pred == y).astype(np.float64)
    ece = 0.0
    for b in range(N_BINS):
        lo, hi = b / N_BINS, (b + 1) / N_BINS
        m = (conf >= lo) & (conf <= hi if hi == 1.0 else conf < hi)
        n = m.sum()
        if n == 0:
            continue
        ece += (n / len(y)) * abs(accv[m].mean() - conf[m].mean())
    return float(ece)


def brier_mc(proba, y):
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def main():
    data = build_data5(smoke=False)
    y_all = data['y_fine']
    model_dirs = {
        'M1': f'{ROOT}/outputs/exp6/m1_s',
        'M1-Fixed': f'{ROOT}/paper_preparation/10_Phase32_Results/'
                    '05_OOF_Predictions/m1fixed_s',
        'M1-Global': f'{ROOT}/paper_preparation/10_Phase32_Results/'
                     '05_OOF_Predictions/m1global_s',
    }
    rows = []
    for model, base in model_dirs.items():
        for s in SEEDS:
            yp, pp = [], []
            for f in FOLDS:
                d = np.load(f'{base}{s}/pred_fold{f}_test.npz')
                yp.append(y_all[d['idx']])
                pp.append(np.asarray(d['proba_f'], dtype=np.float64))
            y = np.concatenate(yp)
            p = np.concatenate(pp)
            # 一致性检查
            assert len(y) == 169283, f'{model} s{s} n={len(y)}'
            assert np.abs(p.sum(1) - 1).max() < 1e-5, 'proba not normalized'
            rows.append(dict(model=model, seed=s, n_cells=len(y),
                             ece=ece15(p, y), brier=brier_mc(p, y)))
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/calibration_comparison.csv', index=False)
    summ = df.groupby('model')[['ece', 'brier']].agg(
        ['mean', 'std']).round(6)
    print(summ.to_string())
    print('CALIBRATION_DONE', flush=True)


if __name__ == '__main__':
    main()
