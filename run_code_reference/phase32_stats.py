# -*- coding: utf-8 -*-
"""Phase 3.2 统计分析：M1 vs M1-Fixed vs M1-Global。

主估计量（任务定义）：患者平均配对 Accuracy 差——
先对每位患者五个随机种子求平均，再对 37 位患者等权平均。
次要：细胞加权合并 OOF。报告：效应量(dz)、95%CI（患者聚类 bootstrap）、
p 值（Wilcoxon 符号秩）、多重比较（2 个比较 BH）、五种子分别结果、
8 类 per-class F1。禁止把种子/细胞当独立生物学重复。

数据：
- M1        ：outputs/exp6/m1_s{42,7,99,123,2024}/pred_fold{1,2,3}_test.npz（冻结）
- M1-Global ：outputs/exp6/m3_s{...}/pred_fold*_test.npz（=M3，冻结；若用户选择
               全新训练则替换为 10_Phase32_Results/05_OOF_Predictions/m1global_s*）
- M1-Fixed  ：10_Phase32_Results/05_OOF_Predictions/m1fixed_s*/pred_fold*_test.npz（新训练）
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, '/home/hst/research/openset_st_cf/'
                   'paper_preparation/09_Phase31_Results/10_Scripts')
from phase31_common import (fold_level_metrics, accuracy, macro_f1,
                            balanced_accuracy, ece15, brier_mc, per_class_f1,
                            wilcoxon_paired, cohens_dz, patient_bootstrap_ci,
                            bh_fdr, patient_accuracy, sd_sample, sd_pop, fmt)
from phase31_infra import (load_build_data, load_pred_npz, patient_of_idx,
                           SEEDS, FOLDS, N_FINE, ROOT)

P32 = ROOT + '/paper_preparation/10_Phase32_Results'
OUT = P32 + '/06_Statistical_Analysis'
os.makedirs(OUT, exist_ok=True)


def load_model_preds(model, seed):
    """返回 (y, proba, folds, patients) 拼接三折。"""
    data = load_build_data()
    y_all, core_all = data['y_fine'], data['core_arr']
    if model == 'M1':
        d = {f: load_pred_npz(seed, f, 'm1') for f in FOLDS}
        yp = [y_all[d[f]['idx']] for f in FOLDS]
        pp = [np.asarray(d[f]['proba_f'], dtype=np.float64) for f in FOLDS]
        pats = [patient_of_idx(d[f]['idx'], core_all) for f in FOLDS]
    elif model == 'M3_frozen':
        d = {f: load_pred_npz(seed, f, 'm3') for f in FOLDS}
        yp = [y_all[d[f]['idx']] for f in FOLDS]
        pp = [np.asarray(d[f]['proba_f'], dtype=np.float64) for f in FOLDS]
        pats = [patient_of_idx(d[f]['idx'], core_all) for f in FOLDS]
    else:  # M1-Fixed / M1-Global（新训练）
        tag = 'm1fixed' if model == 'M1-Fixed' else 'm1global'
        yp, pp, pats = [], [], []
        for f in FOLDS:
            d = np.load(f'{P32}/05_OOF_Predictions/{tag}_s{seed}/'
                        f'pred_fold{f}_test.npz')
            yp.append(y_all[d['idx']])
            pp.append(np.asarray(d['proba_f'], dtype=np.float64))
            pats.append(patient_of_idx(d['idx'], core_all))
    folds = np.concatenate([np.full(len(yp[i]), f) for i, f in enumerate(FOLDS)])
    return (np.concatenate(yp), np.concatenate(pp), folds,
            np.concatenate(pats))


def main():
    models = ['M1', 'M1-Fixed', 'M1-Global', 'M3_frozen']
    # ---- 每种子 OOF 指标 ----
    oof_rows = []
    for model in models:
        for s in SEEDS:
            y, p, fl, pats = load_model_preds(model, s)
            met = fold_level_metrics(y, p.argmax(1), p, fl, N_FINE)
            oof_rows.append(dict(model=model, seed=s,
                                 eq_acc=met['eq_acc_mean'],
                                 eq_acc_sd_sample=met['eq_acc_sd_sample'],
                                 pooled_acc=met['pooled_acc'],
                                 eq_macro=met['eq_macro_mean'],
                                 pooled_macro=met['pooled_macro_f1'],
                                 pooled_bal=met['pooled_bal_acc'],
                                 pooled_ece=met['pooled_ece'],
                                 pooled_brier=met['pooled_brier']))
            print(f'{model} s{s}: pooled_acc={met["pooled_acc"]:.6f} '
                  f'eq={met["eq_acc_mean"]:.6f} pooled_macro='
                  f'{met["pooled_macro_f1"]:.6f}', flush=True)
    df_oof = pd.DataFrame(oof_rows)
    df_oof.to_csv(f'{OUT}/phase32_oof_comparison.csv', index=False)

    # ---- 协议等价性：fresh M1-Global vs 冻结 M3（期望高度一致）----
    eq_rows = []
    for s in SEEDS:
        yg, pg, _, _ = load_model_preds('M1-Global', s)
        ym, pm, _, _ = load_model_preds('M3_frozen', s)
        dprob = float(np.abs(pg - pm).max())
        dacc = accuracy(yg, pg.argmax(1)) - accuracy(ym, pm.argmax(1))
        eq_rows.append(dict(seed=s, max_dproba=dprob,
                            acc_diff_pp=dacc * 100))
        print(f'M1-Global vs M3_frozen s{s}: max|dproba|={dprob:.3e} '
              f'acc_diff={dacc * 100:+.4f}pp', flush=True)
    pd.DataFrame(eq_rows).to_csv(
        f'{OUT}/phase32_global_vs_m3_equivalence.csv', index=False)

    # ---- 患者级：先种子平均、后患者等权 ----
    pat_acc = {m: {} for m in models}      # model -> patient -> [5 seeds acc]
    for model in models:
        for s in SEEDS:
            y, p, fl, pats = load_model_preds(model, s)
            yh = p.argmax(1)
            patients, acc = patient_accuracy(y, yh, pats)
            for pt, a in zip(patients, acc):
                pat_acc[model].setdefault(pt, []).append(a)
    patients37 = sorted(pat_acc['M1'].keys())
    assert len(patients37) == 37, f'患者数 {len(patients37)} != 37'
    rows = []
    for model in models:
        vals = np.array([np.mean(pat_acc[model][pt]) for pt in patients37])
        rows.append(dict(model=model, seed_avg_patient_acc_mean=vals.mean(),
                         sd_across_patients=sd_sample(vals)))
    for pair_a, pair_b in [('M1', 'M1-Fixed'), ('M1', 'M1-Global')]:
        a = np.array([np.mean(pat_acc[pair_a][pt]) for pt in patients37])
        b = np.array([np.mean(pat_acc[pair_b][pt]) for pt in patients37])
        d = a - b
        w_stat, w_p, n_nz = wilcoxon_paired(d)
        dz = cohens_dz(d)
        lo, hi, pneg, mean_d, n = patient_bootstrap_ci(a, b, n_rep=2000,
                                                       rng_seed=20260331)
        rows.append(dict(model=f'{pair_a} - {pair_b} (patient seed-avg)',
                         seed_avg_patient_acc_mean=mean_d,
                         sd_across_patients=np.std(d, ddof=1),
                         wilcoxon_p=w_p, wilcoxon_n_nonzero=n_nz,
                         cohens_dz=dz, bootstrap_lo=lo, bootstrap_hi=hi,
                         bootstrap_prop_neg=pneg))
        print(f'{pair_a} vs {pair_b}: patient seed-avg diff='
              f'{mean_d * 100:+.4f}pp W p={w_p:.4g} dz={dz:.3f} '
              f'CI=[{lo * 100:+.3f},{hi * 100:+.3f}]pp', flush=True)
        # 每种子患者级（报告用）
        for s in SEEDS:
            d_s = np.array([pat_acc[pair_a][pt][SEEDS.index(s)]
                            - pat_acc[pair_b][pt][SEEDS.index(s)]
                            for pt in patients37])
            w_s, p_s, nz_s = wilcoxon_paired(d_s)
            rows.append(dict(model=f'{pair_a} - {pair_b} seed{s}',
                             seed_avg_patient_acc_mean=d_s.mean(),
                             sd_across_patients=np.std(d_s, ddof=1),
                             wilcoxon_p=p_s, wilcoxon_n_nonzero=nz_s,
                             cohens_dz=cohens_dz(d_s)))
    df_pat = pd.DataFrame(rows)
    # 多重比较（2 个主比较）
    main_idx = [i for i, r in enumerate(rows)
                if '(patient seed-avg)' in str(r['model'])]
    p_main = np.array([rows[i]['wilcoxon_p'] for i in main_idx])
    q_main = bh_fdr(p_main)
    for k, i in enumerate(main_idx):
        df_pat.loc[i, 'bh_2_comparisons'] = q_main[k]
    df_pat.to_csv(f'{OUT}/phase32_patient_paired_statistics.csv', index=False)

    # ---- 8 类 per-class F1（池化，5 种子均值）----
    cls_rows = []
    for model in models:
        f1_by_seed = np.zeros((len(SEEDS), N_FINE))
        for si, s in enumerate(SEEDS):
            y, p, fl, pats = load_model_preds(model, s)
            f1_by_seed[si] = per_class_f1(y, p.argmax(1), N_FINE)
        for c in range(N_FINE):
            cls_rows.append(dict(model=model, class_idx=c,
                                 f1_mean=f1_by_seed[:, c].mean(),
                                 f1_sd=sd_sample(f1_by_seed[:, c]),
                                 f1_seeds=';'.join(f'{v:.6f}'
                                                   for v in f1_by_seed[:, c])))
    df_cls = pd.DataFrame(cls_rows)
    df_cls.to_csv(f'{OUT}/phase32_per_class_f1.csv', index=False)
    # 受益/未受益类别
    for pair_b in ['M1-Fixed', 'M1-Global']:
        a = df_cls[df_cls.model == 'M1'].set_index('class_idx').f1_mean
        b = df_cls[df_cls.model == pair_b].set_index('class_idx').f1_mean
        d = (a - b) * 100
        names = load_build_data()['fine_names']
        print(f'M1 vs {pair_b} per-class F1 diff (pp):',
              {names[c]: round(float(d[c]), 3) for c in range(N_FINE)},
              flush=True)
    print('PHASE32_STATS_DONE', flush=True)


if __name__ == '__main__':
    main()
