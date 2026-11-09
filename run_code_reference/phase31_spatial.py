# -*- coding: utf-8 -*-
"""任务 2.4：空间统计修复。

原口径（scripts/experiment6/spatial_analysis.py / p16_spatial_rerun.py）：
- 每核心（n>=30 细胞）内：g_i 与其 K=5 最近邻平均门控的 Pearson 相关 r
- 100 次随机置换（RNG 42），p = mean(|r_perm|>=|r|)（直接除法，可产生 p=0）
- 符号检验：i_vs_epi / i_vs_bas / j_vs_epi / j_vs_bas（合格核心：每类>=10 且核心>=50）
- Spearman：核心 r vs 免疫比例(i+j)、核心 r vs 核心 gate 均值

本轮修复：
1. 1999 次置换，加一 p 值 (b+1)/(B+1)（双侧），正/负方向分别报告
2. 150 核心 BH（核心为检验单位）+ 全家族 BH（150 自相关 + 2 Spearman）
3. 患者聚类敏感性：(a) 每患者平均 r 的 Wilcoxon 符号秩与 bootstrap CI；
   (b) 患者块置换（核心重抽样以患者为块）的自相关均值分布；
   (c) 患者间/患者内方差分解（ICC 风格）
锚点：名义显著 90（89 正 + 1 负，i+j 比例 ρ=-0.163 p≈0.046）；加一后 89 正向；
     150 核心 BH 后 80 正向；符号检验 i_vs_bas 27/27 p=1.49e-8、i_vs_epi 4/4 p=0.125。
"""
import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase31_common import bh_fdr, perm_pvalue_plus1, wilcoxon_paired
from phase31_infra import load_build_data, patient_of_idx, ROOT

OUT = ROOT + '/paper_preparation/09_Phase31_Results/02_Biological_Attribution'
E = ROOT + '/data/raw/ist_benchmarking/extracted'
FINE_NAMES = ['a', 'b', 'c', 'd', 'e', 'g', 'i', 'j']
K = 5
N_PERMS = 1999
RNG_SEED = 42


def main():
    os.makedirs(OUT, exist_ok=True)
    data = load_build_data()
    a = ad.read_h5ad(E + '/2023_cosmx_multitissue_tTMA1.h5ad', backed='r')
    obs = a.obs
    core_all = np.asarray(obs['core'].astype(str), dtype=str)
    x_all = np.asarray(obs['x_centroid'], dtype=float)
    y_all = np.asarray(obs['y_centroid'], dtype=float)
    labcol = [c for c in obs.columns if 'nbclust' in c and 'cluster' in c
              and 'posterior' not in c][0]
    cats = list(obs[labcol].cat.categories)
    lab_arr = np.asarray(obs[labcol].cat.codes)
    mapping = pd.read_csv(ROOT + '/data/metadata/celltype_mapping.tsv', sep='\t')
    FINE = mapping[mapping.unmapped_label != 'low_confidence'].fine_label.tolist()
    y_str = np.array([cats[c] if cats[c] in FINE else 'AMB' for c in lab_arr],
                     dtype=object)
    sca = pd.read_csv(ROOT + '/data/metadata/sceneA_final_manifest.csv')
    elig = set(sca[sca.eligible == True].core.astype(str))
    keep_mask = np.isin(core_all, list(elig)) & (y_str != 'AMB')
    h5ad_rows = np.where(keep_mask)[0]
    a.file.close()

    # 拼接三折预测（与 build_data5 行序对齐）
    ys, gates, cores = [], [], []
    data_pos = np.zeros(len(data['y_fine']), dtype=int)
    pos = 0
    for f in [1, 2, 3]:
        z = np.load(f'{ROOT}/outputs/exp6/m1_s42/pred_fold{f}_test.npz')
        idx = z['idx']
        ys.append(data['y_fine'][idx])
        gates.append(z['gate'])
        cores.append(data['core_arr'][idx])
        data_pos[pos:pos + len(idx)] = idx
        pos += len(idx)
    y = np.concatenate(ys)
    g = np.concatenate(gates)
    core = np.concatenate(cores)
    h5rows = h5ad_rows[data_pos]
    x = x_all[h5rows]
    ycoord = y_all[h5rows]
    patients = patient_of_idx(np.concatenate([np.load(
        f'{ROOT}/outputs/exp6/m1_s42/pred_fold{f}_test.npz')['idx']
        for f in [1, 2, 3]]), data['core_arr'])

    from sklearn.neighbors import NearestNeighbors

    def knn_neighbors(coords, k=K):
        nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
        _, idx = nn.kneighbors(coords)
        return idx[:, 1:]

    rng = np.random.RandomState(RNG_SEED)
    rows = []
    uniq_cores = sorted(np.unique(core))
    for c in uniq_cores:
        m = core == c
        n = int(m.sum())
        if n < 30:
            continue
        coords = np.stack([x[m], ycoord[m]], axis=1)
        gi = g[m]
        nb = knn_neighbors(coords, k=K)
        neigh_mean = gi[nb].mean(axis=1)
        r = float(np.corrcoef(gi, neigh_mean)[0, 1])
        b_ge = 0
        for _ in range(N_PERMS):
            gp = rng.permutation(gi)
            rp = float(np.corrcoef(gp, gp[nb].mean(axis=1))[0, 1])
            if abs(rp) >= abs(r):
                b_ge += 1
        p_plus1 = (b_ge + 1) / (N_PERMS + 1)  # 加一双侧置换 p 值
        imm_frac = float(((y[m] == 6) | (y[m] == 7)).mean())
        rows.append(dict(core=c, n=n, gate_autocorr=r, direction='pos' if r > 0 else 'neg',
                         immune_frac=imm_frac, gate_mean=float(gi.mean()),
                         patient=patients[m][0], p_plus1_twosided=p_plus1))

    dfc = pd.DataFrame(rows)
    dfc['bh_150'] = bh_fdr(dfc.p_plus1_twosided.values)
    n_pos_nom = int((dfc.p_plus1_twosided < 0.05).sum())
    n_neg = int((dfc.gate_autocorr < 0).sum())
    n_pos = int((dfc.gate_autocorr > 0).sum())
    print(f'cores n>=30: {len(dfc)} | 正相关 {n_pos} 负相关 {n_neg} | '
          f'加一名义显著 {n_pos_nom} | BH(150) 显著 {int((dfc.bh_150 < 0.05).sum())}',
          flush=True)
    # 负相关核心明细
    print('负相关核心:')
    print(dfc[dfc.gate_autocorr < 0][['core', 'n', 'gate_autocorr', 'p_plus1_twosided']]
          .to_string(index=False), flush=True)

    # ---- 患者聚类敏感性 ----
    pat_mean = dfc.groupby('patient').gate_autocorr.mean()
    w_stat, w_p, n_nz = wilcoxon_paired(pat_mean.values)
    print(f'患者级平均自相关: n={len(pat_mean)} 均值={pat_mean.mean():.4f} '
          f'Wilcoxon vs 0: p={w_p:.4g} (n_nonzero={n_nz})', flush=True)
    # 患者块 bootstrap
    pat_list = sorted(dfc.patient.unique())
    rng_b = np.random.RandomState(20260331)
    boots = np.empty(2000)
    for r_ in range(2000):
        sp = rng_b.choice(pat_list, size=len(pat_list), replace=True)
        sub = dfc[dfc.patient.isin(sp)]
        boots[r_] = sub.gate_autocorr.mean()
    print(f'患者块 bootstrap: 均值 {dfc.gate_autocorr.mean():.4f} '
          f'95%CI [{np.percentile(boots, 2.5):.4f}, {np.percentile(boots, 97.5):.4f}]',
          flush=True)
    # 方差分解（患者间 vs 患者内）
    grand = dfc.gate_autocorr.mean()
    n_pat = len(pat_list)
    ss_between = sum(len(dfc[dfc.patient == p]) * (pat_mean[p] - grand) ** 2
                     for p in pat_list)
    ss_within = float(sum(((dfc[dfc.patient == p].gate_autocorr - pat_mean[p]) ** 2).sum()
                          for p in pat_list))
    n_cores = len(dfc)
    ms_between = ss_between / (n_pat - 1)
    ms_within = ss_within / (n_cores - n_pat)
    icc = ms_between / (ms_between + ms_within) if (ms_between + ms_within) > 0 else 0
    print(f'方差分解: MS_between={ms_between:.5f} MS_within={ms_within:.5f} '
          f'ICC={icc:.3f}', flush=True)

    # ---- Spearman（含免疫比例负相关锚点）----
    rho1, p1 = stats.spearmanr(dfc.gate_autocorr, dfc.immune_frac)
    rho2, p2 = stats.spearmanr(dfc.gate_autocorr, dfc.gate_mean)
    print(f'Spearman(r, immune_frac) rho={rho1:.3f} p={p1:.4g} (anchor -0.163, 0.046)')
    print(f'Spearman(r, gate_mean) rho={rho2:.3f} p={p2:.4g}')
    # 全家族 BH（150 自相关 + 2 Spearman）
    allp = np.concatenate([dfc.p_plus1_twosided.values, [p1, p2]])
    allq = bh_fdr(allp)
    print(f'全家族(152) BH 后自相关显著 {int((allq[:len(dfc)] < 0.05).sum())} '
          f'| immune_frac q={allq[len(dfc)]:.4g} | gate_mean q={allq[-1]:.4g}', flush=True)

    # ---- 符号检验 ----
    pairs = {'i_vs_epi': ('i', ['a', 'c']), 'i_vs_bas': ('i', ['d', 'e']),
             'j_vs_epi': ('j', ['a', 'c']), 'j_vs_bas': ('j', ['d', 'e'])}
    diff_rows = []
    for c in uniq_cores:
        m = core == c
        if m.sum() < 50:
            continue
        gm = {nm: g[m][y[m] == i].mean() for i, nm in enumerate(FINE_NAMES)
              if (y[m] == i).sum() >= 10}
        for pk, (hi, lows) in pairs.items():
            if hi in gm and all(l in gm for l in lows):
                diff_rows.append(dict(core=c, pair=pk,
                                      diff=gm[hi] - max(gm[l] for l in lows)))
    df_diff = pd.DataFrame(diff_rows)
    sign_rows = []
    for pk in pairs:
        sub = df_diff[df_diff.pair == pk]
        if len(sub) == 0:
            continue
        n_pos = int((sub['diff'] > 0).sum())
        pv = float(stats.binomtest(n_pos, len(sub), 0.5).pvalue)
        sign_rows.append(dict(pair=pk, n_valid_cores=len(sub), n_positive=n_pos,
                              frac_positive=n_pos / len(sub),
                              mean_diff=sub['diff'].mean(),
                              median_diff=sub['diff'].median(), binom_p=pv))
        print(f'{pk}: {n_pos}/{len(sub)} p={pv:.4g} (anchor i_vs_bas 27/27 1.49e-8, '
              f'i_vs_epi 4/4 0.125)', flush=True)
    # 患者级符号检验敏感性（每患者聚合）
    pat_diff = df_diff.merge(dfc[['core', 'patient']], on='core')
    pat_sign_rows = []
    for pk in pairs:
        sub = pat_diff[pat_diff.pair == pk]
        if len(sub) == 0:
            continue
        pm = sub.groupby('patient')['diff'].mean()
        n_pos = int((pm > 0).sum())
        pv = float(stats.binomtest(n_pos, len(pm), 0.5).pvalue)
        pat_sign_rows.append(dict(pair=pk, n_patients=len(pm), n_positive=n_pos,
                                  binom_p=pv))
        print(f'患者级 {pk}: {n_pos}/{len(pm)} p={pv:.4g}', flush=True)

    dfc.to_csv(f'{OUT}/spatial_reanalysis.csv', index=False)
    pd.DataFrame(sign_rows).to_csv(f'{OUT}/spatial_sign_test.csv', index=False)
    pd.DataFrame(pat_sign_rows).to_csv(f'{OUT}/spatial_sign_test_patient_level.csv',
                                       index=False)
    print('SPATIAL_DONE', flush=True)


if __name__ == '__main__':
    main()
