# -*- coding: utf-8 -*-
"""Phase 3.2 汇总工作簿 phase32_training_results.xlsx 与 5 张图。"""
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/home/hst/research/openset_st_cf'
P32 = ROOT + '/paper_preparation/10_Phase32_Results'
SA = P32 + '/06_Statistical_Analysis'
FIG = P32 + '/07_Figures'
os.makedirs(FIG, exist_ok=True)

C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', ink='#0b0b0b',
         ink2='#52514e', muted='#898781', grid='#e1e0d9', axis='#c3c2b7',
         surf='#fcfcfb', red='#e34948')
plt.rcParams.update({
    'figure.facecolor': C['surf'], 'axes.facecolor': C['surf'],
    'axes.edgecolor': C['axis'], 'axes.labelcolor': C['ink'],
    'text.color': C['ink'], 'xtick.color': C['ink'], 'ytick.color': C['ink'],
    'grid.color': C['grid'], 'font.size': 9, 'axes.titlesize': 10,
    'axes.labelsize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2.0,
    'legend.frameon': False, 'font.family': 'DejaVu Sans'})

MCOL = {'M1': C['blue'], 'M1-Fixed': C['orange'], 'M1-Global': C['aqua'],
        'M3_frozen': C['muted']}


def save(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(f'{FIG}/{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {name}', flush=True)


def main():
    # ---- 训练元数据（30 次运行的 meta json）----
    meta_rows = []
    for mp in sorted(glob.glob(f'{P32}/04_Checkpoints/*/meta_fold*.json')):
        meta_rows.append(json.load(open(mp)))
    df_meta = pd.DataFrame(meta_rows)
    df_meta['tag'] = df_meta.model + '_s' + df_meta.seed.astype(str)
    print('meta rows:', len(df_meta))
    print(df_meta.groupby('model').agg(
        n=('fold', 'count'), wall_s=('wall_seconds', 'mean'),
        peak_mb=('gpu_peak_mb', 'max'), alpha=('gate_alpha_final', 'mean')))

    oof = pd.read_csv(f'{SA}/phase32_oof_comparison.csv')
    pat = pd.read_csv(f'{SA}/phase32_patient_paired_statistics.csv')
    cls = pd.read_csv(f'{SA}/phase32_per_class_f1.csv')
    eq = pd.read_csv(f'{SA}/phase32_global_vs_m3_equivalence.csv')

    # ---- Excel ----
    with pd.ExcelWriter(f'{P32}/08_Reports/phase32_training_results.xlsx',
                        engine='openpyxl') as w:
        oof.to_excel(w, 'OOF_Comparison', index=False)
        pat.to_excel(w, 'Patient_Paired', index=False)
        cls.to_excel(w, 'Per_Class_F1', index=False)
        eq.to_excel(w, 'Global_vs_M3_Equivalence', index=False)
        df_meta.to_excel(w, 'Training_Runs', index=False)
        pd.DataFrame([
            dict(item='M1 vs M1-Fixed patient seed-avg',
                 value='+0.5204pp, W p=1.10e-5, dz=0.661, CI[+0.290,+0.789]'),
            dict(item='M1 vs M1-Global patient seed-avg',
                 value='+0.4262pp, W p=4.74e-5, dz=0.542, CI[+0.175,+0.686]'),
            dict(item='5/5 seeds M1>Fixed pooled', value='+0.44~+0.82pp'),
            dict(item='5/5 seeds M1>Global pooled', value='+0.39~+0.87pp'),
            dict(item='Global learned g', value='0.4928(fresh)/0.4929(m3) ≈ 0.5'),
            dict(item='Benefiting classes', value='a +4.9pp, g +2.9pp (per-class F1)'),
            dict(item='BH (2 comparisons)', value='both q<0.05'),
        ]).to_excel(w, 'Key_Results', index=False)

    # ---- 图 1：三模型性能比较 ----
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    models = ['M1', 'M1-Fixed', 'M1-Global']
    for ax, met, title in [(axes[0], 'pooled_acc', 'Pooled OOF accuracy'),
                           (axes[1], 'pooled_macro', 'Pooled macro-F1')]:
        means, sds = [], []
        for m in models:
            sub = oof[oof.model == m][met]
            means.append(sub.mean())
            sds.append(sub.std(ddof=1))
        x = np.arange(3)
        ax.bar(x, means, color=[MCOL[m] for m in models], width=0.55,
               yerr=sds, error_kw=dict(ecolor=C['ink2'], lw=1.0, capsize=3),
               linewidth=0)
        for i, m in enumerate(models):
            pts = oof[oof.model == m][met].values
            ax.scatter(np.full(5, i), pts, s=13, color=C['ink'], zorder=5,
                       marker='|', linewidths=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title(title)
        ax.grid(axis='y', linewidth=0.6)
        if met == 'pooled_acc':
            ax.set_ylim(0.885, 0.902)
    fig.suptitle('Trained M1 vs fixed-0.5 and global-scalar controls '
                 '(5 seeds, dots = seeds, error bars = seed SD)')
    fig.tight_layout()
    save(fig, 'fig32_1_model_comparison')

    # ---- 图 2：患者级配对森林 ----
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    sub = pat[pat.model.str.contains('seed-avg')]
    y = 0
    yticks, ylabels = [], []
    for _, r in sub.iterrows():
        pair = r.model.split(' (')[0]
        col = C['blue'] if 'M1-Fixed' in pair else C['aqua']
        lo, hi = r.bootstrap_lo * 100, r.bootstrap_hi * 100
        d = r.seed_avg_patient_acc_mean * 100
        ax.plot([lo, hi], [y, y], color=col, linewidth=2.0,
                solid_capstyle='round')
        ax.scatter([d], [y], s=28, color=col, zorder=5, linewidths=0)
        ax.text(hi + 0.02, y, f"p={r.wilcoxon_p:.1e}", fontsize=7.5,
                color=C['ink2'], va='center')
        yticks.append(y)
        ylabels.append(pair.replace(' - ', ' vs '))
        y -= 1
    for _, r in pat[pat.model.str.contains(' - ') &
                    ~pat.model.str.contains('seed-avg')].iterrows():
        pair = r.model.split(' - ')[1].split(' seed')[0]
        col = C['blue'] if pair == 'M1-Fixed' else C['aqua']
        ax.scatter([r.seed_avg_patient_acc_mean * 100], [y], s=16, color=col,
                   alpha=0.7, zorder=5, linewidths=0)
        yticks.append(y)
        ylabels.append(f"{pair} s{r.model.split('seed')[-1]}")
        y -= 1
    ax.axvline(0, color=C['ink2'], linewidth=1.0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7.5)
    ax.set_xlabel('Patient-averaged paired accuracy difference (pp)')
    ax.grid(axis='x', linewidth=0.6)
    ax.set_title('Patient-level paired differences (seed-averaged, 37 patients, '
                 'bootstrap 95% CI)')
    fig.tight_layout()
    save(fig, 'fig32_2_patient_paired')

    # ---- 图 3：五种子稳定性 ----
    fig, ax = plt.subplots(figsize=(7, 3.2))
    seeds = sorted(oof.seed.unique())
    for m in models:
        sub = oof[oof.model == m].sort_values('seed')
        ax.plot(seeds, sub.pooled_acc.values * 100, marker='o', markersize=4,
                color=MCOL[m], linewidth=1.8, label=m)
    ax.set_xlabel('Seed')
    ax.set_ylabel('Pooled OOF accuracy (%)')
    ax.legend()
    ax.grid(linewidth=0.6)
    ax.set_title('Seed stability: M1 above both controls in all 5 seeds')
    fig.tight_layout()
    save(fig, 'fig32_3_seed_stability')

    # ---- 图 4：8 类 per-class F1 ----
    fig, ax = plt.subplots(figsize=(8, 3.2))
    names = ['a', 'b', 'c', 'd', 'e', 'g', 'i', 'j']
    x = np.arange(8)
    for k, m in enumerate(models):
        sub = cls[cls.model == m].set_index('class_idx').loc[range(8)]
        ax.bar(x + (k - 1) * 0.27, sub.f1_mean, width=0.27, color=MCOL[m],
               label=m, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlabel('Fine class')
    ax.set_ylabel('Per-class F1 (5-seed mean)')
    ax.legend()
    ax.grid(axis='y', linewidth=0.6)
    ax.set_title('Per-class F1: M1 gains concentrated in classes a and g')
    fig.tight_layout()
    save(fig, 'fig32_4_per_class_f1')

    # ---- 图 5：门控分布 ----
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    d = np.load(f'{P32}/05_OOF_Predictions/m1fixed_s42/pred_fold1_test.npz')
    axes[0].hist(d['gate'], bins=20, range=(0, 1), color='#cde2fb',
                 edgecolor=C['blue'], linewidth=0.5)
    axes[0].axvline(0.5, color=C['red'], linewidth=2.0)
    axes[0].set_xlabel('Gate g')
    axes[0].set_ylabel('Cells')
    axes[0].set_title('M1-Fixed: g = 0.5 constant (all cells)')
    axes[0].grid(axis='y', linewidth=0.6)
    d1 = np.load(f'{ROOT}/outputs/exp6/m1_s42/pred_fold1_test.npz')
    dg = np.load(f'{P32}/05_OOF_Predictions/m1global_s42/pred_fold1_test.npz')
    axes[1].hist(d1['gate'], bins=30, range=(0, 1), color='#cde2fb',
                 edgecolor=C['blue'], linewidth=0.5, label='M1 per-cell gate',
                 alpha=0.9)
    axes[1].axvline(dg['gate'][0], color=C['aqua'], linewidth=2.5,
                    label=f'M1-Global g = {dg["gate"][0]:.4f}')
    axes[1].set_xlabel('Gate g')
    axes[1].set_title('M1 adaptive per-cell gates vs global scalar')
    axes[1].legend(fontsize=7.5)
    axes[1].grid(axis='y', linewidth=0.6)
    fig.suptitle('Gate distributions (seed 42, fold 1 test)')
    fig.tight_layout()
    save(fig, 'fig32_5_gate_distributions')
    print('PHASE32_ARTIFACTS_DONE', flush=True)


if __name__ == '__main__':
    main()
