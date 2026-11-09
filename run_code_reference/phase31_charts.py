# -*- coding: utf-8 -*-
"""Phase 3.1 图表脚本：9 张证据图（PNG+PDF+每图数据 CSV）。

遵循 dataviz 规范：分类 8 槽固定顺序（#2a78d6/#eb6834/#1baf7a/#eda100/
#e87ba4/#008300/#6250d6/#e34948）、蓝序列、蓝红发散+灰中点、
细标记/2px 线/隐性网格、无双轴、英文图注（论文图）。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = '/home/hst/research/openset_st_cf'
P31 = ROOT + '/paper_preparation/09_Phase31_Results'
FIG = P31 + '/09_Figures'
os.makedirs(FIG, exist_ok=True)

C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100',
         magenta='#e87ba4', green='#008300', violet='#6250d6', red='#e34948',
         ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9',
         axis='#c3c2b7', surf='#fcfcfb', gray='#b9b8b2')

plt.rcParams.update({
    'figure.facecolor': C['surf'], 'axes.facecolor': C['surf'],
    'axes.edgecolor': C['axis'], 'axes.labelcolor': C['ink'],
    'text.color': C['ink'], 'xtick.color': C['ink'], 'ytick.color': C['ink'],
    'grid.color': C['grid'], 'font.size': 9, 'axes.titlesize': 10,
    'axes.labelsize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2.0,
    'legend.frameon': False, 'font.family': 'DejaVu Sans'})


def save(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(f'{FIG}/{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {name}', flush=True)


# ----------------------------------------------------------------------
# Fig 1: M1-M2 / M1-M3 15 seed-fold differences
# ----------------------------------------------------------------------
def fig1():
    df = pd.read_csv(f'{P31}/01_Numerical_Repair/m1m2m3_15fold_diffs.csv')
    df.to_csv(f'{FIG}/data_fig1_m1m2m3_fold_diffs.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, col, title in [(axes[0], 'd12_pp', 'M1 - M2'),
                           (axes[1], 'd13_pp', 'M1 - M3')]:
        colors = [C['blue'] if v > 0 else C['red'] for v in df[col]]
        ax.bar(range(15), df[col], color=colors, width=0.68, linewidth=0)
        ax.axhline(0, color=C['ink2'], linewidth=1.0)
        ax.set_xticks(range(15))
        ax.set_xticklabels([f'{s}\nf{f}' for s, f in zip(df.seed, df.fold)],
                           fontsize=6.5)
        ax.set_ylabel('Accuracy difference (pp)')
        ax.set_title(title)
        ax.grid(axis='y', linewidth=0.6)
        n_pos = int((df[col] > 0).sum())
        ax.text(0.02, 0.94, f'{n_pos}/15 positive',
                transform=ax.transAxes, fontsize=8, color=C['ink2'])
    fig.suptitle('Per-fold M1 advantage over M2 and M3 (5 seeds x 3 folds)',
                 y=1.02)
    save(fig, 'fig31_1_m1m2m3_fold_diffs')


# ----------------------------------------------------------------------
# Fig 2: counterfactual interventions
# ----------------------------------------------------------------------
def fig2():
    df = pd.read_csv(f'{P31}/04_Frozen_Gate_Counterfactual/'
                     'counterfactual_seed_level.csv')
    df.to_csv(f'{FIG}/data_fig2_counterfactual_seed_level.csv', index=False)
    order = ['A_original', 'G_sens_0.25', 'E_train_mean', 'C_spec_only',
             'D_fixed_0.5', 'G_sens_0.75', 'B_shared_only']
    labels = ['A learned gate', 'G 0.25 fixed', 'E train-mean', 'C spec only',
              'D 0.5 fixed', 'G 0.75 fixed', 'B shared only']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, met, title in [(axes[0], 'eq_acc_mean', 'Accuracy (equal-fold mean)'),
                           (axes[1], 'eq_macro_mean', 'Macro-F1 (equal-fold mean)')]:
        means, sds = [], []
        for iv in order:
            sub = df[df.intervention == iv][met]
            means.append(sub.mean())
            sds.append(sub.std(ddof=1))
        y = np.arange(len(order))[::-1]
        cols = [C['blue'], C['aqua'], C['orange'], C['magenta'],
                C['yellow'], C['violet'], C['red']]
        ax.barh(y, means, color=cols, xerr=sds, height=0.62,
                error_kw=dict(ecolor=C['ink2'], lw=1.0, capsize=2))
        for i, iv in enumerate(order):
            pts = df[df.intervention == iv][met].values
            ax.scatter(pts, np.full(len(pts), y[i]), s=14, color=C['ink'],
                       zorder=5, marker='|', linewidths=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.grid(axis='x', linewidth=0.6)
        ax.set_xlim(left=min(means) - 3 * max(sds) - 0.01)
    fig.suptitle('Frozen-checkpoint gate counterfactuals (5 seeds, dots = seeds)',
                 y=1.02)
    fig.tight_layout()
    save(fig, 'fig31_2_counterfactual_interventions')


# ----------------------------------------------------------------------
# Fig 3: gate permutation
# ----------------------------------------------------------------------
def fig3():
    df = pd.read_csv(f'{P31}/04_Frozen_Gate_Counterfactual/'
                     'gate_permutation_results.csv')
    df.to_csv(f'{FIG}/data_fig3_gate_permutation.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    # (a) 每种子×折 original vs perm mean
    df2 = df.sort_values(['seed', 'fold'])
    x = np.arange(len(df2))
    axes[0].bar(x - 0.19, df2.orig_acc, width=0.38, color=C['blue'],
                label='Original gate')
    axes[0].bar(x + 0.19, df2.perm_acc_mean, width=0.38, color=C['orange'],
                label='Permuted gate (mean of 100)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{s}\nf{f}' for s, f in zip(df2.seed, df2.fold)],
                            fontsize=6.5)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0.85, 0.92)
    axes[0].legend(loc='lower left')
    axes[0].grid(axis='y', linewidth=0.6)
    axes[0].set_title('Original vs permuted gate (15 seed-folds)')
    # (b) 直方图（seed42 fold1）
    rng = np.random.RandomState(20260331)
    sim = rng.normal(df2.iloc[0].perm_acc_mean, df2.iloc[0].perm_acc_sd, 100)
    axes[1].hist(sim, bins=14, color='#cde2fb', edgecolor=C['blue'],
                 linewidth=0.6, label='Permutation acc (simulated, n=100)')
    axes[1].axvline(df2.iloc[0].orig_acc, color=C['red'], linewidth=2.0,
                    label='Original gate acc')
    axes[1].set_xlabel('Accuracy')
    axes[1].set_ylabel('Permutations')
    axes[1].legend(loc='upper left', fontsize=7.5)
    axes[1].set_title('seed 42 fold 1: p = 1/101 (all 100 perms lower)')
    axes[1].grid(axis='y', linewidth=0.6)
    fig.suptitle('Permuted cell-level gate destroys gate-cell matching', y=1.02)
    fig.tight_layout()
    save(fig, 'fig31_3_gate_permutation')


# ----------------------------------------------------------------------
# Fig 4: gate distribution + spatial autocorr
# ----------------------------------------------------------------------
def fig4():
    gd = pd.read_csv(f'{P31}/04_Frozen_Gate_Counterfactual/gate_distribution.csv')
    sp = pd.read_csv(f'{P31}/02_Biological_Attribution/spatial_reanalysis.csv')
    gd.to_csv(f'{FIG}/data_fig4_gate_distribution.csv', index=False)
    sp.to_csv(f'{FIG}/data_fig4_spatial.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    g42 = gd[gd.seed == 42]
    for f in [1, 2, 3]:
        sub = g42[g42.fold == f]
        axes[0].plot(sub.bin_left + 0.025, sub['count'], linewidth=1.6,
                     label=f'fold {f}')
    axes[0].axvline(0.5, color=C['ink2'], linewidth=1.0, linestyle='--')
    axes[0].text(0.51, 0.9, 'g = 0.5', transform=axes[0].get_xaxis_transform(),
                 fontsize=7.5, color=C['ink2'])
    axes[0].set_xlabel('Cell-level gate g')
    axes[0].set_ylabel('Cells')
    axes[0].legend()
    axes[0].set_title('Gate distribution, seed 42 (88.7% spec-dominant)')
    axes[0].grid(axis='y', linewidth=0.6)
    axes[1].scatter(sp.immune_frac, sp.gate_autocorr, s=14, color=C['blue'],
                    alpha=0.7, linewidths=0)
    axes[1].axhline(0, color=C['ink2'], linewidth=1.0)
    axes[1].set_xlabel('Immune fraction (i + j) per core')
    axes[1].set_ylabel('Gate spatial autocorrelation r')
    axes[1].set_title('Core-level gate autocorr vs immune fraction '
                      '(Spearman rho=-0.163, p=0.046)')
    axes[1].grid(linewidth=0.6)
    fig.suptitle('Gate distribution and spatial structure (150 cores)', y=1.02)
    fig.tight_layout()
    save(fig, 'fig31_4_gate_distribution_spatial')


# ----------------------------------------------------------------------
# Fig 5: patient-level paired forest
# ----------------------------------------------------------------------
def fig5():
    df = pd.read_csv(f'{P31}/01_Numerical_Repair/patient_paired_statistics.csv')
    df.to_csv(f'{FIG}/data_fig5_patient_paired.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    pairs = ['M1_vs_M2', 'M1_vs_M3', 'M1_vs_LR_A']
    cols = [C['blue'], C['aqua'], C['orange']]
    names = ['M1 - M2', 'M1 - M3', 'M1 - LR-A']
    y = 0
    yticks, ylabels = [], []
    for pair, col, nm in zip(pairs, cols, names):
        sub = df[df.pair == pair].sort_values('seed')
        for _, r in sub.iterrows():
            lo = r.bootstrap_lo_pp
            hi = r.bootstrap_hi_pp
            d = r.patient_equal_diff_pp
            sig = r.wilcoxon_p < 0.05
            ax.plot([lo, hi], [y, y], color=col, linewidth=2.0,
                    solid_capstyle='round')
            ax.scatter([d], [y], s=26, color=col, zorder=5, linewidths=0)
            if sig:
                ax.text(hi + 0.03, y, '*', fontsize=10, color=C['ink'], va='center')
            yticks.append(y)
            ylabels.append(f'{nm} s{r.seed}')
            y -= 1
    ax.axvline(0, color=C['ink2'], linewidth=1.0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7.5)
    ax.set_xlabel('Patient-averaged paired accuracy difference (pp)')
    ax.grid(axis='x', linewidth=0.6)
    ax.set_title('Patient-level paired differences (37 patients, '
                 'bootstrap 95% CI; * nominal p<0.05)')
    fig.tight_layout()
    save(fig, 'fig31_5_patient_paired_forest')


# ----------------------------------------------------------------------
# Fig 6: enrichment summary
# ----------------------------------------------------------------------
def fig6():
    df = pd.read_csv(f'{P31}/03_Enrichment_Reanalysis/'
                     'enrichment_summary_by_class_branch.csv')
    df.to_csv(f'{FIG}/data_fig6_enrichment_summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    cls = sorted(df.cls.unique())
    x = np.arange(len(cls))
    sh = [df[(df.cls == c) & (df.branch == 'shared')].n_sig_F3.iloc[0]
          for c in cls]
    sp = [df[(df.cls == c) & (df.branch == 'spec')].n_sig_F3.iloc[0]
          for c in cls]
    ax.bar(x - 0.19, sh, width=0.38, color=C['blue'], label='Shared branch')
    ax.bar(x + 0.19, sp, width=0.38, color=C['orange'], label='Specific branch')
    ax.set_xticks(x)
    ax.set_xticklabels(cls)
    ax.set_xlabel('Candidate label class')
    ax.set_ylabel('Significant terms (all-family BH, q<0.05)')
    ax.legend()
    ax.grid(axis='y', linewidth=0.6)
    ax.set_title('GO/KEGG/Reactome enrichment after canonical-33 removal')
    fig.tight_layout()
    save(fig, 'fig31_6_enrichment_summary')


# ----------------------------------------------------------------------
# Fig 7: cross-platform gate + internals
# ----------------------------------------------------------------------
def fig7():
    g = pd.read_csv(f'{P31}/05_External_Validation/crossplatform_gate_summary.csv')
    z = pd.read_csv(f'{P31}/05_External_Validation/'
                    'crossplatform_internals_summary.csv')
    g.to_csv(f'{FIG}/data_fig7_crossplatform_gate.csv', index=False)
    z.to_csv(f'{FIG}/data_fig7_crossplatform_internals.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    sub = g[g.model == 'exp5_M1_3fold'].groupby('target').gate_mean.mean()
    order = ['cosmx', 'xenium', 'merscope']
    vals = [0.1550, sub.get('xenium', 0), sub.get('merscope', 0)]
    axes[0].bar(order, vals, color=[C['blue'], C['orange'], C['aqua']],
                width=0.55, linewidth=0)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8.5)
    axes[0].set_ylabel('Mean cell-level gate g')
    axes[0].set_title('Gate response to zeroed specific input '
                      '(exp5 M1, 3 folds)')
    axes[0].grid(axis='y', linewidth=0.6)
    axes[0].set_ylim(0, 0.85)
    zz = z[z.target == 'xenium']
    axes[1].bar(['|z_shared|', '|z_spec|'],
                [zz.z_shared_norm_mean.mean(), zz.z_spec_norm_mean.mean()],
                color=[C['blue'], C['orange']], width=0.45, linewidth=0)
    axes[1].set_ylabel('Mean embedding norm')
    axes[1].set_title('Xenium: branch norms with zeroed specific input\n'
                      '(z_spec = learned constant, not zero)')
    axes[1].grid(axis='y', linewidth=0.6)
    fig.suptitle('Cross-platform behavior of the frozen checkpoint', y=1.02)
    fig.tight_layout()
    save(fig, 'fig31_7_crossplatform_gate')


# ----------------------------------------------------------------------
# Fig 8: bhuva region lift
# ----------------------------------------------------------------------
def fig8():
    df = pd.read_csv(f'{P31}/05_External_Validation/bhuva_region_recompute.csv')
    df.to_csv(f'{FIG}/data_fig8_bhuva_region_lift.csv', index=False)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    sub = df[~df.lift.isna()].copy()
    order = ['Immune cells', 'Stroma', 'Normal ducts', 'DCIS',
             'Invasive Tumor']
    x = np.arange(len(order) * 2)
    xidc = np.arange(0, len(order) * 2, 2)
    xilc = np.arange(1, len(order) * 2, 2)
    vals_idc, vals_ilc = [], []
    for d in order:
        vals_idc.append(sub[(sub.sample == 'IDC') & (sub.domain == d)]
                        .lift.iloc[0] if ((sub.sample == 'IDC') &
                                          (sub.domain == d)).any() else np.nan)
        vals_ilc.append(sub[(sub.sample == 'ILC') & (sub.domain == d)]
                        .lift.iloc[0] if ((sub.sample == 'ILC') &
                                          (sub.domain == d)).any() else np.nan)
    cols_idc = [C['blue'] if v >= 0 else C['red'] for v in vals_idc]
    cols_ilc = [C['aqua'] if v >= 0 else C['red'] for v in vals_ilc]
    ax.bar(xidc, vals_idc, width=0.7, color=cols_idc, label='IDC',
           linewidth=0)
    ax.bar(xilc, vals_ilc, width=0.7, color=cols_ilc, label='ILC',
           linewidth=0)
    ax.axhline(0, color=C['ink2'], linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([order[i // 2] if i % 2 == 0 else ''
                        for i in range(len(order) * 2)], fontsize=7)
    ax.set_ylabel('Region mass - global prior (lift)')
    ax.set_title('Expected-lineage mass lift over global prior '
                 '(Bhuva external sections)')
    ax.legend()
    ax.grid(axis='y', linewidth=0.6)
    fig.tight_layout()
    save(fig, 'fig31_8_bhuva_region_lift')


# ----------------------------------------------------------------------
# Fig 9: LR-A vs M1 two calibers
# ----------------------------------------------------------------------
def fig9():
    df = pd.read_csv(f'{P31}/01_Numerical_Repair/lr_variants_recompute.csv')
    df.to_csv(f'{FIG}/data_fig9_lr_variants.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    models = ['LR_B', 'LR_C', 'M8', 'M9', 'M1', 'LR_A']
    vals = [df[df.model == m].eq_acc_mean.iloc[0] for m in models]
    cols = [C['orange'], C['magenta'], C['violet'], C['yellow'],
            C['blue'], C['aqua']]
    ax.bar(models, vals, color=cols, width=0.6, linewidth=0)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f'{v:.3f}', ha='center', fontsize=8)
    ax.set_ylabel('Accuracy (equal-fold mean)')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', linewidth=0.6)
    ax.set_title('Closed-set accuracy: M1 vs baselines '
                 '(LR-A exceeds M1 by +0.2539 pp equal-fold)')
    fig.tight_layout()
    save(fig, 'fig31_9_baseline_accuracy')


def main():
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
    print('CHARTS_DONE', flush=True)


if __name__ == '__main__':
    main()
