# -*- coding: utf-8 -*-
"""任务 2.3：分支特异性 GO/KEGG 富集重分析（canonical 33 剔除、分支匹配背景、完整家族）。

口径（与 phase15 审计 enrichment_rerun 一致 + 审稿要求的全家族 BH）：
- 候选：冻结 exp6_ig_attr.npz 每类每分支 top-30，剔除 canonical 33 后【不补位】
- 背景（主口径）：分支匹配背景 = 分支基因 − canonical33（共享 86 / 专属 881）
- 敏感性：全部 1000 模型输入基因（仅 GO_BP + KEGG，注明非分支匹配、审稿指出的问题）
- 库：GO_BP / GO_MF / GO_CC / KEGG / Reactome（本地 MSigDB 2024.1 GMT）
- 检验：gseapy enrichr（超几何）+ BH-FDR
- 家族（全部报告）：
  F1 = 每类×分支×库内 BH（phase15 口径）
  F2 = 每类×分支 跨库合并 BH
  F3 = 全部检验（8类×2分支×5库）一次 BH（审稿"一次覆盖全部家族的 BH"）
- 保存全部显著与不显著结果（完整表）
锚点：共享分支 F3 无显著；专属 b/c/e/i/j F3 仍有显著；ECM structural constituent
     库内 q≈1.9e-5、全家族 q≈9.16e-4。
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import gseapy as gp
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase31_infra import ROOT

OUT = ROOT + '/paper_preparation/09_Phase31_Results/03_Enrichment_Reanalysis'
FINE = ['a', 'b', 'c', 'd', 'e', 'g', 'i', 'j']
TOP_N = 30
GMT_DIR = ROOT + '/data/external/experiment6'
LIBS = {
    'GO_BP': 'c5.go.bp.v2024.1.Hs.symbols.gmt',
    'GO_MF': 'c5.go.mf.v2024.1.Hs.symbols.gmt',
    'GO_CC': 'c5.go.cc.v2024.1.Hs.symbols.gmt',
    'KEGG': 'c2.cp.kegg_medicus.v2024.1.Hs.symbols.gmt',
    'Reactome': 'c2.cp.reactome.v2024.1.Hs.symbols.gmt',
}


def main():
    os.makedirs(OUT, exist_ok=True)
    fs = json.load(open(f'{ROOT}/data/metadata/feature_sets.json'))
    MARK33 = set(fs['label_defining_in_cosmx'])
    assert len(MARK33) == 33

    z = np.load(f'{ROOT}/outputs/exp6/exp6_ig_attr.npz')
    shared = np.asarray(z['shared_genes'], dtype=str)
    spec = np.asarray(z['spec_genes'], dtype=str)
    union = np.array(sorted(set(shared) | set(spec)), dtype=str)

    all_rows = []
    for nm in FINE:
        ig_s = z['ig_s_%s' % nm]
        ig_p = z['ig_p_%s' % nm]
        for branch, ig, genes, bg in [
                ('shared', ig_s, shared, sorted(set(shared) - MARK33)),
                ('spec', ig_p, spec, sorted(set(spec) - MARK33))]:
            order = np.argsort(-ig)
            top = [genes[i] for i in order[:TOP_N]]
            top_after = [g for g in top if g not in MARK33]
            print(f'== {nm}/{branch}: top30 剔除标志 {len(top)-len(top_after)} 个, '
                  f'候选 {len(top_after)} 基因, 背景 {len(bg)} ==', flush=True)
            for lib_name, gmt in LIBS.items():
                try:
                    res = gp.enrichr(gene_list=top_after, gene_sets=GMT_DIR + '/' + gmt,
                                     background=list(bg), outdir=None, cutoff=1.0,
                                     no_plot=True)
                except Exception as e:
                    print(f'[{nm}/{branch}/{lib_name}] 失败: {e}', flush=True)
                    continue
                if res is None or res.results is None or len(res.results) == 0:
                    print(f'[{nm}/{branch}/{lib_name}] no results', flush=True)
                    continue
                r = res.results.copy()
                for _, row in r.iterrows():
                    all_rows.append(dict(cls=nm, branch=branch, library=lib_name,
                                         term=row['Term'], overlap=row['Overlap'],
                                         p=float(row['P-value']), n_genes_candidate=len(top_after),
                                         n_genes_bg=len(bg),
                                         background='branch_minus_33'))
                n_sig = int((r['P-value'] < 0.05).sum())
                print(f'[{nm}/{branch}/{lib_name}] 名义显著 {n_sig}/{len(r)}', flush=True)

    df = pd.DataFrame(all_rows)
    # F1: 每类×分支×库内 BH
    df['fdr_F1_within_lib'] = np.nan
    for (nm, br, lib), idx in df.groupby(['cls', 'branch', 'library']).groups.items():
        df.loc[list(idx), 'fdr_F1_within_lib'] = multipletests(
            df.loc[list(idx), 'p'].values, method='fdr_bh')[1]
    # F2: 每类×分支跨库 BH
    df['fdr_F2_class_branch'] = np.nan
    for (nm, br), idx in df.groupby(['cls', 'branch']).groups.items():
        df.loc[list(idx), 'fdr_F2_class_branch'] = multipletests(
            df.loc[list(idx), 'p'].values, method='fdr_bh')[1]
    # F3: 全部检验一次 BH
    df['fdr_F3_all'] = multipletests(df.p.values, method='fdr_bh')[1]
    df.to_csv(f'{OUT}/enrichment_reanalysis_full.csv', index=False)

    # ---- 汇总 ----
    summary_rows = []
    for (nm, br), sub in df.groupby(['cls', 'branch']):
        summary_rows.append(dict(
            cls=nm, branch=br,
            n_terms=len(sub),
            n_sig_F1=int((sub.fdr_F1_within_lib < 0.05).sum()),
            n_sig_F2=int((sub.fdr_F2_class_branch < 0.05).sum()),
            n_sig_F3=int((sub.fdr_F3_all < 0.05).sum())))
    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(f'{OUT}/enrichment_summary_by_class_branch.csv', index=False)
    print(sdf.to_string(index=False), flush=True)

    # ---- 四个问题 ----
    print('\n=== Q1: F3 显著的身份相关条目（每类×分支）===', flush=True)
    print(df[df.fdr_F3_all < 0.05].groupby(['cls', 'branch']).size().to_string(),
          flush=True)
    print('\n=== Q3: 成纤维 b 类 ECM 条目（F3 显著）===', flush=True)
    b_spec = df[(df.cls == 'b') & (df.branch == 'spec')]
    ecm = b_spec[b_spec.term.str.contains('EXTRACELLULAR|COLLAGEN', case=False, na=False)]
    print(ecm[['term', 'library', 'p', 'fdr_F1_within_lib', 'fdr_F3_all']]
          .sort_values('fdr_F3_all').head(12).to_string(index=False), flush=True)
    print('\n=== Q4: 共享分支免疫(i)/髓系(j) ===', flush=True)
    sh = df[df.branch == 'shared']
    for nm in ['i', 'j']:
        sub = sh[sh.cls == nm]
        print(f'{nm}/shared: n_terms={len(sub)} F3 显著 {int((sub.fdr_F3_all < 0.05).sum())}',
              flush=True)
        if len(sub):
            print(sub.sort_values('p')[['term', 'library', 'p', 'fdr_F3_all']]
                  .head(3).to_string(index=False), flush=True)
    print('ENRICHMENT_DONE', flush=True)


if __name__ == '__main__':
    main()
