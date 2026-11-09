#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验5：跨平台间接评估（Xenium/MERSCOPE 无金标准，仅间接指标）。
指标：107 共同核心上预测组成 vs CosMx 参考的 JS 距离、共享分支 gate 行为、标志证据一致性。"""
import os
import re
import sys
import json
import numpy as np
import pandas as pd
import anndata as ad
import yaml
import torch
from scipy import sparse
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, '/home/hst/research/openset_st_cf')
sys.path.insert(0, '/home/hst/research/openset_st_cf/scripts/experiment5')
from train5 import load_p_g

OUT = '/home/hst/research/openset_st_cf/outputs/exp5'
E = '/home/hst/research/openset_st_cf/data/raw/ist_benchmarking/extracted'
M = '/home/hst/research/openset_st_cf/data/metadata'
CTRL = re.compile(r'blank|negative|negcontrol|systemcontrol', re.I)
raw = yaml.safe_load(open('/home/hst/research/openset_st_cf/configs/experiment5.yaml'))
merged = {}
for sec in ['data', 'model', 'train']:
    merged.update(raw[sec])
for k, v in raw.items():
    if k not in ('data', 'model', 'train', 'variants'):
        merged[k] = v

audit = json.load(open('/home/hst/research/openset_st_cf/data/interim/exp2_core_audit_sets.json'))
trio = sorted(int(c) for c in audit['tri_run_107'])
common = sorted(l.strip() for l in open(M + '/common_genes.tsv') if l.strip() != 'gene')


def load_platform(fp, cores, genes_wanted):
    a = ad.read_h5ad(fp, backed='r')
    all_genes = np.asarray(a.var_names, dtype=str)
    idx = sorted(int(np.where(all_genes == g)[0][0]) for g in genes_wanted if g in set(all_genes))
    core_arr = np.asarray(a.obs['core'].astype(str), dtype=str)
    keep = np.isin(core_arr, [str(c) for c in cores])
    X = a.X[:, idx]
    X = sparse.csr_matrix(X)[keep]
    a.file.close()
    return X, core_arr[keep], [all_genes[i] for i in idx]


# CosMx 参考组成（M1 在 CosMx 测试折的预测按核心聚合）
rows = []
for variant in ['M1', 'M4', 'M5']:
    import src.models.experiment5.model5 as model5_mod
    from types import SimpleNamespace
    merged_v = dict(merged)
    merged_v.update(raw['variants'][variant])
    cfg = SimpleNamespace(**merged_v)
    pg_map_all = {}
    for f in [1, 2, 3]:
        pg_map_all.update(load_p_g(f))
    p_g_shared = np.array([min(pg_map_all.get(g, 1.0), 1.0) for g in common], dtype=np.float32)
    for tgt_name, tgt_fp in [('xenium', E + '/2023_xenium_breast_tTMA1.h5ad'),
                             ('merscope', E + '/2023_merscope_breast_tTMA1.h5ad')]:
        # 目标平台共同基因（重排到字母序 common 顺序，与模型共享分支一致）
        Xt, core_t, genes_t = load_platform(tgt_fp, trio, common)
        order = [genes_t.index(g) for g in common if g in genes_t]
        Xt = Xt[:, order]
        genes_t = [genes_t[i] for i in order]
        for f in [1, 2, 3]:
            ckpt = os.path.join(OUT, variant, 'model_fold%d.pt' % f)
            if not os.path.exists(ckpt):
                continue
            model = model5_mod.AdaptiveGatingModel(cfg, p_g_shared,
                                                   np.ones(cfg.n_specific, dtype=np.float32))
            model.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
            model.eval()
            # 目标细胞只喂共享分支（专属置零）
            for c in trio:
                m = core_t == str(c)
                if m.sum() < 20:
                    continue
                Xc = Xt[m].toarray().astype(np.float32)
                Xs = torch.from_numpy(Xc)
                Xp = torch.zeros(Xc.shape[0], cfg.n_specific)
                qs = torch.from_numpy((Xc > 0).mean(axis=1).astype(np.float32))
                qp = torch.zeros(Xc.shape[0])
                Kk = torch.zeros(Xc.shape[0])
                with torch.no_grad():
                    extra = None
                    if cfg.gate_mode == 'cell_evidence':
                        extra = torch.stack([qs, qp, Kk], dim=-1)
                    elif cfg.gate_mode == 'platform':
                        extra = torch.stack([torch.zeros(len(qs)), torch.zeros(len(qs))], dim=-1)
                    o = model(Xs, Xp, extra=extra)
                    proba = torch.softmax(o['logits_fine'], -1).numpy()
                    g = o['gate'].squeeze(-1).numpy()
                comp = proba.mean(0)
                rows.append(dict(variant=variant, target=tgt_name, fold=f, core=c,
                                 n_cells=int(m.sum()), gate_mean=float(g.mean()),
                                 comp_a=float(comp[0]), comp_b=float(comp[1]),
                                 comp_c=float(comp[2]), comp_d=float(comp[3]),
                                 comp_e=float(comp[4]), comp_g=float(comp[5]),
                                 comp_i=float(comp[6]), comp_j=float(comp[7])))
        print(variant, tgt_name, 'done', flush=True)
rdf = pd.DataFrame(rows)
rdf.to_csv(os.path.join(OUT, 'exp5_cross_platform.csv'), index=False)
print('\n=== 跨平台汇总 ===')
print('gate 均值（共享分支为主则高）:')
print(rdf.groupby(['variant', 'target']).gate_mean.mean().round(4).to_string())
print('预测组成熵（高=分布更平）:')
comp_cols = [c for c in rdf.columns if c.startswith('comp_')]
rdf['comp_entropy'] = -(rdf[comp_cols].values * np.log(np.clip(rdf[comp_cols].values, 1e-9, 1))).sum(1)
print(rdf.groupby(['variant', 'target']).comp_entropy.mean().round(4).to_string())
print('CROSS_PLATFORM_DONE')
