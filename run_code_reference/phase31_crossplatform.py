# -*- coding: utf-8 -*-
"""任务 4.1：Xenium / MERSCOPE 跨平台重算——保存分支内部状态。

协议（复现 scripts/experiment5/cross_platform5.py + 增加内部状态落盘）：
- 核心：tri_run_107（三平台 2023 乳腺批交集）；MERSCOPE 另报 quality strata
- 输入：目标平台 108 共享基因按 CosMx 字母序重排（记录每平台缺失基因）；
  专属分支输入全部置零（Xp=0）
- 模型：(a) exp5 M1 三折（冻结口径，锚点 gate 0.708/0.513）；
       (b) exp6 m1_s42 fold1（规范模型敏感性）
- 落盘每细胞：z_shared、z_spec、gate、z（融合）、proba_f、core、n_matched_genes
- 解释缺失输入响应：z_spec 统计量（置零输入经专属编码器输出非常量——
  BN 平移+ReLU 后的常量向量）、gate 分布、z_shared vs z_spec 范数
"""
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase31_infra import ROOT

OUT = ROOT + '/paper_preparation/09_Phase31_Results/05_External_Validation'
E = ROOT + '/data/raw/ist_benchmarking/extracted'
M = ROOT + '/data/metadata'
CTRL = re.compile(r'blank|negative|negcontrol|systemcontrol', re.I)
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + '/scripts/experiment5')


def load_platform(fp, cores, genes_wanted):
    a = ad.read_h5ad(fp, backed='r')
    all_genes = np.asarray(a.var_names, dtype=str)
    idx = sorted(int(np.where(all_genes == g)[0][0]) for g in genes_wanted
                 if g in set(all_genes))
    core_arr = np.asarray(a.obs['core'].astype(str), dtype=str)
    keep = np.isin(core_arr, [str(c) for c in cores])
    X = sparse.csr_matrix(a.X[:, idx])[keep]
    a.file.close()
    return X, core_arr[keep], [all_genes[i] for i in idx]


def main():
    os.makedirs(OUT, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    audit = json.load(open(ROOT + '/data/interim/exp2_core_audit_sets.json'))
    trio = sorted(int(c) for c in audit['tri_run_107'])
    common = sorted(l.strip() for l in open(M + '/common_genes.tsv')
                    if l.strip() != 'gene')

    # ---- exp5 M1 三折（冻结口径复现）----
    raw = yaml.safe_load(open(ROOT + '/configs/experiment5.yaml'))
    merged = {}
    for sec in ['data', 'model', 'train']:
        merged.update(raw[sec])
    for k, v in raw.items():
        if k not in ('data', 'model', 'train', 'variants'):
            merged[k] = v
    merged_v = dict(merged)
    merged_v.update(raw['variants']['M1'])
    from types import SimpleNamespace
    cfg5 = SimpleNamespace(**merged_v)
    from train5 import load_p_g
    pg_map_all = {}
    for f in [1, 2, 3]:
        pg_map_all.update(load_p_g(f))
    p_g_shared = np.array([min(pg_map_all.get(g, 1.0), 1.0) for g in common],
                          dtype=np.float32)
    import src.models.experiment5.model5 as model5_mod

    # ---- exp6 M1 fold1（规范模型）----
    from src.models.experiment6.model6 import M1Model6
    from train6 import make_cfg as make_cfg6
    cfg6 = make_cfg6('base', False, False)

    meta_rows, gate_rows, zspec_rows = [], [], []
    for tgt_name, tgt_fp in [('xenium', E + '/2023_xenium_breast_tTMA1.h5ad'),
                             ('merscope', E + '/2023_merscope_breast_tTMA1.h5ad')]:
        Xt, core_t, genes_t = load_platform(tgt_fp, trio, common)
        order = [genes_t.index(g) for g in common if g in genes_t]
        Xt = Xt[:, order]
        genes_t = [genes_t[i] for i in order]
        n_matched = len(genes_t)
        missing = [g for g in common if g not in genes_t]
        print(f'{tgt_name}: matched {n_matched}/108 shared genes; missing: {missing}',
              flush=True)
        # merscope 质量分层
        if tgt_name == 'merscope':
            qc = pd.read_csv(f'{M}/qc_core_strata.csv', dtype=str)
            qc = qc[qc.file == '2023_merscope_breast_tTMA1']
            qc_map = dict(zip(qc.core, qc.stratum))
        for c in trio:
            m = core_t == str(c)
            if m.sum() < 20:
                continue
            Xc = Xt[m].toarray().astype(np.float32)
            Xs = torch.from_numpy(Xc).to(device)
            Xp = torch.zeros(Xc.shape[0], cfg5.n_specific, device=device)
            stratum = qc_map.get(str(c), 'NA') if tgt_name == 'merscope' else 'NA'
            # ---- exp5 M1 三折 ----
            g_all, proba_all = [], []
            for f in [1, 2, 3]:
                ckpt = f'{ROOT}/outputs/exp5/M1/model_fold{f}.pt'
                model5 = model5_mod.AdaptiveGatingModel(
                    cfg5, p_g_shared, np.ones(cfg5.n_specific, dtype=np.float32))
                model5.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
                model5.to(device).eval()
                with torch.no_grad():
                    o = model5(Xs, Xp, extra=None)
                    g_all.append(o['gate'].squeeze(-1).cpu().numpy())
                    proba_all.append(torch.softmax(o['logits_fine'], -1).cpu().numpy())
                del model5
            gate_rows.append(dict(target=tgt_name, core=c, stratum=stratum,
                                  n_cells=int(m.sum()), model='exp5_M1_3fold',
                                  gate_mean=float(np.mean(np.concatenate(g_all))),
                                  n_genes_matched=n_matched))
            # ---- exp6 M1 fold1（保存完整内部状态）----
            model6 = M1Model6(cfg6, input_transform='log1p').to(device)
            ck6 = torch.load(f'{ROOT}/outputs/exp6/m1_s42/model_fold1.pt',
                             map_location='cpu', weights_only=False)
            model6.load_state_dict(ck6['model'])
            model6.eval()
            zs_c, zp_c, g_c, z_c, pr_c = [], [], [], [], []
            with torch.no_grad():
                for b in range(0, Xc.shape[0], 4096):
                    o = model6(Xs[b:b + 4096], Xp[b:b + 4096])
                    zs_c.append(o['z_shared'].cpu().numpy())
                    zp_c.append(o['z_spec'].cpu().numpy())
                    g_c.append(o['gate'].cpu().numpy().reshape(-1))
                    z_c.append(o['z'].cpu().numpy())
                    pr_c.append(torch.softmax(o['logits_fine'], -1).cpu().numpy())
            zs_c = np.concatenate(zs_c)
            zp_c = np.concatenate(zp_c)
            g_c = np.concatenate(g_c)
            z_c = np.concatenate(z_c)
            pr_c = np.concatenate(pr_c)
            np.savez_compressed(
                f'{OUT}/exp6_m1_fold1_{tgt_name}_core{c}_internals.npz',
                z_shared=zs_c, z_spec=zp_c, gate=g_c, z=z_c, proba_f=pr_c,
                core=np.array([str(c)] * len(g_c)))
            zs_n = np.linalg.norm(zs_c, axis=1)
            zp_n = np.linalg.norm(zp_c, axis=1)
            zspec_rows.append(dict(target=tgt_name, core=c, stratum=stratum,
                                   n_cells=int(m.sum()),
                                   z_spec_mean=float(zp_c.mean()),
                                   z_spec_sd=float(zp_c.std()),
                                   z_spec_norm_mean=float(zp_n.mean()),
                                   z_spec_norm_sd=float(zp_n.std()),
                                   z_shared_norm_mean=float(zs_n.mean()),
                                   gate_mean=float(g_c.mean()),
                                   gate_sd=float(g_c.std()),
                                   gate_min=float(g_c.min()),
                                   gate_max=float(g_c.max()),
                                   gate_pct_gt_0_5=float((g_c > 0.5).mean()) * 100))
            gate_rows.append(dict(target=tgt_name, core=c, stratum=stratum,
                                  n_cells=int(m.sum()), model='exp6_m1_fold1',
                                  gate_mean=float(g_c.mean()),
                                  n_genes_matched=n_matched))
            print(f'{tgt_name} core{c}: n={m.sum()} exp5gate3f='
                  f'{np.mean(np.concatenate(g_all)):.4f} exp6gate='
                  f'{g_c.mean():.4f} |z_spec|={zp_n.mean():.3f} |z_shared|='
                  f'{zs_n.mean():.3f}', flush=True)
            del model6
            torch.cuda.empty_cache()

    df_gate = pd.DataFrame(gate_rows)
    df_gate.to_csv(f'{OUT}/crossplatform_gate_summary.csv', index=False)
    pd.DataFrame(zspec_rows).to_csv(f'{OUT}/crossplatform_internals_summary.csv',
                                    index=False)
    print('=== exp5 M1 3-fold gate 均值（锚点 xenium 0.708 / merscope 0.513）===')
    print(df_gate[df_gate.model == 'exp5_M1_3fold'].groupby('target').gate_mean
          .mean().round(4).to_string(), flush=True)
    print('CROSSPLATFORM_DONE', flush=True)


if __name__ == '__main__':
    main()
