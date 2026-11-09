#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验6.6：Bhuva 病理学家区域标注外部验证（M1 跨平台前向）。

- 输入：Bhuva 表达数据（EH8567 处理后）outputs/exp6/bhuva/{idc,ilc}_expr.npz
       + 区域标注 data/external/experiment6/{idc,ilc}.csv（cell→domain）
- 模型：M1 fold1（CosMx 训练），专属分支置零（设计回退路径），108 共同基因 log1p
- 评价：区域级合理性（非细胞型监督精度）：
    Immune cells → P(i或j) 应高；Stroma → P(b)；Invasive Tumor/DCIS/Normal ducts → P(上皮系)；
    Adipose/Cyst/Blood vessels → 无对应类（开放集：max-prob 应低/熵应高）
- 输出：outputs/exp6/bhuva/ 预测 + 汇总 CSV
"""
import os
import sys
import numpy as np
import pandas as pd
import yaml
import torch

sys.path.insert(0, '/home/hst/research/openset_st_cf')
sys.path.insert(0, '/home/hst/research/openset_st_cf/scripts/experiment5')
from train5 import build_data5
from src.models.experiment6.model6 import M1Model6
from types import SimpleNamespace

OUT = '/home/hst/research/openset_st_cf/outputs/exp6'
BHUVAD = '/home/hst/research/openset_st_cf/outputs/exp6/bhuva'
EXT = '/home/hst/research/openset_st_cf/data/external/experiment6'
FINE_NAMES = ['a', 'b', 'c', 'd', 'e', 'g', 'i', 'j']
DOMAIN_GROUPS = {
    'Immune cells': ('immune', ['i', 'j']),
    'Stroma': ('fibroblast', ['b']),
    'Invasive Tumor': ('epithelial', ['a', 'c', 'd', 'e']),
    'DCIS': ('epithelial', ['a', 'c', 'd', 'e']),
    'Normal ducts': ('epithelial', ['a', 'c', 'd', 'e']),
    'Adipose tissue (fat)': ('open', []),
    'Cyst': ('open', []),
    'Blood vessels': ('open', []),
}


def make_cfg():
    raw = yaml.safe_load(open('/home/hst/research/openset_st_cf/configs/experiment5.yaml'))
    merged = {}
    for sec in ['data', 'model', 'train']:
        merged.update(raw[sec])
    for k, v in raw.items():
        if k not in ('data', 'model', 'train', 'variants'):
            merged[k] = v
    merged['gate_mode'] = 'base'
    merged['use_specific'] = True
    merged['use_gene_weight'] = False
    merged['zero_shared'] = False
    return SimpleNamespace(**merged)


def main():
    data = build_data5(smoke=False)
    shared_genes = data['shared_genes']
    cfg = make_cfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = M1Model6(cfg, input_transform='log1p').to(device)
    model.load_state_dict(torch.load(os.path.join(OUT, 'm1_s42', 'model_fold1.pt'),
                                     map_location=device)['model'])
    model.eval()
    rows = []
    from scipy.io import mmread
    from scipy import sparse
    for sample in ['IDC', 'ILC']:
        Xm = mmread(os.path.join(BHUVAD, '%s_X.mtx' % sample))  # cells × 541 genes
        cell_ids = [l.strip() for l in open(os.path.join(BHUVAD, '%s_cells.txt' % sample))]
        dom_full = [l.strip() for l in open(os.path.join(BHUVAD, '%s_domain.txt' % sample))]
        genes_all = [l.strip() for l in open(os.path.join(BHUVAD, '%s_genes.txt' % sample))] if os.path.exists(os.path.join(BHUVAD, '%s_genes.txt' % sample)) else None
        # 基因顺序：聚合时按 common 字母序 → 直接对 108 共同基因列取
        Xm = sparse.csr_matrix(Xm)
        # 541 基因列 = common(108) 顺序？聚合脚本用的是 common 排序，故前 108 列即 common 字母序
        X = Xm[:, :108].toarray().astype(np.float32)
        dom = np.array([None if d == 'NA' else d for d in dom_full], dtype=object)
        has_ann = dom != None  # noqa: E711
        print('%s: n=%d 有区域标注=%d (%.1f%%)' % (sample, X.shape[0], has_ann.sum(),
                                                  has_ann.mean()), flush=True)
        Xt = torch.from_numpy(X).float().to(device)
        probas, gates = [], []
        with torch.no_grad():
            for b in range(0, len(Xt), 8192):
                o = model(Xt[b:b + 8192], torch.zeros(len(Xt[b:b + 8192]), 892, device=device))
                probas.append(torch.softmax(o['logits_fine'], -1).cpu().numpy())
                gates.append(o['gate'].squeeze(-1).cpu().numpy())
        proba = np.concatenate(probas)
        gate = np.concatenate(gates)
        pred = proba.argmax(1)
        ent = -(proba * np.log(proba + 1e-12)).sum(1)
        np.savez_compressed(os.path.join(BHUVAD, '%s_pred.npz' % sample),
                            proba=proba, gate=gate, pred=pred, entropy=ent,
                            domain=dom.astype(str))
        for dom_name, (grp, expected) in DOMAIN_GROUPS.items():
            m = has_ann & (dom == dom_name)
            if m.sum() == 0:
                continue
            exp_idx = [FINE_NAMES.index(x) for x in expected] if grp != 'open' else []
            if grp == 'open':
                rows.append(dict(sample=sample, domain=dom_name, group=grp, n=int(m.sum()),
                                 expected_hit=np.nan, max_prob_mean=float(proba[m].max(1).mean()),
                                 entropy_mean=float(ent[m].mean()), gate_mean=float(gate[m].mean())))
            else:
                hit = float(proba[m][:, exp_idx].sum(1).mean())
                rows.append(dict(sample=sample, domain=dom_name, group=grp, n=int(m.sum()),
                                 expected_hit=hit,
                                 max_prob_mean=float(proba[m].max(1).mean()),
                                 entropy_mean=float(ent[m].mean()), gate_mean=float(gate[m].mean())))
            print('  %s: n=%d 期望类命中率=%.4f maxprob=%.4f entropy=%.4f gate=%.4f' % (
                dom_name, m.sum(),
                (float(proba[m][:, exp_idx].sum(1).mean()) if grp != 'open' else np.nan),
                float(proba[m].max(1).mean()), float(ent[m].mean()), float(gate[m].mean())),
                flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(BHUVAD, 'bhuva_domain_summary.csv'), index=False)
    print('BHUVAD_DONE')


if __name__ == '__main__':
    main()
