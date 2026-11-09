#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验6.5（9.5.1）：M1 Integrated Gradients 基因归因（手工实现，无新依赖）。

- 模型：outputs/exp6/m1_s42/（规范协议 log1p/15ep，3 折）
- 每折对测试集每类抽样 150 细胞，对预测类 logit 计算 IG（基线=零向量，n_steps=32）
- 输出每类每分支的基因平均 |IG| 排名（共享 108 + 专属 892）
"""
import os
import sys
import numpy as np
import yaml
import torch

sys.path.insert(0, '/home/hst/research/openset_st_cf')
sys.path.insert(0, '/home/hst/research/openset_st_cf/scripts/experiment5')
from train5 import build_data5
from src.models.experiment6.model6 import M1Model6
from types import SimpleNamespace

OUT = '/home/hst/research/openset_st_cf/outputs/exp6'
FINE_NAMES = ['a', 'b', 'c', 'd', 'e', 'g', 'i', 'j']
N_PER_CLASS = 150
N_STEPS = 32
SEED = 42


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


def ig_attribution(model, xs, xp, target, n_steps=N_STEPS):
    """对 (xs, xp) 在目标 logit 上做 IG。返回 (attr_shared, attr_spec)。"""
    xs = xs.clone().requires_grad_(True)
    xp = xp.clone().requires_grad_(True)
    base_s = torch.zeros_like(xs)
    base_p = torch.zeros_like(xp)
    alphas = torch.linspace(0, 1, n_steps + 1, device=xs.device)[1:]
    total_s = torch.zeros_like(xs)
    total_p = torch.zeros_like(xp)
    for a in alphas:
        xs_in = base_s + a * (xs - base_s)
        xp_in = base_p + a * (xp - base_p)
        out = model(xs_in, xp_in)
        logit = out['logits_fine'][torch.arange(xs.shape[0], device=xs.device), target]
        grads = torch.autograd.grad(logit.sum(), (xs_in, xp_in))
        total_s += grads[0]
        total_p += grads[1]
    attr_s = ((xs - base_s) * total_s / n_steps).detach()
    attr_p = ((xp - base_p) * total_p / n_steps).detach()
    return attr_s, attr_p


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = build_data5(smoke=False)
    cfg = make_cfg()
    shared_genes = data['shared_genes']
    spec_genes = data['spec_genes']
    Xs_t = torch.from_numpy(data['X_shared']).float().to(device)
    Xp_t = torch.from_numpy(data['X_specific']).float().to(device)
    yf = data['y_fine']
    folds = data['folds']
    all_attr = {c: dict(s=[], p=[]) for c in FINE_NAMES}
    rng = np.random.RandomState(SEED)
    for f in [1, 2, 3]:
        model = M1Model6(cfg, input_transform='log1p').to(device)
        ckpt = os.path.join(OUT, 'm1_s42', 'model_fold%d.pt' % f)
        model.load_state_dict(torch.load(ckpt, map_location=device)['model'])
        model.eval()
        te = np.where(folds == f)[0]
        for ci, nm in enumerate(FINE_NAMES):
            cand = te[yf[te] == ci]
            if len(cand) == 0:
                continue
            sub = rng.choice(cand, size=min(N_PER_CLASS, len(cand)), replace=False)
            for b in range(0, len(sub), 64):
                bidx = sub[b:b + 64]
                xs = Xs_t[bidx]
                xp = Xp_t[bidx]
                with torch.no_grad():
                    out = model(xs, xp)
                    pred = out['logits_fine'].argmax(1)
                a_s, a_p = ig_attribution(model, xs, xp, pred)
                all_attr[nm]['s'].append(a_s.abs().cpu().numpy())
                all_attr[nm]['p'].append(a_p.abs().cpu().numpy())
        print('fold%d IG done' % f, flush=True)
    # 汇总
    rows = []
    for ci, nm in enumerate(FINE_NAMES):
        mS = np.concatenate(all_attr[nm]['s']).mean(0)
        mP = np.concatenate(all_attr[nm]['p']).mean(0)
        top_s = np.argsort(-mS)
        top_p = np.argsort(-mP)
        rows.append(dict(ftype=nm, n_cells=len(all_attr[nm]['s']),
                         top_shared=';'.join(shared_genes[i] for i in top_s[:30]),
                         top_shared_val=';'.join('%.4f' % mS[i] for i in top_s[:30]),
                         top_spec=';'.join(spec_genes[i] for i in top_p[:30]),
                         top_spec_val=';'.join('%.4f' % mP[i] for i in top_p[:30])))
        print('%s: top shared: %s' % (nm, ' '.join(shared_genes[i] for i in top_s[:12])))
        print('%s: top spec:   %s' % (nm, ' '.join(spec_genes[i] for i in top_p[:12])))
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'exp6_ig_top_genes.csv'), index=False)
    np.savez_compressed(os.path.join(OUT, 'exp6_ig_attr.npz'),
                        shared_genes=np.array(shared_genes), spec_genes=np.array(spec_genes),
                        **{('ig_s_%s' % nm): np.concatenate(all_attr[nm]['s']).mean(0)
                           for nm in FINE_NAMES},
                        **{('ig_p_%s' % nm): np.concatenate(all_attr[nm]['p']).mean(0)
                           for nm in FINE_NAMES})
    print('IG_DONE')


if __name__ == '__main__':
    main()
