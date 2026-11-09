#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验5：自适应门控模型训练（患者级 3 折，串行 GPU，断点续训，GPU 日志）。
用法: python train5.py --config configs/experiment5.yaml [--smoke] [--fold 1] [--variant M5]
"""
import os
import re
import sys
import time
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import yaml
import torch
import torch.nn as nn
from scipy import sparse

sys.path.insert(0, '/home/hst/research/openset_st_cf')
E = '/home/hst/research/openset_st_cf/data/raw/ist_benchmarking/extracted'
M = '/home/hst/research/openset_st_cf/data/metadata'
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MARKERS = {
    'Alveolar cells': ['PIGR', 'KIT', 'SCGB2A1'],
    'Basal cells': ['KRT5', 'KRT14', 'DST', 'MYLK'],
    'Epithelial cells': ['EPCAM', 'KRT7', 'KRT8', 'KRT18', 'KRT19'],
    'Fibroblast cells': ['LUM', 'DCN', 'MMP2', 'CXCL12'],
    'Hormone-sensing cells': ['ESR1', 'PGR', 'AR', 'FOXA1'],
    'Immune cells': ['PTPRC', 'CD3E', 'CD79A', 'MS4A1'],
    'Myeloid cells': ['CD68', 'LYZ', 'CD163', 'AIF1'],
    'Vascular & Lymphatic cells': ['PECAM1', 'VWF', 'CLDN5'],
}


def build_data5(smoke=False):
    ctrl = re.compile(r'blank|negative|negcontrol|systemcontrol', re.I)
    fp = E + '/2023_cosmx_multitissue_tTMA1.h5ad'
    a = ad.read_h5ad(fp, backed='r')
    genes = np.asarray(a.var_names, dtype=str)
    obs = a.obs
    sca = pd.read_csv(M + '/sceneA_final_manifest.csv')
    elig = set(sca[sca.eligible == 1].core.astype(str))
    man = pd.read_csv(M + '/core_split_manifest.csv')
    fold_map = dict(zip(man[man.tma == 'tTMA1'].core.astype(str), man[man.tma == 'tTMA1'].fold))
    labcol = [c for c in obs.columns if 'nbclust' in c and 'cluster' in c and 'posterior' not in c][0]
    cats = list(obs[labcol].cat.categories)
    lab_arr = np.asarray(obs[labcol].cat.codes)
    mapping = pd.read_csv(M + '/celltype_mapping.tsv', sep='\t')
    FINE = mapping[mapping.unmapped_label != 'low_confidence'].fine_label.tolist()
    fine2coarse = dict(zip(mapping.fine_label, mapping.coarse_label))
    y = np.array([cats[c] if cats[c] in FINE else 'AMB' for c in lab_arr], dtype=object)
    core_arr = np.asarray(obs['core'].astype(str), dtype=str)
    keep = np.array([(c in elig) and (y[i] != 'AMB') for i, c in enumerate(core_arr)])
    counts = np.asarray(obs['transcript_counts'], dtype=float)[keep]
    real_idx = sorted(i for i, g in enumerate(genes) if not ctrl.search(g))
    X = sparse.csr_matrix(a.X[:, real_idx])
    a.file.close()
    X = X[keep]
    genes_real = [genes[i] for i in real_idx]
    common = sorted(l.strip() for l in open(M + '/common_genes.tsv') if l.strip() != 'gene')
    gidx = {g: i for i, g in enumerate(genes_real)}
    shared_idx = sorted(gidx[g] for g in common if g in gidx)
    spec_idx = sorted(i for i in range(len(genes_real)) if i not in set(shared_idx))
    Xs = X[:, shared_idx].toarray().astype(np.float32)
    Xp = X[:, spec_idx].toarray().astype(np.float32)
    y_f = np.array([FINE.index(v) for v in y[keep]])
    coarse_names = sorted(set(fine2coarse[v] for v in FINE))
    y_c = np.array([coarse_names.index(fine2coarse[v]) for v in y[keep]])
    folds = np.array([fold_map.get(c, 0) for c in core_arr[keep]], dtype=int)
    marker_idx = [[gidx[g] for g in v if g in gidx] for v in MARKERS.values()]
    K = np.zeros(len(y_f), dtype=np.float32)
    Xd = X.toarray().astype(np.float32)
    for b in range(0, len(y_f), 20000):
        sub = Xd[b:b + 20000]
        scores = np.stack([(sub[:, idx] > 0).mean(axis=1) for idx in marker_idx], axis=1)
        top2 = np.sort(scores, axis=1)[:, ::-1][:, :2]
        K[b:b + 20000] = top2[:, 0] - top2[:, 1]
    # 每细胞证据覆盖
    q_shared = (Xs > 0).mean(axis=1).astype(np.float32)
    q_spec = (Xp > 0).mean(axis=1).astype(np.float32)
    core_arr = core_arr[keep]
    if smoke:
        rng = np.random.RandomState(SEED)
        sub = rng.choice(len(y_f), size=6000, replace=False)
        Xs, Xp, y_f, y_c, folds, K, q_shared, q_spec, core_arr = (
            Xs[sub], Xp[sub], y_f[sub], y_c[sub], folds[sub], K[sub],
            q_shared[sub], q_spec[sub], core_arr[sub])
    return dict(X_shared=Xs, X_specific=Xp, y_fine=y_f, y_coarse=y_c,
                folds=folds, K=K, q_shared=q_shared, q_spec=q_spec, core_arr=core_arr,
                counts=counts, fine_names=FINE, coarse_names=coarse_names,
                shared_genes=[genes_real[i] for i in shared_idx],
                spec_genes=[genes_real[i] for i in spec_idx])


def load_p_g(fold):
    """从 4.1 参数读取基因级保留概率（cosmx→merscope，held_out_fold=fold）。"""
    P = M + '/experiment4'
    pr = pd.read_csv(P + '/perturbation_parameter_manifest.csv')
    pr = pr[pr.held_out_fold == fold]
    return dict(zip(pr[pr.target == 'merscope'].gene, pr[pr.target == 'merscope'].p_g))


def train_fold5(cfg, data, fold, smoke=False):
    from src.models.experiment5 import model5
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    tr = (data['folds'] != fold) & (data['folds'] > 0)
    te = data['folds'] == fold
    idx_tr = np.where(tr)[0]
    core_tr = data['core_arr'][tr]
    uniq = np.unique(core_tr)
    rng = np.random.RandomState(SEED)
    rng.shuffle(uniq)
    n_val = max(1, int(0.2 * len(uniq)))
    val_cores = set(uniq[:n_val])
    val_mask = np.array([c in val_cores for c in core_tr])
    idx_fit = idx_tr[~val_mask]
    idx_val = idx_tr[val_mask]
    idx_te = np.where(te)[0]
    if smoke:
        idx_fit = idx_fit[:4000]
        idx_val = idx_val[:1000]
        idx_te = idx_te[:2000]
    # 基因检测概率（共享+专属，cosmx→merscope）
    pg_map = load_p_g(fold)
    p_g_shared = np.array([min(pg_map.get(g, 1.0), 1.0) for g in data['shared_genes']], dtype=np.float32)
    p_g_spec = np.array([min(pg_map.get(g, 1.0), 1.0) for g in data['spec_genes']], dtype=np.float32)
    model = model5.AdaptiveGatingModel(cfg, p_g_shared, p_g_spec).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    n_batch = cfg.batch_size
    epochs = cfg.epochs
    Xs_t = torch.from_numpy(data['X_shared']).float().to(device)
    Xp_t = torch.from_numpy(data['X_specific']).float().to(device)
    yf_t = torch.from_numpy(data['y_fine']).long().to(device)
    yc_t = torch.from_numpy(data['y_coarse']).long().to(device)
    Kt = torch.from_numpy(data['K']).float().to(device)
    Qs_t = torch.from_numpy(data['q_shared']).float().to(device)
    Qp_t = torch.from_numpy(data['q_spec']).float().to(device)
    # 平台效率摘要（cosmx→merscope 的 log 均值）
    eff_s = float(np.log(np.clip(p_g_shared.mean(), 1e-4, 1)))
    eff_p = float(np.log(np.clip(p_g_spec.mean(), 1e-4, 1)))
    os.makedirs(cfg.out_dir, exist_ok=True)
    # 断点续训
    ckpt = os.path.join(cfg.out_dir, 'model_fold%d.pt' % fold)
    start_ep = 0
    if os.path.exists(ckpt) and not smoke:
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        start_ep = state['epoch'] + 1
        print('  续训自 epoch %d' % start_ep, flush=True)
    torch.cuda.reset_peak_memory_stats() if device.type == 'cuda' else None
    t0 = time.time()
    for ep in range(start_ep, epochs):
        model.train()
        order = np.random.RandomState(SEED + ep).permutation(idx_fit)
        tl = {}
        n_b = 0
        for b in range(0, len(order), n_batch):
            bidx = order[b:b + n_batch]
            extra = None
            if cfg.gate_mode == 'cell_evidence':
                extra = torch.stack([Qs_t[bidx], Qp_t[bidx], Kt[bidx]], dim=-1)
            elif cfg.gate_mode == 'platform':
                extra = torch.stack([torch.full((len(bidx),), eff_s, device=device),
                                     torch.full((len(bidx),), eff_p, device=device)], dim=-1)
            elif cfg.gate_mode == 'full':
                extra = torch.stack([Qs_t[bidx], Qp_t[bidx], Kt[bidx],
                                     torch.full((len(bidx),), eff_s, device=device),
                                     torch.full((len(bidx),), eff_p, device=device)], dim=-1)
            xs_b = Xs_t[bidx]
            if getattr(cfg, 'zero_shared', False):
                xs_b = xs_b * 0.0
            out = model(xs_b, Xp_t[bidx], extra=extra)
            l_cls = nn.functional.cross_entropy(out['logits_fine'], yf_t[bidx])
            l_cls_c = nn.functional.cross_entropy(out['logits_coarse'], yc_t[bidx])
            total = l_cls + l_cls_c
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl['l_cls'] = tl.get('l_cls', 0) + l_cls.item()
            tl['l_cls_c'] = tl.get('l_cls_c', 0) + l_cls_c.item()
            n_b += 1
        if ep % 2 == 0 or ep == epochs - 1:
            gpu_mem = (torch.cuda.max_memory_allocated() / 1024**2) if device.type == 'cuda' else 0
            print('  ep%d l_cls=%.4f l_cls_c=%.4f gpu_peak=%.0fMB' % (
                ep, tl['l_cls'] / n_b, tl['l_cls_c'] / n_b, gpu_mem), flush=True)
    dt = time.time() - t0
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=epochs - 1), ckpt)
    model.eval()

    def predict(idx):
        outs = []
        with torch.no_grad():
            for b in range(0, len(idx), 4096):
                bidx = idx[b:b + 4096]
                extra = None
                if cfg.gate_mode == 'cell_evidence':
                    extra = torch.stack([Qs_t[bidx], Qp_t[bidx], Kt[bidx]], dim=-1)
                elif cfg.gate_mode == 'platform':
                    extra = torch.stack([torch.full((len(bidx),), eff_s, device=device),
                                         torch.full((len(bidx),), eff_p, device=device)], dim=-1)
                elif cfg.gate_mode == 'full':
                    extra = torch.stack([Qs_t[bidx], Qp_t[bidx], Kt[bidx],
                                         torch.full((len(bidx),), eff_s, device=device),
                                         torch.full((len(bidx),), eff_p, device=device)], dim=-1)
                xs_b = Xs_t[bidx]
                if getattr(cfg, 'zero_shared', False):
                    xs_b = xs_b * 0.0
                o = model(xs_b, Xp_t[bidx], extra=extra)
                g = o['gate']
                if g is not None:
                    g = g.squeeze(-1).cpu().numpy()
                else:
                    g = np.zeros(len(bidx))
                outs.append(dict(gate=g,
                                 proba_f=torch.softmax(o['logits_fine'], -1).cpu().numpy(),
                                 proba_c=torch.softmax(o['logits_coarse'], -1).cpu().numpy(),
                                 z=o['z'].cpu().numpy()))
        return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
    for tag, idx in [('val', idx_val), ('test', idx_te)]:
        d = predict(idx)
        np.savez_compressed(os.path.join(cfg.out_dir, 'pred_fold%d_%s.npz' % (fold, tag)),
                            idx=idx, **{k: d[k] for k in d})
    print('fold%d done in %.0fs, n_fit=%d' % (fold, dt, len(idx_fit)), flush=True)
    return len(idx_fit)


def main(cfg, smoke=False, fold=None):
    data = build_data5(smoke=smoke)
    folds = [fold] if fold else [1, 2, 3]
    for f in folds:
        train_fold5(cfg, data, f, smoke=smoke)
    print('TRAIN5_DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/experiment5.yaml')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--fold', type=int, default=None)
    ap.add_argument('--variant', default=None)
    args = ap.parse_args()
    from types import SimpleNamespace
    raw = yaml.safe_load(open(args.config))
    merged = {}
    for sec in ['data', 'model', 'train']:
        if sec in raw:
            merged.update(raw[sec])
    for k, v in raw.items():
        if k not in ('data', 'model', 'train', 'variants'):
            merged[k] = v
    if args.variant and 'variants' in raw and args.variant in raw['variants']:
        merged.update(raw['variants'][args.variant])
    merged['variant_name'] = args.variant or 'base'
    merged['out_dir'] = os.path.join('outputs/exp5', merged['variant_name'])
    cfg = SimpleNamespace(**merged)
    main(cfg, smoke=args.smoke, fold=args.fold)
