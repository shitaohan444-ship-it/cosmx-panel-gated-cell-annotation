#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验6：统一 M1 复现与种子稳定性训练。

与实验5训练协议逐行对齐（同数据、同患者级3折、同 fit/val 核心划分、同优化器），
新增可控维度：
  --input raw|log1p   输入变换（4.3=raw，exp5=log1p）
  --epochs N          训练轮数（4.3=12，exp5=15）
  --seed S            随机种子（只影响模型初始化/批序/dropout；val 划分固定用 42）
  --fusion base|concat|fixed  门控模式（M1/M2/M3）
  --zero-shared / --zero-spec  分支消融（M8/M9 等价物）
输出：outputs/exp6/<variant_name>/model_fold{f}.pt + pred_fold{f}_{val,test}.npz
用法: python scripts/experiment6/train6.py --input log1p --epochs 15 --seed 42 \
        --fusion base --variant-name m1_ref --fold 1
"""
import os
import sys
import time
import argparse
import numpy as np
import yaml
import torch

sys.path.insert(0, '/home/hst/research/openset_st_cf')
sys.path.insert(0, '/home/hst/research/openset_st_cf/scripts/experiment5')
from train5 import build_data5, load_p_g
from src.models.experiment6.model6 import M1Model6

SEED_VAL = 42  # fit/val 核心划分固定用 42，与实验5一致


def make_cfg(fusion, zero_shared, zero_spec):
    from types import SimpleNamespace
    raw = yaml.safe_load(open('/home/hst/research/openset_st_cf/configs/experiment5.yaml'))
    merged = {}
    for sec in ['data', 'model', 'train']:
        merged.update(raw[sec])
    for k, v in raw.items():
        if k not in ('data', 'model', 'train', 'variants'):
            merged[k] = v
    merged['gate_mode'] = fusion
    merged['use_specific'] = True
    merged['use_gene_weight'] = False
    merged['zero_shared'] = zero_shared
    merged['zero_spec'] = zero_spec
    merged['variant_name'] = 'm1ref'
    merged['out_dir'] = 'outputs/exp6/tmp'
    return SimpleNamespace(**merged)


def train_fold6(cfg, data, fold, args, out_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr = (data['folds'] != fold) & (data['folds'] > 0)
    te = data['folds'] == fold
    idx_tr = np.where(tr)[0]
    core_tr = data['core_arr'][tr]
    uniq = np.unique(core_tr)
    rng = np.random.RandomState(SEED_VAL)  # 固定 val 划分（与实验5一致）
    rng.shuffle(uniq)
    n_val = max(1, int(0.2 * len(uniq)))
    val_cores = set(uniq[:n_val])
    val_mask = np.array([c in val_cores for c in core_tr])
    idx_fit = idx_tr[~val_mask]
    idx_val = idx_tr[val_mask]
    idx_te = np.where(te)[0]
    print('fold%d: fit=%d val=%d test=%d (seed=%d input=%s epochs=%d fusion=%s)' % (
        fold, len(idx_fit), len(idx_val), len(idx_te),
        args.seed, args.input, args.epochs, args.fusion), flush=True)

    model = M1Model6(cfg, input_transform=args.input).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    n_batch = args.batch_size
    epochs = args.epochs
    Xs_t = torch.from_numpy(data['X_shared']).float().to(device)
    Xp_t = torch.from_numpy(data['X_specific']).float().to(device)
    yf_t = torch.from_numpy(data['y_fine']).long().to(device)
    yc_t = torch.from_numpy(data['y_coarse']).long().to(device)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, 'model_fold%d.pt' % fold)
    start_ep = 0
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        start_ep = state['epoch'] + 1
        print('  续训自 epoch %d' % start_ep, flush=True)
    torch.cuda.reset_peak_memory_stats() if device.type == 'cuda' else None
    t0 = time.time()
    for ep in range(start_ep, epochs):
        model.train()
        order = np.random.RandomState(args.seed + ep).permutation(idx_fit)
        tl = {}
        n_b = 0
        for b in range(0, len(order), n_batch):
            bidx = order[b:b + n_batch]
            out = model(Xs_t[bidx], Xp_t[bidx])
            l_cls = torch.nn.functional.cross_entropy(out['logits_fine'], yf_t[bidx])
            l_cls_c = torch.nn.functional.cross_entropy(out['logits_coarse'], yc_t[bidx])
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
                o = model(Xs_t[bidx], Xp_t[bidx])
                g = o['gate']
                g = g.squeeze(-1).cpu().numpy() if g is not None else np.zeros(len(bidx))
                outs.append(dict(gate=g,
                                 proba_f=torch.softmax(o['logits_fine'], -1).cpu().numpy(),
                                 proba_c=torch.softmax(o['logits_coarse'], -1).cpu().numpy(),
                                 z=o['z'].cpu().numpy()))
        return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
    for tag, idx in [('val', idx_val), ('test', idx_te)]:
        d = predict(idx)
        np.savez_compressed(os.path.join(out_dir, 'pred_fold%d_%s.npz' % (fold, tag)),
                            idx=idx, **{k: d[k] for k in d})
    print('fold%d done in %.0fs' % (fold, dt), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='log1p', choices=['raw', 'log1p'])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fold', type=int, default=None)
    ap.add_argument('--fusion', default='base', choices=['base', 'concat', 'fixed'])
    ap.add_argument('--zero-shared', action='store_true')
    ap.add_argument('--zero-spec', action='store_true')
    ap.add_argument('--variant-name', default='m1ref')
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--out-root', default='outputs/exp6')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = build_data5(smoke=False)
    cfg = make_cfg(args.fusion, args.zero_shared, args.zero_spec)
    out_dir = os.path.join(args.out_root, args.variant_name)
    folds = [args.fold] if args.fold else [1, 2, 3]
    for f in folds:
        train_fold6(cfg, data, f, args, out_dir)
    print('TRAIN6_DONE', flush=True)


if __name__ == '__main__':
    main()
