# -*- coding: utf-8 -*-
"""Phase 3.1 第三阶段：M1 冻结检查点门控反事实诊断实验（Frozen-checkpoint counterfactual diagnostic）。

干预设置（全部冻结推理，不更新任何参数，共用同一测试细胞/标签/患者折/检查点/预处理/分类头）：
  A Original M1           原学习到的逐细胞门控（基准）
  B Shared-only           g=1（分类头仍为原 M1 头——冻结反事实，非重训 M8）
  C Panel-specific-only   g=0（非重训 M9）
  D Fixed 0.5             g=0.5
  E Training-fold mean gate  该折训练细胞平均门控（标量，来自训练数据，不用测试标签）
  F Permuted cell-level gate 同一测试折内打乱 g（保持分布、破坏细胞匹配），100 次/种子×折
  G 敏感性                g=0.25 / g=0.75（仅敏感性分析，不据测试结果选择门控再宣称独立验证）

关键数学：z = g·z_shared + (1−g)·z_spec；fine 为线性头 → 预计算 zs/zp 后，
所有干预为 CPU 线性代数：logits(g) = (zp@W.T+b) + g·(zs−zp)@W.T。
所有指标经 phase31_common 统一口径。结果解释限制（分布偏移等）见 report。
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase31_common import (accuracy, macro_f1, balanced_accuracy, ece15, brier_mc,
                            sd_sample, sd_pop, perm_pvalue_plus1, fmt)
from phase31_infra import (load_build_data, load_m1_checkpoint, load_pred_npz,
                           FOLDS, SEEDS, N_FINE, patient_of_idx, ROOT)

P31 = ROOT + '/paper_preparation/09_Phase31_Results'
OUT = P31 + '/04_Frozen_Gate_Counterfactual'
BRANCH = OUT + '/branch_outputs'
N_PERMS = 100
PERM_RNG_BASE = 20260331

INTERVENTIONS = [
    ('A_original', 'learned per-cell gate'),
    ('B_shared_only', 'g=1'),
    ('C_spec_only', 'g=0'),
    ('D_fixed_0.5', 'g=0.5'),
    ('E_train_mean', 'training-fold mean gate'),
    ('G_sens_0.25', 'g=0.25 (sensitivity)'),
    ('G_sens_0.75', 'g=0.75 (sensitivity)'),
]


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def forward_batches(model, Xs_gpu, Xp_gpu, rows, batch=4096):
    """返回 zs, zp, g, logits_fine（CPU float32）。"""
    zs, zp, g, logits = [], [], [], []
    with torch.no_grad():
        for b in range(0, len(rows), batch):
            r = rows[b:b + batch]
            out = model(Xs_gpu[r], Xp_gpu[r])
            zs.append(out['z_shared'].cpu().numpy())
            zp.append(out['z_spec'].cpu().numpy())
            g.append(out['gate'].cpu().numpy().reshape(-1))
            logits.append(out['logits_fine'].cpu().numpy())
    return (np.concatenate(zs), np.concatenate(zp),
            np.concatenate(g), np.concatenate(logits))


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(BRANCH, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}', flush=True)

    data = load_build_data()
    y_all, folds_all, core_all = data['y_fine'], data['folds'], data['core_arr']
    Xs_gpu = torch.from_numpy(data['X_shared']).float().to(device)
    Xp_gpu = torch.from_numpy(data['X_specific']).float().to(device)

    repro_rows, fold_rows, seed_rows, perm_rows, gate_rows = [], [], [], [], []
    gcls_rows, gpat_rows, gdist_rows, f1_rows = [], [], [], []

    for s in SEEDS:
        print(f'--- seed {s} ---', flush=True)
        seed_pooled = {iv: [] for iv, _ in INTERVENTIONS}
        for f in FOLDS:
            print(f'  fold {f}: loading checkpoint', flush=True)
            model, cfg, meta = load_m1_checkpoint(s, f, device=device,
                                                  input_transform='log1p')
            te = np.where(folds_all == f)[0]
            tr = np.where(folds_all != f)[0]  # folds 均 ∈ {1,2,3}
            print(f'  fold {f}: forward test n={len(te)}', flush=True)
            zs, zp, g_test, logits = forward_batches(model, Xs_gpu, Xp_gpu, te)
            print(f'  fold {f}: forward train n={len(tr)} (gate only)', flush=True)
            _, _, g_train, _ = forward_batches(model, Xs_gpu, Xp_gpu, tr)
            train_mean_gate = float(g_train.mean())

            # ---- 冻结预测复现检查 ----
            npz = load_pred_npz(s, f, 'm1')
            proba = softmax(logits)
            dprob = float(np.abs(proba - np.asarray(npz['proba_f'])).max())
            dgate = float(np.abs(g_test - np.asarray(npz['gate'])).max())
            repro_rows.append(dict(seed=s, fold=f, max_dproba=dprob,
                                   max_dgate=dgate, train_mean_gate=train_mean_gate,
                                   n_test=len(te), n_train=len(tr)))
            print(f'  repro: max|dproba|={dprob:.3e} max|dgate|={dgate:.3e} '
                  f'train_mean_g={train_mean_gate:.6f}', flush=True)

            # ---- 分支输出落盘（冻结证据）----
            pats = patient_of_idx(te, core_all)
            np.savez_compressed(
                f'{BRANCH}/m1_s{s}_fold{f}_branch.npz',
                zs=zs, zp=zp, g=g_test, idx=te, y=y_all[te],
                patient=pats, fold=np.full(len(te), f),
                train_mean_gate=np.float32(train_mean_gate))
            # 计算 SHA 由后续清单脚本统一完成

            # ---- 干预线性代数 ----
            W = model.fine.weight.detach().cpu().numpy().astype(np.float32)
            b = model.fine.bias.detach().cpu().numpy().astype(np.float32)
            A0 = zp @ W.T + b                    # [N,8]
            D = (zs - zp) @ W.T                  # [N,8]
            y_te = y_all[te]
            gvec = {'A_original': g_test,
                    'B_shared_only': np.ones(len(te), np.float32),
                    'C_spec_only': np.zeros(len(te), np.float32),
                    'D_fixed_0.5': np.full(len(te), 0.5, np.float32),
                    'E_train_mean': np.full(len(te), train_mean_gate, np.float32),
                    'G_sens_0.25': np.full(len(te), 0.25, np.float32),
                    'G_sens_0.75': np.full(len(te), 0.75, np.float32)}
            for key, _ in INTERVENTIONS:
                gv = gvec[key]
                lg = A0 + gv[:, None] * D
                pr = softmax(lg)
                yh = lg.argmax(axis=1)
                fold_rows.append(dict(seed=s, fold=f, intervention=key,
                                      n_cells=len(te), n_classes_covered=int(len(np.unique(y_te))),
                                      acc=accuracy(y_te, yh),
                                      macro_f1=macro_f1(y_te, yh, N_FINE),
                                      bal_acc=balanced_accuracy(y_te, yh, N_FINE),
                                      ece=ece15(pr, y_te)[0],
                                      brier=brier_mc(pr, y_te)))
            # ---- 每类 F1（seed42 与各种子）----
            from phase31_common import per_class_f1
            for key, _ in INTERVENTIONS:
                gv = gvec[key]
                yh = (A0 + gv[:, None] * D).argmax(axis=1)
                f1c = per_class_f1(y_te, yh, N_FINE)
                for c in range(N_FINE):
                    f1_rows.append(dict(seed=s, fold=f, intervention=key,
                                        class_idx=c, class_name=data['fine_names'][c],
                                        f1=f1c[c], n_cells=int((y_te == c).sum())))

            # ---- F: 打乱门控（保持分布、破坏匹配）----
            lg_orig = A0 + g_test[:, None] * D
            acc_orig = accuracy(y_te, lg_orig.argmax(axis=1))
            perm_accs = np.empty(N_PERMS)
            perm_macro = np.empty(N_PERMS)
            perm_bal = np.empty(N_PERMS)
            for p in range(N_PERMS):
                rng = np.random.RandomState(PERM_RNG_BASE + s * 1000 + f * 10 + p)
                gp = g_test[rng.permutation(len(g_test))]
                lg = A0 + gp[:, None] * D
                yh = lg.argmax(axis=1)
                perm_accs[p] = accuracy(y_te, yh)
                perm_macro[p] = macro_f1(y_te, yh, N_FINE)
                perm_bal[p] = balanced_accuracy(y_te, yh, N_FINE)
            p_plus1 = perm_pvalue_plus1(acc_orig, perm_accs, alternative='greater')
            perm_rows.append(dict(seed=s, fold=f, orig_acc=acc_orig,
                                  perm_acc_mean=perm_accs.mean(),
                                  perm_acc_sd=sd_sample(perm_accs),
                                  perm_acc_min=perm_accs.min(),
                                  perm_acc_max=perm_accs.max(),
                                  perm_macro_mean=perm_macro.mean(),
                                  perm_bal_mean=perm_bal.mean(),
                                  p_plus1_greater=p_plus1, n_perms=N_PERMS,
                                  n_perm_ge_orig=int((perm_accs >= acc_orig).sum())))
            print(f'  F perm: orig={acc_orig:.6f} perm_mean={perm_accs.mean():.6f} '
                  f'perm_sd={perm_accs.std():.6f} p_plus1={p_plus1:.4f}', flush=True)

            # ---- 门控统计 ----
            gate_rows.append(dict(seed=s, fold=f, mean_g=float(g_test.mean()),
                                  sd_g=sd_sample(g_test),
                                  pct_lt_0_5=float((g_test < 0.5).mean()) * 100,
                                  q25=float(np.percentile(g_test, 25)),
                                  q50=float(np.percentile(g_test, 50)),
                                  q75=float(np.percentile(g_test, 75)),
                                  train_mean_g=train_mean_gate))
            for c in range(N_FINE):
                m = y_te == c
                if m.sum() == 0:
                    continue
                gcls_rows.append(dict(seed=s, fold=f, class_idx=c,
                                      class_name=data['fine_names'][c],
                                      n_cells=int(m.sum()), mean_g=float(g_test[m].mean()),
                                      sd_g=sd_sample(g_test[m])))
            upat, ipat = np.unique(pats, return_inverse=True)
            for pi in range(len(upat)):
                m = ipat == pi
                gpat_rows.append(dict(seed=s, fold=f, patient=upat[pi],
                                      n_cells=int(m.sum()), mean_g=float(g_test[m].mean())))
            hist, edges = np.histogram(g_test, bins=20, range=(0, 1))
            for bi in range(20):
                gdist_rows.append(dict(seed=s, fold=f, bin_left=edges[bi],
                                       bin_right=edges[bi + 1], count=int(hist[bi])))
            del model
            torch.cuda.empty_cache()

        # ---- 种子级汇总（合并该种子三折测试细胞）----
        from collections import defaultdict
        pooled = defaultdict(lambda: defaultdict(list))
        for r in fold_rows:
            if r['seed'] != s:
                continue
            pooled[r['intervention']]['acc'].append(r['acc'])
            pooled[r['intervention']]['macro'].append(r['macro_f1'])
            pooled[r['intervention']]['bal'].append(r['bal_acc'])
            pooled[r['intervention']]['ece'].append(r['ece'])
            pooled[r['intervention']]['brier'].append(r['brier'])
        for key, _ in INTERVENTIONS:
            d = pooled[key]
            seed_rows.append(dict(seed=s, intervention=key,
                                  eq_acc_mean=np.mean(d['acc']),
                                  eq_acc_sd_sample=sd_sample(d['acc']),
                                  eq_acc_sd_pop=sd_pop(d['acc']),
                                  eq_macro_mean=np.mean(d['macro']),
                                  eq_macro_sd_sample=sd_sample(d['macro']),
                                  eq_bal_mean=np.mean(d['bal']),
                                  eq_ece_mean=np.mean(d['ece']),
                                  eq_brier_mean=np.mean(d['brier'])))
        print(f'  seed {s} done', flush=True)

    # ---- 落盘 ----
    import pandas as pd
    pd.DataFrame(fold_rows).to_csv(f'{OUT}/counterfactual_fold_level.csv', index=False)
    pd.DataFrame(seed_rows).to_csv(f'{OUT}/counterfactual_seed_level.csv', index=False)
    pd.DataFrame(perm_rows).to_csv(f'{OUT}/gate_permutation_results.csv', index=False)
    pd.DataFrame(gate_rows).to_csv(f'{OUT}/gate_stats.csv', index=False)
    pd.DataFrame(gcls_rows).to_csv(f'{OUT}/gate_per_class.csv', index=False)
    pd.DataFrame(gpat_rows).to_csv(f'{OUT}/gate_per_patient.csv', index=False)
    pd.DataFrame(gdist_rows).to_csv(f'{OUT}/gate_distribution.csv', index=False)
    pd.DataFrame(f1_rows).to_csv(f'{OUT}/per_class_f1_interventions.csv', index=False)
    pd.DataFrame(repro_rows).to_csv(f'{OUT}/frozen_reproduction_check.csv', index=False)
    print('ALL_CSV_SAVED', flush=True)


if __name__ == '__main__':
    main()
