# -*- coding: utf-8 -*-
"""Phase 3.1 数据与模型加载基础设施。

统一提供：
- build_data5 重建数据（X_shared/X_specific/y_fine/y_coarse/folds/core_arr/genes）
- M1 检查点加载（outputs/exp6/m1_s{seed}/model_fold{fold}.pt）
- 冻结预测 NPZ 加载（outputs/exp6/{m1,m2,m3}_s{seed}/pred_fold{f}_test.npz）
- LR 基线预测 NPZ 加载（outputs/exp3_baselines/closed_LR_{A,B,C}_fold{f}.npz）
- 患者 ID 映射（core -> patient，来自 sceneA_final_manifest.csv / core_split_manifest.csv）

注意：build_data5 重建数组的行序与冻结预测 NPZ 的 idx 列对应（全局行索引），
本模块的 verify_idx_alignment() 会做一致性检查。
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

ROOT = '/home/hst/research/openset_st_cf'
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + '/scripts/experiment5')
sys.path.insert(0, ROOT + '/scripts/experiment6')

SEEDS = [42, 7, 99, 123, 2024]
FOLDS = [1, 2, 3]
N_FINE = 8
N_COARSE = 6

_cache = {}


def load_build_data(use_cache=True):
    """重建数据（与训练协议完全一致的 build_data5）。"""
    if use_cache and 'build_data' in _cache:
        return _cache['build_data']
    from train5 import build_data5
    data = build_data5(smoke=False)
    if use_cache:
        _cache['build_data'] = data
    return data


def make_cfg():
    """与 train6.py 训练协议一致的配置对象。"""
    from train6 import make_cfg as _make_cfg
    return _make_cfg('base', False, False)


def load_m1_checkpoint(seed, fold, device='cuda', input_transform='log1p'):
    """加载 M1 检查点（eval 模式）。返回 (model, cfg, ckpt_meta)。"""
    from src.models.experiment6.model6 import M1Model6
    cfg = make_cfg()
    model = M1Model6(cfg, input_transform=input_transform)
    ck_path = f'{ROOT}/outputs/exp6/m1_s{seed}/model_fold{fold}.pt'
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    model.to(device).eval()
    meta = {k: v for k, v in ck.items() if k != 'model'}
    meta['ckpt_path'] = ck_path
    return model, cfg, meta


def load_pred_npz(seed, fold, model='m1', part='test'):
    """冻结预测 NPZ。返回 np.lib.npyio.NpzFile（含 idx/gate/proba_f/proba_c/z）。"""
    p = f'{ROOT}/outputs/exp6/{model}_s{seed}/pred_fold{fold}_{part}.npz'
    return np.load(p)


def load_lr_npz(fold, variant):
    """LR 基线冻结预测（含 cell_idx/y_true/y_pred/proba）。"""
    p = f'{ROOT}/outputs/exp3_baselines/closed_LR_{variant}_fold{fold}.npz'
    return np.load(p)


def core_to_patient():
    """core -> patient 映射（core_split_manifest 的 patient_id 列，限定 tTMA1——
    三个 TMA 的 core 编号重叠，必须限定，否则映射被覆盖出错）。"""
    man = pd.read_csv(f'{ROOT}/data/metadata/core_split_manifest.csv')
    man = man[man.tma == 'tTMA1']
    return dict(zip(man.core.astype(str), man.patient_id.astype(str)))


def patient_of_idx(idx, core_arr):
    """给定全局行索引，返回患者 ID 数组。"""
    c2p = core_to_patient()
    return np.array([c2p[c] for c in core_arr[idx]], dtype=object)


def verify_idx_alignment():
    """验证冻结预测 idx 与 build_data5 全局行序对齐、与 LR cell_idx 逐位一致。

    返回 dict：{per_fold: {n_cells, idx_is_global, matches_lr}, ...}
    """
    data = load_build_data()
    n_total = len(data['y_fine'])
    out = {}
    for f in FOLDS:
        d = load_pred_npz(42, f, 'm1')
        idx = d['idx']
        lr = load_lr_npz(f, 'A')
        out[f] = dict(
            n_cells=int(len(idx)),
            idx_is_global=bool(idx.min() >= 0 and idx.max() < n_total),
            idx_max=int(idx.max()),
            n_total=int(n_total),
            matches_lr=bool(np.array_equal(idx, lr['cell_idx'])),
        )
    return out


def gate_of_pred_npz(seed, fold, model='m1'):
    d = load_pred_npz(seed, fold, model)
    return np.asarray(d['gate'], dtype=np.float32)
