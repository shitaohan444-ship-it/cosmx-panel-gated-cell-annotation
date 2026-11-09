# -*- coding: utf-8 -*-
"""Data loading for the Phase 3.6 release.

Expected preprocessed npz schema (identical to the outputs of build_data5 in
the run code, scripts/experiment5/train5.py):
  X_shared  : (N, 108) float32 raw counts (log1p applied inside the model)
  X_specific: (N, 892) float32 raw counts
  y_fine    : (N,) int64, 8-class labels
  y_coarse  : (N,) int64, 6-class labels
  folds     : (N,) int64, patient-level fold ids (1..3; 0 = excluded)
  core_arr  : (N,) str, core ids (used for the fixed fit/val split)
The original pipeline reads a CosMx h5ad plus metadata manifests; that
preprocessing is data-license-dependent and is NOT shipped here. Users must
obtain the public dataset (see docs/reproduction_guide.md) and export the
npz with the same schema.
"""
import os
from types import SimpleNamespace

import numpy as np
import yaml


def load_preprocessed(path):
    d = np.load(path, allow_pickle=True)
    return dict(X_shared=d['X_shared'], X_specific=d['X_specific'],
                y_fine=d['y_fine'], y_coarse=d['y_coarse'],
                folds=d['folds'], core_arr=d['core_arr'])


def make_cfg(model_config_yaml=None):
    """Model/training config identical to configs/experiment5.yaml."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw = yaml.safe_load(open(model_config_yaml or os.path.join(
        here, 'configs', 'model_config.example.yaml')))
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
    merged['zero_spec'] = False
    return SimpleNamespace(**merged)
