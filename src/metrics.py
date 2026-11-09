# -*- coding: utf-8 -*-
"""Phase 3.1 共享统计口径模块。

所有 Phase 3.1 重算必须经由此模块的指标函数，保证口径统一、可追溯、可复核。
口径定义与 docs/manuscript/06_reports/final_evidence_registry.md（冻结 2026-11-08）
及 paper_preparation/08_Peer_Review/03_Numerical_Consistency_Audit.xlsx 一致。

口径清单（本模块逐一实现并注明）：
- fold-level metric    ：在每一测试折内单独计算的指标（3 个值）
- equal-fold mean      ：三折指标等权平均（np.mean）
- sample SD            ：np.std(ddof=1)（样本标准差）
- population SD        ：np.std(ddof=0)（总体标准差）
- pooled OOF           ：拼接全部测试折细胞后计算（混淆矩阵来自合并计数）
- pooled macro-F1      ：合并混淆矩阵 → 每类 P/R → 每类 F1 → 8 类无权重平均
- fold-mean macro-F1   ：每折 macro-F1 的三折等权平均（与 pooled 不同，两者分列）
- ECE                  ：15 个等宽置信箱（[0,1] 均分），无温度缩放
- Brier (multi-class)  ：每细胞 8 类平方误差求和后平均（与登记表口径一致）
- 患者级配对           ：每患者内模型 A/B 的准确率差，患者为独立单位；
                        Wilcoxon 符号秩（zero_method='wilcox'，报告非零差对数）、
                        Cohen's dz = mean(d)/SD(d, ddof=1)、
                        患者聚类 bootstrap 百分位 CI（重抽样患者而非细胞）
- 置换 p 值             ：(b+1)/(B+1) 加一校正，避免有限置换下 p=0
- 多重比较             ：BH-FDR（各检验家族单独标注）、Holm
"""
import numpy as np
from scipy import stats


# --------------------------------------------------------------------------
# 基础指标
# --------------------------------------------------------------------------
def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_counts(y_true, y_pred, n_classes):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_f1(y_true, y_pred, n_classes):
    cm = confusion_counts(y_true, y_pred, n_classes)
    f1 = np.zeros(n_classes)
    for c in range(n_classes):
        tp = cm[c, c]
        prec = tp / cm[:, c].sum() if cm[:, c].sum() > 0 else 0.0
        rec = tp / cm[c, :].sum() if cm[c, :].sum() > 0 else 0.0
        f1[c] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1


def macro_f1(y_true, y_pred, n_classes):
    """macro-F1：每类 F1 的无权重平均（在给定样本集上）。"""
    return float(np.mean(per_class_f1(y_true, y_pred, n_classes)))


def balanced_accuracy(y_true, y_pred, n_classes):
    cm = confusion_counts(y_true, y_pred, n_classes)
    rec = np.zeros(n_classes)
    for c in range(n_classes):
        rec[c] = cm[c, c] / cm[c, :].sum() if cm[c, :].sum() > 0 else 0.0
    return float(np.mean(rec))


def ece15(proba, y_true):
    """15 等宽置信箱 ECE。proba: [N, K] 每行和为 1；无温度缩放。"""
    proba = np.asarray(proba, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    acc = (pred == y_true).astype(np.float64)
    n_bins = 15
    ece = 0.0
    counts = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        n = int(m.sum())
        counts.append(n)
        if n == 0:
            continue
        ece += (n / len(y_true)) * abs(acc[m].mean() - conf[m].mean())
    return float(ece), counts


def brier_mc(proba, y_true):
    """多类 Brier：每细胞 8 类平方误差求和后平均。"""
    proba = np.asarray(proba, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


# --------------------------------------------------------------------------
# 折级 / 池化 汇总口径
# --------------------------------------------------------------------------
def sd_sample(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.std(x, ddof=1)) if len(x) > 1 else float('nan')


def sd_pop(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.std(x, ddof=0))


def fold_level_metrics(y_true, y_pred, proba, folds, n_classes):
    """返回完整口径字典。

    结果字段：
      per_fold          : [dict] 每折的 acc / macro_f1 / bal_acc / ece / brier
      eq_acc_mean       : 三折等权平均 acc
      eq_acc_sd_sample  : 三折 acc 的样本 SD（ddof=1）
      eq_acc_sd_pop     : 三折 acc 的总体 SD（ddof=0）
      eq_macro_mean     : 三折 macro-F1 等权平均
      eq_macro_sd_sample/ sd_pop
      pooled_acc        : 合并 OOF acc
      pooled_macro_f1   : 合并混淆计数后计算的 macro-F1（与折均不同口径，分列）
      pooled_bal_acc
      pooled_ece / pooled_ece_counts
      pooled_brier
      n_cells_folds     : 每折细胞数
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int64)
    per_fold = []
    accs, macros = [], []
    for f in np.unique(folds):
        m = folds == f
        accs.append(accuracy(y_true[m], y_pred[m]))
        macros.append(macro_f1(y_true[m], y_pred[m], n_classes))
        ece, _ = ece15(proba[m], y_true[m])
        per_fold.append(dict(
            fold=int(f), n_cells=int(m.sum()),
            acc=accs[-1], macro_f1=macros[-1],
            bal_acc=balanced_accuracy(y_true[m], y_pred[m], n_classes),
            ece=ece, brier=brier_mc(proba[m], y_true[m])))
    pooled_ece, pooled_ece_counts = ece15(proba, y_true)
    return dict(
        per_fold=per_fold,
        eq_acc_mean=float(np.mean(accs)),
        eq_acc_sd_sample=sd_sample(accs),
        eq_acc_sd_pop=sd_pop(accs),
        eq_macro_mean=float(np.mean(macros)),
        eq_macro_sd_sample=sd_sample(macros),
        eq_macro_sd_pop=sd_pop(macros),
        pooled_acc=accuracy(y_true, y_pred),
        pooled_macro_f1=macro_f1(y_true, y_pred, n_classes),
        pooled_bal_acc=balanced_accuracy(y_true, y_pred, n_classes),
        pooled_ece=pooled_ece,
        pooled_ece_counts=pooled_ece_counts,
        pooled_brier=brier_mc(proba, y_true),
        n_cells_folds=[int((folds == f).sum()) for f in np.unique(folds)],
    )


# --------------------------------------------------------------------------
# 患者级统计（患者 = 独立统计单位）
# --------------------------------------------------------------------------
def patient_accuracy(y_true, y_pred, patient_ids):
    """每患者测试细胞内的准确率。返回 (patients 列表, per-patient acc 数组)。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    patient_ids = np.asarray(patient_ids)
    uniq, inv = np.unique(patient_ids, return_inverse=True)
    acc = np.zeros(len(uniq))
    for i in range(len(uniq)):
        m = inv == i
        acc[i] = np.mean(y_true[m] == y_pred[m])
    return list(uniq), acc


def wilcoxon_paired(diff):
    """Wilcoxon 符号秩检验（双侧）。返回 (stat, p, n_nonzero_diff)。"""
    d = np.asarray(diff, dtype=np.float64)
    nz = d[d != 0]
    if len(nz) == 0:
        return float('nan'), 1.0, 0
    w = stats.wilcoxon(nz, zero_method='wilcox', alternative='two-sided')
    return float(w.statistic), float(w.pvalue), len(nz)


def cohens_dz(diff):
    d = np.asarray(diff, dtype=np.float64)
    s = np.std(d, ddof=1)
    if s == 0:
        return float('nan')
    return float(np.mean(d) / s)


def patient_bootstrap_ci(acc_a_pat, acc_b_pat, n_rep=2000, rng_seed=20260331):
    """患者聚类 bootstrap：重抽样患者（非细胞）。

    返回 (ci_lo, ci_hi, prop_neg, mean_diff, n_patients)。
    ci 为百分位 2.5/97.5；prop_neg 为重抽样中差值 <= 0 的比例（描述性）。
    """
    acc_a = np.asarray(acc_a_pat, dtype=np.float64)
    acc_b = np.asarray(acc_b_pat, dtype=np.float64)
    d = acc_a - acc_b
    n = len(d)
    rng = np.random.RandomState(rng_seed)
    boots = np.empty(n_rep)
    for r in range(n_rep):
        s = rng.choice(n, size=n, replace=True)
        boots[r] = d[s].mean()
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)),
            float(np.mean(boots <= 0)), float(d.mean()), int(n))


# --------------------------------------------------------------------------
# 多重比较
# --------------------------------------------------------------------------
def bh_fdr(pvals):
    """Benjamini-Hochberg FDR 校正。返回 q 值数组（与输入同序）。"""
    p = np.asarray(pvals, dtype=np.float64)
    q = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    order = np.argsort(pv)
    m = len(pv)
    qv = np.empty(m)
    for rank, idx in enumerate(order):
        qv[idx] = min(1.0, pv[idx] * m / (rank + 1))
    # 单调性修正（从大到小）
    for i in range(m - 2, -1, -1):
        j = order[i]
        jn = order[i + 1]
        if qv[j] > qv[jn]:
            qv[j] = qv[jn]
    q[valid] = qv
    return q


def holm(pvals):
    """Holm 校正。返回校正后 p 值数组。"""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    for rank, idx in enumerate(order):
        adj[idx] = min(1.0, p[idx] * (m - rank))
    return adj


# --------------------------------------------------------------------------
# 置换检验
# --------------------------------------------------------------------------
def perm_pvalue_plus1(observed, null_dist, alternative='greater'):
    """加一置换 p 值 (b+1)/(B+1)。null_dist 为置换统计量数组。

    alternative:
      'greater' : b = #(null >= observed)
      'less'    : b = #(null <= observed)
      'two-sided': b = #(|null - center| >= |observed - center|), center = null 中位数
    """
    null = np.asarray(null_dist, dtype=np.float64)
    B = len(null)
    if alternative == 'greater':
        b = int(np.sum(null >= observed))
    elif alternative == 'less':
        b = int(np.sum(null <= observed))
    elif alternative == 'two-sided':
        c = np.median(null)
        b = int(np.sum(np.abs(null - c) >= abs(observed - c)))
    else:
        raise ValueError(alternative)
    return (b + 1) / (B + 1)


# --------------------------------------------------------------------------
# 输出辅助
# --------------------------------------------------------------------------
def fmt(x, nd=6):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'NA'
    return f'{x:.{nd}f}'


def summary_lines(metrics, label):
    """把 fold_level_metrics 结果格式化为可写入报告的文本行。"""
    L = [f'== {label} ==',
         f"equal-fold mean acc : {fmt(metrics['eq_acc_mean'])} "
         f"(sample SD {fmt(metrics['eq_acc_sd_sample'])}, pop SD {fmt(metrics['eq_acc_sd_pop'])})",
         f"pooled OOF acc      : {fmt(metrics['pooled_acc'])}",
         f"fold-mean macro-F1  : {fmt(metrics['eq_macro_mean'])} "
         f"(sample SD {fmt(metrics['eq_macro_sd_sample'])})",
         f"pooled macro-F1     : {fmt(metrics['pooled_macro_f1'])}",
         f"pooled bal-acc      : {fmt(metrics['pooled_bal_acc'])}",
         f"pooled ECE (15-bin) : {fmt(metrics['pooled_ece'])}",
         f"pooled Brier (mc)   : {fmt(metrics['pooled_brier'])}",
         f"fold cell counts    : {metrics['n_cells_folds']}"]
    for pf in metrics['per_fold']:
        L.append(f"  fold {pf['fold']}: n={pf['n_cells']} acc={fmt(pf['acc'])} "
                 f"macroF1={fmt(pf['macro_f1'])} bal={fmt(pf['bal_acc'])} "
                 f"ece={fmt(pf['ece'])} brier={fmt(pf['brier'])}")
    return '\n'.join(L)
