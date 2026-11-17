"""Neighbour smoother with a learned per-column metric.

    w(x, x_i) = exp(-sum_j lam_j [x_j != x_ij])
    p(y | x)  = (sum_i w [y_i = y] + a * p_base(y | x)) / (sum_i w + a)

The column weights lam_j are shared by all items of an instrument and fitted
by coordinate search on out-of-fold log score (training rows in the query's
own fold get weight 0), using a random subsample of query rows and the items
where the base model is least certain. A column whose differences carry no
information gets lam_j -> 0; a column that separates unlike respondents gets
a large lam_j. Then, per item, a global multiplier on lam and the shrinkage a
toward the base model are chosen by OOF log score over all training rows.
"""
import time

import numpy as np

from .core import FLOOR

GRID = (0.0, 0.3, 1.0, 2.0, 4.0, 8.0)
SCALES = (0.5, 1.0, 2.0)
SHRINK = (0.02, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
FIT_SHRINK = (0.1, 1.0, 10.0)


def _onehot_targets(inst, items=None):
    K = inst.K
    starts = np.concatenate([[0], np.cumsum(K)[:-1]]).astype(np.int64)
    items = list(range(len(K))) if items is None else list(items)
    cols, off = [], 0
    sub_starts = []
    for t in items:
        sub_starts.append(off)
        off += int(K[t])
    Y = np.zeros((inst.Y.shape[0], off), dtype=np.float32)
    for s, t in zip(sub_starts, items):
        Y[np.arange(Y.shape[0]), s + inst.Y[:, t]] = 1.0
    return Y, np.asarray(sub_starts, dtype=np.int64), items


def fit_metric(inst, base_oof, deadline, q=2000, max_options=256, seed=0,
               sweeps=2, log=None):
    XF = inst.XF
    n, d = XF.shape
    rng = np.random.default_rng(seed)
    Q = np.sort(rng.choice(n, size=min(q, n), replace=False))
    # Items where the base model is least certain carry the metric signal.
    loss = np.array([-np.mean(np.log(np.maximum(
        base_oof[t][np.arange(n), inst.Y[:, t]], 1e-12)))
        for t in range(len(inst.K))])
    order = np.argsort(-loss)
    items, tot = [], 0
    for t in order:
        if loss[t] <= 1e-4:
            break
        if tot + inst.K[t] > max_options and items:
            continue
        items.append(int(t))
        tot += int(inst.K[t])
    if not items:
        return np.ones(d, dtype=np.float32)
    Yoh, starts, items = _onehot_targets(inst, items)
    M = [(XF[Q, j][:, None] != XF[None, :, j]).view(np.uint8)
         for j in range(d)]
    same = inst.folds[Q][:, None] == inst.folds[None, :]
    yq = [inst.Y[Q, t] for t in items]
    pbq = [base_oof[t][Q, inst.Y[Q, t]] for t in items]
    rows = np.arange(Q.size)

    def evaluate(Dw):
        W = np.exp(-Dw)
        W[same] = 0.0
        num = W @ Yoh
        den = W.sum(1)
        tot = 0.0
        for s, t, y, pb in zip(starts, items, yq, pbq):
            k = int(inst.K[t])
            ny = num[rows, s + y]
            best = -np.inf
            for a in FIT_SHRINK:
                p = (ny + a * pb) / (den + a)
                best = max(best, float(np.mean(np.log(FLOOR + (1 - k * FLOOR) * p))))
            tot += best
        return tot / len(items)

    lam = np.ones(d, dtype=np.float32)
    Dw = np.zeros((Q.size, n), dtype=np.float32)
    for Mj in M:
        Dw += Mj
    cur = evaluate(Dw)
    for _ in range(sweeps):
        improved = False
        for j in range(d):
            if time.time() > deadline:
                return lam
            for v in GRID:
                if v == lam[j]:
                    continue
                cand = Dw + np.float32(v - lam[j]) * M[j]
                sc = evaluate(cand)
                if sc > cur + 1e-6:
                    cur, Dw, lam[j], improved = sc, cand, v, True
        if log is not None:
            log.append((lam.copy(), cur))
        if not improved:
            break
    return lam


def _wdist(A, B, lam, out=None):
    D = np.zeros((A.shape[0], B.shape[0]), dtype=np.float32) if out is None else out
    D[:] = 0.0
    for j in range(A.shape[1]):
        if lam[j] != 0:
            D += lam[j] * (A[:, j][:, None] != B[:, j][None, :])
    return D


def knnw_learner(inst, base_oof, base_test, deadline=None, chunk=2048,
                 q=2000, seed=0, lam=None):
    deadline = deadline or (time.time() + 1e9)
    if lam is None:
        lam = fit_metric(inst, base_oof, deadline, q=q, seed=seed)
    XF, XH = inst.XF, inst.XH
    nF, nH = XF.shape[0], XH.shape[0]
    K = inst.K
    T = len(K)
    Yoh, starts, _ = _onehot_targets(inst)
    folds = inst.folds
    rows = np.arange(nF)
    nums = {s: np.zeros((nF, Yoh.shape[1]), dtype=np.float32) for s in SCALES}
    dens = {s: np.zeros(nF, dtype=np.float32) for s in SCALES}
    for c0 in range(0, nF, chunk):
        if time.time() > deadline:
            return base_oof, base_test, lam
        D = _wdist(XF[c0:c0 + chunk], XF, lam)
        same = folds[c0:c0 + chunk][:, None] == folds[None, :]
        for s in SCALES:
            W = np.exp(-s * D)
            W[same] = 0.0
            nums[s][c0:c0 + chunk] = W @ Yoh
            dens[s][c0:c0 + chunk] = W.sum(1)
    best = [(-np.inf, None, None)] * T
    oof = [None] * T
    for t in range(T):
        k = int(K[t])
        sl = slice(starts[t], starts[t] + k)
        y = inst.Y[:, t]
        pb = base_oof[t]
        for s in SCALES:
            nt = nums[s][:, sl].astype(np.float64)
            for a in SHRINK:
                p = (nt + a * pb) / (dens[s][:, None] + a)
                sc = np.mean(np.log(FLOOR + (1 - k * FLOOR) * p[rows, y]))
                if sc > best[t][0]:
                    best[t] = (sc, s, a)
                    oof[t] = p
    test = [None] * T
    numH = {s: np.zeros((nH, Yoh.shape[1]), dtype=np.float32) for s in SCALES}
    denH = {s: np.zeros(nH, dtype=np.float32) for s in SCALES}
    for c0 in range(0, nH, chunk):
        D = _wdist(XH[c0:c0 + chunk], XF, lam)
        for s in SCALES:
            W = np.exp(-s * D)
            numH[s][c0:c0 + chunk] = W @ Yoh
            denH[s][c0:c0 + chunk] = W.sum(1)
    for t in range(T):
        _, s, a = best[t]
        k = int(K[t])
        sl = slice(starts[t], starts[t] + k)
        test[t] = (numH[s][:, sl].astype(np.float64) + a * base_test[t]) / (
            denH[s][:, None] + a)
    return oof, test, lam
