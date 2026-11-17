"""Hamming-kernel nearest-neighbour smoother over the full GIVEN profile.

    p(y | x) = (sum_i exp(-lam * d(x, x_i)) [y_i = y] + a * p_base(y | x))
               / (sum_i exp(-lam * d(x, x_i)) + a)

d = number of GIVEN columns that differ (unweighted Hamming distance), shared
by all items. lam and the shrinkage a toward a base estimate (the ladder) are
chosen per item by out-of-fold log score. Out-of-fold weights drop every
training row in the query's own fold, so OOF predictions stack with the
other learners on the same folds.
"""
import time

import numpy as np

from .core import FLOOR

LAMBDAS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0)
SHRINK = (0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0)


def hamming(A, B, chunk=2048):
    """(len(A), len(B)) uint8 count of differing columns."""
    out = np.zeros((A.shape[0], B.shape[0]), dtype=np.uint8)
    for s in range(0, A.shape[0], chunk):
        blk = A[s:s + chunk]
        acc = np.zeros((blk.shape[0], B.shape[0]), dtype=np.uint8)
        for j in range(A.shape[1]):
            acc += (blk[:, j][:, None] != B[:, j][None, :])
        out[s:s + chunk] = acc
    return out


def _segments(K):
    starts = np.concatenate([[0], np.cumsum(K)[:-1]]).astype(np.int64)
    return starts


def knn_learner(inst, base_oof, base_test, lambdas=LAMBDAS, shrink=SHRINK,
                deadline=None, chunk=2048):
    """base_oof / base_test: per-item probability lists (e.g. the ladder)."""
    deadline = deadline or (time.time() + 1e9)
    XF, XH = inst.XF, inst.XH
    nF, nH = XF.shape[0], XH.shape[0]
    K = inst.K
    T = len(K)
    starts = _segments(K)
    S = int(K.sum())
    Yoh = np.zeros((nF, S), dtype=np.float32)
    Yoh[np.arange(nF)[:, None], inst.Y + starts[None, :]] = 1.0
    folds = inst.folds
    D = hamming(XF, XF, chunk)
    same = folds[:, None] == folds[None, :]
    rows = np.arange(nF)
    best = [(-np.inf, None, None)] * T          # (score, lam, a)
    oof = [None] * T
    for lam in lambdas:
        if time.time() > deadline:
            break
        table = np.exp(-lam * np.arange(256, dtype=np.float32)).astype(np.float32)
        num = np.zeros((nF, S), dtype=np.float32)
        den = np.zeros(nF, dtype=np.float32)
        for s in range(0, nF, chunk):
            W = table[D[s:s + chunk]]
            W[same[s:s + chunk]] = 0.0
            num[s:s + chunk] = W @ Yoh
            den[s:s + chunk] = W.sum(1)
        for t in range(T):
            k = int(K[t])
            sl = slice(starts[t], starts[t] + k)
            y = inst.Y[:, t]
            nt = num[:, sl].astype(np.float64)
            pb = base_oof[t]
            for a in shrink:
                p = (nt + a * pb) / (den[:, None] + a)
                sc = np.mean(np.log(FLOOR + (1 - k * FLOOR) * p[rows, y]))
                if sc > best[t][0]:
                    best[t] = (sc, lam, a)
                    oof[t] = p
    # Hidden rows: one kernel pass per distinct lam actually chosen.
    test = [None] * T
    DH = hamming(XH, XF, chunk)
    for lam in sorted({b[1] for b in best if b[1] is not None}):
        table = np.exp(-lam * np.arange(256, dtype=np.float32)).astype(np.float32)
        num = np.zeros((nH, S), dtype=np.float32)
        den = np.zeros(nH, dtype=np.float32)
        for s in range(0, nH, chunk):
            W = table[DH[s:s + chunk]]
            num[s:s + chunk] = W @ Yoh
            den[s:s + chunk] = W.sum(1)
        for t in range(T):
            if best[t][1] != lam:
                continue
            k = int(K[t])
            sl = slice(starts[t], starts[t] + k)
            a = best[t][2]
            test[t] = (num[:, sl].astype(np.float64) + a * base_test[t]) / (
                den[:, None] + a)
    for t in range(T):                      # nothing ran: fall back to base
        if oof[t] is None or test[t] is None:
            oof[t], test[t] = base_oof[t], base_test[t]
    return oof, test, best
