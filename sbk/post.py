"""Per-item post-processing of the blended prediction, chosen by OOF score.

  temperature  q ∝ p ** (1 / tau)  (tau > 1 flattens, < 1 sharpens)
  ordinal      on items whose options are mostly numeric bands, mix each
               numeric option with its neighbours in numeric order:
               s_i = (1 - b) q_i + b/2 (q_{i-1} + q_{i+1}), reflected ends,
               total mass preserved; non-numeric options (e.g. 'Not
               answered', NA_GATED) are untouched.

The (tau, b) pair with the best out-of-fold log score (after the grader's
floor-and-mix) is applied to the test predictions; (1, 0) is always a
candidate, so an item never gets worse on its own OOF rows.
"""
import re

import numpy as np

from .core import FLOOR

TAUS = (0.8, 0.9, 1.0, 1.15, 1.3)
BETAS = (0.0, 0.1, 0.2, 0.35)


def _num(v):
    m = re.match(r"^\s*([<>]=?)?\s*(-?\d+(?:\.\d+)?)", str(v))
    return float(m.group(2)) if m else None


def ordinal_index(opts):
    """Indices of numeric options, sorted by value; [] if not ordinal."""
    vals = [(i, _num(o)) for i, o in enumerate(opts)]
    num = [(i, x) for i, x in vals if x is not None]
    if len(num) < 3 or len(num) < 0.6 * len(opts):
        return []
    return [i for i, _ in sorted(num, key=lambda t: t[1])]


def _smooth(P, idx, b):
    if b == 0 or len(idx) < 3:
        return P
    Q = P.copy()
    sub = P[:, idx]
    left = np.concatenate([sub[:, :1], sub[:, :-1]], axis=1)
    right = np.concatenate([sub[:, 1:], sub[:, -1:]], axis=1)
    Q[:, idx] = (1 - b) * sub + 0.5 * b * (left + right)
    return Q


def _temper(P, tau):
    if tau == 1.0:
        return P
    Q = np.power(np.clip(P, 1e-12, None), 1.0 / tau)
    return Q / Q.sum(1, keepdims=True)


def _score(P, y):
    k = P.shape[1]
    P = P / P.sum(1, keepdims=True)
    return float(np.mean(np.log(FLOOR + (1 - k * FLOOR) * P[np.arange(y.size), y])))


def postprocess(oof, test, y, opts):
    idx = ordinal_index(opts)
    betas = BETAS if idx else (0.0,)
    best = (_score(oof, y), 1.0, 0.0)
    for tau in TAUS:
        Pt = _temper(oof, tau)
        for b in betas:
            if tau == 1.0 and b == 0.0:
                continue
            sc = _score(_smooth(Pt, idx, b), y)
            if sc > best[0] + 1e-6:
                best = (sc, tau, b)
    _, tau, b = best
    return _smooth(_temper(test, tau), idx, b), (tau, b)
