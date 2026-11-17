"""Hierarchical Dirichlet backoff ladders (v001 learner), returning OOF too.

Options beyond v001:
  first      force this GIVEN column index as level 1 (for parallel ladders)
  exclude    GIVEN column indices never used
  ordinal    smooth the level-0 base measure across adjacent options
"""
import time

import numpy as np

from .core import FLOOR

ALPHAS = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
# Deep levels may need to be trusted almost fully (near-identical profiles)
# or switched off entirely (alpha so large the level is a no-op).
DEEP_ALPHAS = np.array([0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0,
                        32.0, 64.0, 128.0, 512.0, 4096.0, 1e6])


def _search(y, X, cards, folds, n_folds, k, forced, max_levels, min_gain,
            deadline, base_cv, exclude=(), alphas=None):
    alphas = ALPHAS if alphas is None else alphas
    n = y.size
    rows = np.arange(n)
    p_cv = base_cv
    best_score = np.mean(np.log(FLOOR + (1 - k * FLOOR) * p_cv[rows, y]))
    ladder, used = [], set(exclude)
    key = np.zeros(n, dtype=np.int64)
    for level in range(max_levels):
        if time.time() > deadline:
            break
        cands = [j for j in range(X.shape[1]) if j not in used]
        if level == 0 and forced is not None:
            cands = [forced]
        best = None
        n_strata = np.unique(key).size
        for j in cands:
            if cards[j] < 2:
                continue
            new_key = key * cards[j] + X[:, j]
            uniq, inv = np.unique(new_key, return_inverse=True)
            m = uniq.size
            if m <= n_strata:
                continue
            cell = inv * k + y
            total = np.bincount(cell, minlength=m * k).reshape(m, k)
            fc = np.bincount(folds * (m * k) + cell,
                             minlength=n_folds * m * k).reshape(n_folds, m, k)
            c_out = total[inv] - fc[folds, inv]
            n_out = c_out.sum(1)
            num = c_out[rows, y][:, None] + alphas[None, :] * p_cv[rows, y][:, None]
            den = n_out[:, None] + alphas[None, :]
            sc = np.mean(np.log(FLOOR + (1 - k * FLOOR) * num / den), axis=0)
            a = int(np.argmax(sc))
            if best is None or sc[a] > best[0]:
                best = (sc[a], j, alphas[a], inv.astype(np.int64), c_out, n_out)
        if best is None:
            break
        gain = best[0] - best_score
        if gain < min_gain and not (level == 0 and forced is not None
                                    and gain > 0):
            break
        sc, j, alpha, inv, c_out, n_out = best
        p_cv = (c_out + alpha * p_cv) / (n_out[:, None] + alpha)
        best_score = sc
        key = inv
        used.add(j)
        ladder.append((j, float(alpha)))
    return ladder, p_cv, key


def _base(y, k, folds, n_folds, ordinal_bw=0.0):
    """Level-0: smoothed marginal, out-of-fold and full."""
    strength = 0.5 * k
    flat = np.full(k, 1.0 / k)
    tot = np.bincount(y, minlength=k).astype(float)
    per = np.zeros((n_folds, k))
    np.add.at(per, (folds, y), 1.0)
    out = tot[None, :] - per
    cv = (out + strength * flat) / (out.sum(1, keepdims=True) + strength)
    full = (tot + strength * flat) / (tot.sum() + strength)
    return cv[folds], full


def _apply(y, XF, XH, cards, k, ladder, p0):
    p = np.tile(p0, (XH.shape[0], 1))
    kf = np.zeros(y.size, dtype=np.int64)
    kn = np.zeros(XH.shape[0], dtype=np.int64)
    alive = np.ones(XH.shape[0], dtype=bool)
    for j, alpha in ladder:
        kf = kf * cards[j] + XF[:, j]
        kn = kn * cards[j] + XH[:, j]
        uniq, inv = np.unique(kf, return_inverse=True)
        kf = inv.astype(np.int64)
        m = uniq.size
        pos = np.clip(np.searchsorted(uniq, kn), 0, m - 1)
        alive &= uniq[pos] == kn
        kn = np.where(alive, pos, 0).astype(np.int64)
        table = np.bincount(kf * k + y, minlength=m * k).reshape(m, k).astype(float)
        c = np.where(alive[:, None], table[kn], 0.0)
        p = (c + alpha * p) / (c.sum(1, keepdims=True) + alpha)
    return p


def _extend(y, X, cards, folds, n_folds, k, ladder, p_cv, key, order,
            deadline):
    """Append every remaining column, in `order`, down to the full profile.
    Each level's alpha is chosen by out-of-fold log score from DEEP_ALPHAS;
    a level that does not help gets alpha ~1e6, i.e. it is a no-op."""
    n = y.size
    rows = np.arange(n)
    used = {j for j, _ in ladder}
    ladder = list(ladder)
    for j in order:
        if j in used or cards[j] < 2:
            continue
        if time.time() > deadline:
            break
        new_key = key * cards[j] + X[:, j]
        uniq, inv = np.unique(new_key, return_inverse=True)
        m = uniq.size
        inv = inv.astype(np.int64)
        cell = inv * k + y
        total = np.bincount(cell, minlength=m * k).reshape(m, k)
        fc = np.bincount(folds * (m * k) + cell,
                         minlength=n_folds * m * k).reshape(n_folds, m, k)
        c_out = total[inv] - fc[folds, inv]
        n_out = c_out.sum(1)
        num = (c_out[rows, y][:, None]
               + DEEP_ALPHAS[None, :] * p_cv[rows, y][:, None])
        den = n_out[:, None] + DEEP_ALPHAS[None, :]
        sc = np.mean(np.log(FLOOR + (1 - k * FLOOR) * num / den), axis=0)
        a = float(DEEP_ALPHAS[int(np.argmax(sc))])
        p_cv = (c_out + a * p_cv) / (n_out[:, None] + a)
        key = inv
        used.add(j)
        ladder.append((j, a))
    return ladder, p_cv


def ladder_learner(inst, max_levels=5, min_gain=2e-4, first=None,
                   exclude=(), deadline=None, force_gate_parent=True,
                   X=None, cards=None, deep_order=None, alphas=None):
    """Returns (oof list, test list, ladders)."""
    X = inst.X if X is None else X
    cards = inst.cards if cards is None else cards
    XF, XH = X[inst.F], X[inst.H]
    deadline = deadline or (time.time() + 1e9)
    oofs, tests, ladders = [], [], []
    for t in range(len(inst.targets)):
        y = inst.Y[:, t]
        k = int(inst.K[t])
        base_cv, p0 = _base(y, k, inst.folds, inst.n_folds)
        forced = first
        g = inst.gates[t]
        if forced is None and force_gate_parent and g and g.get("parent") in inst.given:
            forced = inst.given.index(g["parent"])
        if k > 1 and X.shape[1]:
            lad, p_cv, key = _search(y, XF, cards, inst.folds, inst.n_folds,
                                     k, forced, max_levels, min_gain,
                                     deadline, base_cv, exclude, alphas)
            if deep_order is not None:
                lad, p_cv = _extend(y, XF, cards, inst.folds, inst.n_folds, k,
                                    lad, p_cv, key, deep_order, deadline)
        else:
            lad, p_cv = [], base_cv
        oofs.append(p_cv)
        tests.append(_apply(y, XF, XH, cards, k, lad, p0))
        ladders.append(lad)
    return oofs, tests, ladders
