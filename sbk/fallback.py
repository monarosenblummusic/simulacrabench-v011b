"""SimulacraBench submission v001.

Per PREDICT item, p(answer | GIVEN block) by hierarchical Dirichlet backoff:

    p_0        = smoothed crowd marginal
    p_l(y | s) = (n_l(s, y) + a_l * p_{l-1}(y)) / (n_l(s) + a_l)

where s is the stratum formed by the first l GIVEN columns of a per-item
ladder. The ladder and every a_l are chosen greedily by 5-fold
cross-validated log score on the visible rows, using the grader's own
floor-and-mix transform.

The gate sentinel (NA_GATED) is modelled as an ordinary class from the
visible rows, never as a hard indicator: the UNHCR schema states that gates
agree with who actually holds a value only ~83% of the time (median), and two
of its gates are disjunctive with only the first parent named. A GIVEN gate
parent is still forced to the top of the ladder, so where the gate does hold
exactly the counts reproduce it.

Never raises: per-item failures fall back to the smoothed marginal, a global
failure falls back to uniform vectors of the right width.
"""

import time

import numpy as np
import pandas as pd

_T0 = time.time()

FLOOR = 1e-3                      # the grader's mix-with-flat floor
N_FOLDS = 5
MAX_LEVELS = 5
ALPHAS = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
MIN_GAIN = 2e-4                   # nats per training row to accept a level
MIN_ROWS_FOR_CV = 30
SEED = 20261122

# Wall-clock guard. The grader's budget covers all three predict() calls.
BUDGET = {1: 900.0, 2: 3600.0}
SAFETY = 0.55                     # stop searching ladders past this share


def _options(schema, item):
    rec = schema["items"][item]
    opts = list(rec["values"])
    if rec.get("gate"):
        opts.append(schema.get("gated_value", "NA_GATED"))
    return opts


def _codes(series, levels):
    """Codes into `levels`; -1 for NaN or anything not listed."""
    cat = pd.Categorical(series.astype(object), categories=levels)
    return np.asarray(cat.codes, dtype=np.int64)


def _floor_mix(p):
    k = p.shape[-1]
    return FLOOR + (1.0 - k * FLOOR) * p


def _phase(frame, schema, targets):
    split = schema.get("split", {})
    n_visible = int((~frame[targets].isna().any(axis=1)).sum()) if targets else 0
    n_train, n_dev = split.get("n_train", -1), split.get("n_dev", -1)
    return 2 if n_visible == n_train + n_dev and n_dev > 0 else 1


def _fit_item(y, X, cards, folds, forced, k, deadline):
    """Greedy ladder search. Returns [(column_index, alpha), ...] and p0 prior.

    y      (n,) codes in [0, k)
    X      (n, d) GIVEN codes, each column in [0, cards[j])
    folds  (n,) fold ids in [0, N_FOLDS)
    """
    n = y.size
    prior_strength = 0.5 * k                         # half a count per option
    flat = np.full(k, 1.0 / k)

    # Level 0, out-of-fold.
    tot = np.bincount(y, minlength=k).astype(float)
    per_fold = np.zeros((N_FOLDS, k))
    np.add.at(per_fold, (folds, y), 1.0)
    out = tot[None, :] - per_fold                    # counts without fold f
    p0_fold = (out + prior_strength * flat) / (out.sum(1, keepdims=True)
                                               + prior_strength)
    p_cv = p0_fold[folds]                            # (n, k)
    rows = np.arange(n)
    best_score = np.mean(np.log(_floor_mix(p_cv)[rows, y]))

    ladder = []
    key = np.zeros(n, dtype=np.int64)
    used = set()
    for level in range(MAX_LEVELS):
        if time.time() > deadline:
            break
        candidates = [j for j in range(X.shape[1]) if j not in used]
        if level == 0 and forced is not None and forced not in used:
            candidates = [forced]
        best = None
        n_strata = np.unique(key).size
        for j in candidates:
            if cards[j] < 2:
                continue
            new_key = key * cards[j] + X[:, j]
            uniq, inv = np.unique(new_key, return_inverse=True)
            m = uniq.size
            if m <= n_strata:                       # column adds no split
                continue
            cell = inv * k + y
            total = np.bincount(cell, minlength=m * k).reshape(m, k)
            fold_cnt = np.bincount(folds * (m * k) + cell,
                                   minlength=N_FOLDS * m * k
                                   ).reshape(N_FOLDS, m, k)
            c_out = total[inv] - fold_cnt[folds, inv]          # (n, k)
            n_out = c_out.sum(1)                               # (n,)
            c_y = c_out[rows, y]
            p_y_prev = p_cv[rows, y]
            # Score every alpha at once on the true class only.
            num = c_y[:, None] + ALPHAS[None, :] * p_y_prev[:, None]
            den = n_out[:, None] + ALPHAS[None, :]
            p_y = num / den
            scores = np.mean(np.log(FLOOR + (1 - k * FLOOR) * p_y), axis=0)
            a = int(np.argmax(scores))
            if best is None or scores[a] > best[0]:
                best = (scores[a], j, ALPHAS[a], inv.astype(np.int64),
                        c_out, n_out)
        if best is None:
            break
        gain = best[0] - best_score
        if gain < MIN_GAIN and not (level == 0 and forced is not None
                                    and gain > 0):
            break
        score, j, alpha, new_key, c_out, n_out = best
        p_cv = (c_out + alpha * p_cv) / (n_out[:, None] + alpha)
        best_score = score
        key = new_key
        used.add(j)
        ladder.append((j, float(alpha)))
    return ladder, prior_strength


def _apply(y, X_fit, X_new, cards, k, ladder, prior_strength):
    """Probabilities for X_new from the fitted ladder, using all fit rows."""
    flat = np.full(k, 1.0 / k)
    tot = np.bincount(y, minlength=k).astype(float)
    p0 = (tot + prior_strength * flat) / (tot.sum() + prior_strength)
    p = np.tile(p0, (X_new.shape[0], 1))
    key_fit = np.zeros(y.size, dtype=np.int64)
    key_new = np.zeros(X_new.shape[0], dtype=np.int64)
    alive = np.ones(X_new.shape[0], dtype=bool)     # stratum seen so far
    for j, alpha in ladder:
        key_fit = key_fit * cards[j] + X_fit[:, j]
        key_new = key_new * cards[j] + X_new[:, j]
        uniq, inv_fit = np.unique(key_fit, return_inverse=True)
        key_fit = inv_fit.astype(np.int64)           # dense 0..m-1
        m = uniq.size
        pos = np.clip(np.searchsorted(uniq, key_new), 0, m - 1)
        alive &= uniq[pos] == key_new
        key_new = np.where(alive, pos, 0).astype(np.int64)
        table = np.bincount(key_fit * k + y, minlength=m * k
                            ).reshape(m, k).astype(float)
        c = np.where(alive[:, None], table[key_new], 0.0)
        # An unseen stratum has no counts, so it keeps the coarser estimate.
        p = (c + alpha * p) / (c.sum(1, keepdims=True) + alpha)
    return p


def _uniform_all(frame, schema, targets):
    widths = {t: len(_options(schema, t)) for t in targets}
    grid = frame[targets].isna().to_numpy()
    out = []
    for r in range(grid.shape[0]):
        for c, t in enumerate(targets):
            if grid[r, c]:
                out.append([1.0] * widths[t])
    return out


def predict(frame, schema):
    items = [n for n, r in schema["items"].items()
             if r.get("class") in ("GIVEN", "PREDICT")]
    targets = [n for n in items if schema["items"][n]["class"] == "PREDICT"]
    targets = [t for t in targets if t in frame.columns]
    try:
        return _predict(frame, schema, items, targets)
    except Exception:
        return _uniform_all(frame, schema, targets)


def _predict(frame, schema, items, targets):
    t_start = time.time()
    given = [n for n in items if schema["items"][n]["class"] == "GIVEN"
             and n in frame.columns]
    phase = _phase(frame, schema, targets)
    # Spend at most this instrument's share of the budget on ladder search;
    # the three instruments differ a lot in size, so weight by target count.
    share = {213: 0.6, 76: 0.25, 12: 0.15}.get(len(targets), 0.34)
    deadline = t_start + SAFETY * BUDGET[phase] * share

    n = len(frame)
    # GIVEN design: codes over values (+ sentinel if gated), unknown -> last.
    X = np.zeros((n, len(given)), dtype=np.int64)
    cards = []
    for j, g in enumerate(given):
        levels = _options(schema, g)
        c = _codes(frame[g], levels)
        c = np.where(c < 0, len(levels), c)
        X[:, j] = c
        cards.append(len(levels) + 1)
    cards = np.asarray(cards, dtype=np.int64)

    hidden_mask = frame[targets].isna().to_numpy()          # (n, T)
    rng = np.random.default_rng(SEED)

    preds = {}
    for t_i, item in enumerate(targets):
        opts = _options(schema, item)
        k = len(opts)
        hid = np.flatnonzero(hidden_mask[:, t_i])
        if hid.size == 0:
            continue
        try:
            y_all = _codes(frame[item], opts)
            fit = np.flatnonzero((~hidden_mask[:, t_i]) & (y_all >= 0))
            y = y_all[fit]
            if fit.size == 0:
                preds[item] = np.full((hid.size, k), 1.0 / k)
                continue
            gate = schema["items"][item].get("gate")
            forced = None
            if gate and gate.get("parent") in given:
                forced = given.index(gate["parent"])
            ladder, prior_strength = [], 0.5 * k
            if fit.size >= MIN_ROWS_FOR_CV and k > 1 and given:
                folds = rng.integers(0, N_FOLDS, size=fit.size)
                ladder, prior_strength = _fit_item(
                    y, X[fit], cards, folds, forced, k, deadline)
            p = _apply(y, X[fit], X[hid], cards, k, ladder, prior_strength)
        except Exception:
            try:
                y_all = _codes(frame[item], opts)
                y = y_all[(~hidden_mask[:, t_i]) & (y_all >= 0)]
                tot = np.bincount(y, minlength=k).astype(float) + 0.5
                p = np.tile(tot / tot.sum(), (hid.size, 1))
            except Exception:
                p = np.full((hid.size, k), 1.0 / k)
        p = np.nan_to_num(np.asarray(p, dtype=float), nan=0.0,
                          posinf=0.0, neginf=0.0)
        p = np.clip(p, 0.0, None)
        s = p.sum(1, keepdims=True)
        bad = ~(s[:, 0] > 0)
        if bad.any():
            p[bad] = 1.0 / k
            s = p.sum(1, keepdims=True)
        preds[item] = p / s

    # Canonical order: rows top to bottom, items in schema order.
    pos = {t: 0 for t in targets}
    out = []
    for r in range(n):
        row = hidden_mask[r]
        if not row.any():
            continue
        for c, t in enumerate(targets):
            if row[c]:
                out.append(preds[t][pos[t]].tolist())
                pos[t] += 1
    return out
