"""Shared data preparation, scoring and stacking for SimulacraBench learners.

Every learner sees the same thing:
    X  (n, d) int codes of the GIVEN block, column j in [0, cards[j])
    Y  (nF, T) int codes of the PREDICT items on the visible rows
    F  visible row indices, H hidden row indices
    folds (nF,) fold ids shared by all learners, so OOF predictions stack.
and returns, per item t, OOF probabilities on F and test probabilities on H.
"""
import numpy as np
import pandas as pd

FLOOR = 1e-3


def options_for(schema, item):
    rec = schema["items"][item]
    opts = list(rec["values"])
    if rec.get("gate"):
        opts.append(schema.get("gated_value", "NA_GATED"))
    return opts


def codes(series, levels):
    cat = pd.Categorical(series.astype(object), categories=levels)
    return np.asarray(cat.codes, dtype=np.int64)


class Inst:
    """One instrument, encoded."""

    def __init__(self, frame, schema, n_folds=5, seed=0, derive=False):
        items = [n for n, r in schema["items"].items()
                 if r.get("class") in ("GIVEN", "PREDICT")]
        self.schema = schema
        self.targets = [n for n in items
                        if schema["items"][n]["class"] == "PREDICT"
                        and n in frame.columns]
        self.given = [n for n in items
                      if schema["items"][n]["class"] == "GIVEN"
                      and n in frame.columns]
        n = len(frame)
        self.n = n
        X = np.zeros((n, len(self.given)), dtype=np.int64)
        cards = []
        self.given_levels = []
        for j, g in enumerate(self.given):
            lv = options_for(schema, g)
            c = codes(frame[g], lv)
            X[:, j] = np.where(c < 0, len(lv), c)
            cards.append(len(lv) + 1)
            self.given_levels.append(lv)
        self.extra = []
        if derive:
            try:
                from .derive import derived_columns
                cols = derived_columns(frame)
            except Exception:
                cols = {}
            add = []
            for name, vals in cols.items():
                lv = sorted(set(vals.tolist()))
                if len(lv) < 2:
                    continue
                lookup = {v: i for i, v in enumerate(lv)}
                add.append(np.array([lookup[v] for v in vals], dtype=np.int64))
                cards.append(len(lv) + 1)
                self.extra.append(name)
            if add:
                X = np.concatenate([X, np.stack(add, 1)], axis=1)
        self.X = X
        self.cards = np.asarray(cards, dtype=np.int64)
        self.hidden_mask = frame[self.targets].isna().to_numpy()
        vis = ~self.hidden_mask.any(axis=1)
        self.F = np.flatnonzero(vis)
        self.H = np.flatnonzero(self.hidden_mask.any(axis=1))
        self.opts = [options_for(schema, t) for t in self.targets]
        self.K = np.array([len(o) for o in self.opts], dtype=np.int64)
        Y = np.zeros((len(self.F), len(self.targets)), dtype=np.int64)
        for i, t in enumerate(self.targets):
            y = codes(frame[t], self.opts[i])[self.F]
            Y[:, i] = y
        # A visible cell outside the option list should not happen; map it to
        # the modal class so nothing downstream sees -1.
        for i in range(Y.shape[1]):
            bad = Y[:, i] < 0
            if bad.any():
                good = Y[~bad, i]
                Y[bad, i] = np.bincount(good).argmax() if good.size else 0
        self.Y = Y
        rng = np.random.default_rng(seed)
        self.n_folds = n_folds
        self.folds = rng.integers(0, n_folds, size=len(self.F))
        self.gates = [schema["items"][t].get("gate") for t in self.targets]
        self.ordinal = [_looks_ordinal(o, g is not None)
                        for o, g in zip(self.opts, self.gates)]

    @property
    def XF(self):
        return self.X[self.F]

    @property
    def XH(self):
        return self.X[self.H]


def _looks_ordinal(opts, gated):
    """True when most options start with a number (bands, scales, counts)."""
    import re
    vals = opts[:-1] if gated else opts
    num = sum(1 for v in vals if re.match(r"^\s*[<>]?=?\s*\d", str(v)))
    return len(vals) >= 3 and num >= max(3, 0.6 * len(vals))


def floor_mix(p):
    k = p.shape[-1]
    return FLOOR + (1.0 - k * FLOOR) * p


def logscore(p, y):
    """Mean log score of rows of p at labels y, after the grader's mix."""
    p = p / p.sum(1, keepdims=True)
    return float(np.mean(np.log(floor_mix(p)[np.arange(y.size), y])))


def skill_from_items(item_logs, K):
    """item_logs: mean log score per item (equal weight, as in the grader)."""
    U = float(np.mean(np.log(K)))
    return 1.0 + float(np.mean(item_logs)) / U


def normalize(p):
    p = np.nan_to_num(np.asarray(p, float), nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0, None)
    s = p.sum(1, keepdims=True)
    bad = ~(s[:, 0] > 0)
    if bad.any():
        p[bad] = 1.0
        s = p.sum(1, keepdims=True)
    return p / s


# ------------------------------------------------------------- stacking ---

def fit_stack(oofs, y, n_iter=200, shrink_to=None, shrink=0.0):
    """Convex weights over M learners maximizing mean log of the mixture at y.

    oofs: list of (n, K) OOF probability matrices. Exponentiated-gradient
    on the simplex; optional shrinkage toward `shrink_to` weights.
    """
    M = len(oofs)
    P = np.stack([floor_mix(normalize(o))[np.arange(y.size), y]
                  for o in oofs], axis=1)                     # (n, M)
    w = np.full(M, 1.0 / M)
    lr = 1.0
    for _ in range(n_iter):
        mix = P @ w
        grad = (P / mix[:, None]).mean(0)
        if shrink_to is not None and shrink > 0:
            grad = grad + shrink * (np.log(shrink_to + 1e-12)
                                    - np.log(w + 1e-12))
        w = w * np.exp(lr * (grad - grad.max()))
        w = w / w.sum()
    return w


def mix(preds, w):
    out = 0.0
    for p, wi in zip(preds, w):
        if wi > 0:
            out = out + wi * normalize(p)
    return out
