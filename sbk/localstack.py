"""Blend weights that depend on how close a respondent's nearest look-alike is.

For every row we measure the Hamming distance (number of differing GIVEN
columns) to its nearest visible row: for visible rows, among visible rows in
other folds (the rows its out-of-fold predictions could use); for hidden
rows, among all visible rows. Rows are put in bins (0, 1, 2+ differences)
and, per item, the convex blend weights are fitted separately in each bin on
the out-of-fold predictions, shrunk toward the item's global weights. A
respondent whose GIVEN profile has an exact match among the visible rows
can then lean more on the neighbour learners, and one with no close match
more on the smooth learners. Uses only the GIVEN block.
"""
import numpy as np

from .core import fit_stack, mix, normalize
from .knn import hamming


def nn_distance(inst, chunk=1024, extra_folds=()):
    """`extra_folds`: other learners' fold assignments over the visible rows.
    A visible row's look-alike only counts if it sits outside the row's fold
    in every assignment, i.e. if every learner's out-of-fold prediction for
    the row could have used it (as all of them can for a hidden row)."""
    XF, XH = inst.XF, inst.XH
    fold_sets = [inst.folds] + [np.asarray(f) for f in extra_folds]
    nF = XF.shape[0]
    dF = np.full(nF, 255, dtype=np.uint8)
    for s in range(0, nF, chunk):
        D = hamming(XF[s:s + chunk], XF)
        for folds in fold_sets:
            D[folds[s:s + chunk][:, None] == folds[None, :]] = 255
        dF[s:s + chunk] = D.min(1)
    dH = np.full(XH.shape[0], 255, dtype=np.uint8)
    for s in range(0, XH.shape[0], chunk):
        dH[s:s + chunk] = hamming(XH[s:s + chunk], XF).min(1)
    return dF, dH


def bins_of(d, cuts=(0, 1)):
    b = np.full(d.shape[0], len(cuts), dtype=np.int64)
    for i, c in reversed(list(enumerate(cuts))):
        b[d <= c] = i
    return b


def local_blend(oofs, tests, y, bF, bH, w_global, min_rows=300,
                shrink=0.02):
    """Returns (blended OOF, blended test, {bin: weights})."""
    oof = normalize(mix(oofs, w_global))
    test = normalize(mix(tests, w_global))
    ws = {}
    for b in np.unique(bF):
        mF = bF == b
        mH = bH == b
        if mF.sum() < min_rows or not mH.any():
            continue
        w = fit_stack([o[mF] for o in oofs], y[mF], shrink_to=w_global,
                      shrink=shrink)
        ws[int(b)] = w
        oof[mF] = normalize(mix([o[mF] for o in oofs], w))
        test[mH] = normalize(mix([t[mH] for t in tests], w))
    return oof, test, ws


def _blend_rows(oofs, y, rows_fit, rows_eval, bF, w_fit_global, min_rows,
                shrink):
    """Fit global and per-group weights on `rows_fit`; return mean log score
    on `rows_eval` for (global, grouped)."""
    from .core import logscore
    wg = fit_stack([o[rows_fit] for o in oofs], y[rows_fit])
    pg = normalize(mix([o[rows_eval] for o in oofs], wg))
    pb = pg.copy()
    bf, be = bF[rows_fit], bF[rows_eval]
    for b in np.unique(be):
        mf = bf == b
        me = be == b
        if mf.sum() < min_rows:
            continue
        wb = fit_stack([o[rows_fit][mf] for o in oofs], y[rows_fit][mf],
                       shrink_to=wg, shrink=shrink)
        pb[me] = normalize(mix([o[rows_eval][me] for o in oofs], wb))
    ye = y[rows_eval]
    return logscore(pg, ye), logscore(pb, ye)


def gated_blend(oofs, tests, y, bF, bH, w_global, folds, min_rows=300,
                shrink=0.02, margin=0.0):
    """local_blend, used for this item only if a two-way split of the
    visible rows (by fold parity) shows that per-group weights fitted on
    one half beat the global weights on the other half. Returns (OOF,
    test, used)."""
    A = np.flatnonzero(folds % 2 == 0)
    B = np.flatnonzero(folds % 2 == 1)
    gain = 0.0
    if A.size >= min_rows and B.size >= min_rows:
        for fit_rows, eval_rows in ((A, B), (B, A)):
            g, b = _blend_rows(oofs, y, fit_rows, eval_rows, bF, w_global,
                               min_rows // 2, shrink)
            gain += (b - g) * eval_rows.size
    if gain > margin * (A.size + B.size):
        oof, test, _ = local_blend(oofs, tests, y, bF, bH, w_global,
                                   min_rows=min_rows, shrink=shrink)
        return oof, test, True
    return normalize(mix(oofs, w_global)), normalize(mix(tests, w_global)), False
