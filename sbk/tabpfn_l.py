"""TabPFN v2 classifier as a per-item learner (GPU only).

The tabpfn 2.0.9 package (Apache-2.0, Prior Labs) is vendored under
_vendor/ and the v2 classifier weights (Prior-Labs/TabPFN-v2-clf, commit
275f19d2add9c2c0de38f9ebaded3856599fa2e7, Prior Labs License 1.1) are
bundled under weights/, so nothing is installed or downloaded at runtime.
Built with PriorLabs-TabPFN.

Per item, hardest first (by the base model's out-of-fold loss), cross-fitted
on its own `n_folds` random folds: each fold model predicts its held-out fold
(OOF) and the hidden rows in one call; test = mean over fold models (plus,
with `full_fit`, a model on all visible rows weighted like one more fold).

Items with more than `max_classes` (10) observed classes are fitted on the
9 most frequent classes of the training fold plus one 'other' class; the
'other' probability is split among the remaining classes in proportion to
the base model's probabilities for that row. Stops at the deadline; finished
items are kept, unfinished items keep the base model.
"""
import os
import re
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "_vendor")
CKPT = os.path.join(HERE, "weights", "tabpfn-v2-classifier.ckpt")


def available(device="cuda"):
    if not os.path.exists(CKPT):
        return False
    try:
        import torch
    except Exception:
        return False
    if device == "cuda":
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False
    return True


_BAND = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(?:(?:-|–|to)\s*(\d+(?:\.\d+)?)|(\+|and above))?"
    r"(?:\s*years?\s*old)?\s*$", re.IGNORECASE)
_NUMBERED = re.compile(r"^\s*(\d+)\.\s+\S")


def _band_value(label):
    """Numeric value of a band label ('0', '6-10', '11+', '18 to 24 years
    old', '45 and above'); None for anything else."""
    m = _BAND.match(str(label))
    if not m:
        return None
    a = float(m.group(1))
    if m.group(2):
        return (a + float(m.group(2))) / 2.0
    if m.group(3):
        return a + 1.0
    return a


def tab_features(inst, numeric=True):
    """GIVEN block for TabPFN. With `numeric`, columns whose labels are
    numeric bands (counts, age bands, pay bands), and numbered education
    scales, are passed as numbers (unparseable labels and missing values as
    NaN) so TabPFN can use their order; every other column stays a
    categorical code."""
    X = inst.X.astype(np.float32)
    cat = list(range(X.shape[1]))
    if not numeric:
        return X, cat
    for j, (name, levels) in enumerate(zip(inst.given, inst.given_levels)):
        vals = [_band_value(v) for v in levels]
        ok = sum(v is not None for v in vals)
        is_band = len(levels) >= 3 and ok >= max(3, 0.6 * len(levels))
        if not is_band and "educ" in str(name).lower():
            nums = [_NUMBERED.match(str(v)) for v in levels]
            if sum(m is not None for m in nums) >= max(3, 0.6 * len(levels)):
                vals = [float(m.group(1)) if m else None for m in nums]
                is_band = True
        if not is_band:
            continue
        table = np.array([np.nan if v is None else v for v in vals]
                         + [np.nan] * 2, dtype=np.float32)
        X[:, j] = table[np.clip(inst.X[:, j], 0, len(table) - 1)]
        cat.remove(j)
    return X, cat


def _expand(p, cls, k, top, other, base_rows):
    """Map TabPFN output columns (labels `cls`) back to the item's k options."""
    out = np.zeros((p.shape[0], k))
    if top is None:                          # labels are the option codes
        out[:, cls] = p
        return out
    m = len(top)                             # labels 0..m-1 = top, m = other
    for j, c in enumerate(cls):
        if c < m:
            out[:, top[c]] = p[:, j]
        else:
            share = base_rows[:, other] + 1e-12
            share = share / share.sum(1, keepdims=True)
            out[:, other] = p[:, j][:, None] * share
    return out


def tabpfn_learner(inst, base_oof, base_test, deadline, order=None,
                   n_estimators=2, n_folds=None, full_fit=False,
                   max_ctx=10000, max_classes=10, seed=0, device="cuda",
                   status=None, numeric=False):
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from tabpfn import TabPFNClassifier

    Xall, cat = tab_features(inst, numeric=numeric)
    XF, XH = Xall[inst.F], Xall[inst.H]
    nF, nH = XF.shape[0], XH.shape[0]
    T = len(inst.targets)
    oofs = list(base_oof)
    tests = list(base_test)
    done = [False] * T
    rng = np.random.default_rng(seed)
    if n_folds is None or n_folds == inst.n_folds:
        folds, nf = inst.folds, inst.n_folds
    else:
        nf = int(n_folds)
        folds = np.random.default_rng(seed + 7).integers(0, nf, size=nF)
    order = list(range(T)) if order is None else list(order)
    splits = [(np.flatnonzero(folds != f), np.flatnonzero(folds == f))
              for f in range(nf)]
    if full_fit:
        splits.append((np.arange(nF), np.zeros(0, dtype=np.int64)))
    n_test_models = len(splits)

    for t in order:
        if time.time() > deadline:
            break
        y = inst.Y[:, t]
        k = int(inst.K[t])
        if np.unique(y).size < 2:
            continue
        oof = np.zeros((nF, k))
        test = np.zeros((nH, k))
        ok = True
        for si, (tr, va) in enumerate(splits):
            if time.time() > deadline:
                ok = False
                break
            if tr.size > max_ctx:
                tr = np.sort(rng.choice(tr, max_ctx, replace=False))
            ytr = y[tr]
            counts = np.bincount(ytr, minlength=k)
            present = np.flatnonzero(counts)
            if present.size < 2:
                ok = False
                break
            top = other = None
            lab = ytr
            if present.size > max_classes:
                ranked = present[np.argsort(-counts[present], kind="stable")]
                top = ranked[:max_classes - 1]
                other = np.sort(ranked[max_classes - 1:])
                code = np.full(k, max_classes - 1, dtype=np.int64)
                code[top] = np.arange(max_classes - 1)
                lab = code[ytr]
            clf = TabPFNClassifier(
                n_estimators=n_estimators, device=device, model_path=CKPT,
                categorical_features_indices=cat,
                ignore_pretraining_limits=True, random_state=seed + si)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(XF[tr], lab)
                p = clf.predict_proba(np.concatenate([XF[va], XH], axis=0))
            cls = np.asarray(clf.classes_, dtype=np.int64)
            base_rows = None
            if top is not None:
                base_rows = np.concatenate([base_oof[t][va], base_test[t]], 0)
            full = _expand(p, cls, k, top, other, base_rows)
            oof[va] = full[:va.size]
            test += full[va.size:] / n_test_models
        if ok and np.isfinite(oof).all() and np.isfinite(test).all():
            oofs[t], tests[t], done[t] = oof, test, True
            if status is not None:
                status["items"] = status.get("items", 0) + 1
    return oofs, tests, done
