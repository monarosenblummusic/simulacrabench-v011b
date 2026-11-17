"""TabICL v2 classifier as a per-item learner (GPU only).

The tabicl 2.2.0 package (BSD-3-Clause, Soda team @ Inria) is vendored
under _vendor/ and the v2 classifier weights (jingang/TabICL, commit
4dcd344ece2c00be9e831fdd35bed57b5ad83e19, file
tabicl-classifier-v2-20260212.ckpt, BSD-3-Clause) are bundled under
weights/, so nothing is installed or downloaded at runtime.

Same protocol as the TabPFN learner: per item, hardest first, cross-fitted
on its own folds; each fold model predicts its held-out fold and the hidden
rows in one call; test = mean over fold models (plus an optional full-data
fit). TabICL handles more than 10 classes itself. GIVEN columns are passed
as pandas categoricals so TabICL encodes them as categories.
"""
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "_vendor")
FALLBACK = os.path.join(HERE, "_fallback")   # psutil / tqdm stand-ins, last on sys.path
CKPT = os.path.join(HERE, "weights", "tabicl-classifier-v2-20260212.ckpt")


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


def _frame(X, cols):
    import pandas as pd
    df = pd.DataFrame({("c%d" % j): pd.Categorical(X[:, j].astype(np.int64))
                       for j in cols})
    return df


_MODEL_CACHE = {}


def _cached_classifier(base):
    """TabICLClassifier whose fit() reuses one loaded model per checkpoint and
    device instead of re-reading the checkpoint for every fit. The model is
    only used for inference (eval mode; forward passes keep no state)."""
    class CachedTabICL(base):
        def _load_model(self):
            key = (str(self.model_path), str(self.checkpoint_version),
                   str(getattr(self, "device_", None)))
            hit = _MODEL_CACHE.get(key)
            if hit is None:
                super()._load_model()
                self.model_.to(self.device_)
                _MODEL_CACHE[key] = (self.model_, self.model_config_,
                                     self.model_path_)
            else:
                self.model_, self.model_config_, self.model_path_ = hit
    CachedTabICL.__name__ = base.__name__
    return CachedTabICL


def tabicl_learner(inst, base_oof, base_test, deadline, order=None,
                   n_estimators=4, n_folds=3, full_fit=False, max_ctx=20000,
                   seed=0, device="cuda", status=None, temperature=0.9,
                   full_weight=None, max_passes=1, cache_model=True):
    """Anytime version: pass 0 covers the items (hardest first) with
    `n_estimators` members; while the clock allows, passes 1..max_passes-1
    revisit the covered items in the same order with fresh random members on
    the same folds, and each item's OOF and hidden-row predictions become the
    mean over its completed passes (so p passes ~ an ensemble of
    p * n_estimators). An unfinished pass on an item is discarded."""
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    if os.path.isdir(FALLBACK) and FALLBACK not in sys.path:
        sys.path.append(FALLBACK)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from tabicl import TabICLClassifier
    import pandas as pd
    if cache_model:
        TabICLClassifier = _cached_classifier(TabICLClassifier)

    nF, nH = len(inst.F), len(inst.H)
    cols = list(range(inst.X.shape[1]))
    DF = _frame(inst.XF, cols)
    DH = _frame(inst.XH, cols)
    T = len(inst.targets)
    oofs, tests = list(base_oof), list(base_test)
    done = [False] * T
    rng = np.random.default_rng(seed)
    folds = np.random.default_rng(seed + 11).integers(0, n_folds, size=nF)
    splits = [(np.flatnonzero(folds != f), np.flatnonzero(folds == f))
              for f in range(n_folds)]
    if full_fit:
        splits.append((np.arange(nF), np.zeros(0, dtype=np.int64)))
    nm = len(splits)
    # Weight of each split's model in the hidden-row prediction: equal by
    # default; with full_fit and full_weight=w, the all-rows model gets w and
    # the fold models share 1-w.
    wts = [1.0 / nm] * nm
    if full_fit and full_weight is not None:
        nf = nm - 1
        wts = [(1.0 - full_weight) / nf] * nf + [float(full_weight)]
    order = list(range(T)) if order is None else list(order)
    sums = {}                                  # t -> [oof_sum, test_sum, n]

    def one_pass(t, pas):
        y = inst.Y[:, t]
        k = int(inst.K[t])
        oof = np.zeros((nF, k))
        test = np.zeros((nH, k))
        for si, (tr, va) in enumerate(splits):
            if time.time() > deadline:
                return None
            if tr.size > max_ctx:
                tr = np.sort(rng.choice(tr, max_ctx, replace=False))
            if np.unique(y[tr]).size < 2:
                return None
            clf = TabICLClassifier(
                n_estimators=n_estimators, model_path=CKPT,
                allow_auto_download=False, device=device,
                softmax_temperature=temperature,
                random_state=seed + si + 1000 * pas, verbose=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(DF.iloc[tr].reset_index(drop=True), y[tr])
                q = pd.concat([DF.iloc[va], DH], ignore_index=True)
                p = clf.predict_proba(q)
            cls = np.asarray(clf.classes_, dtype=np.int64)
            full = np.zeros((p.shape[0], k))
            full[:, cls] = p
            oof[va] = full[:va.size]
            test += full[va.size:] * wts[si]
        if not (np.isfinite(oof).all() and np.isfinite(test).all()):
            return None
        return oof, test

    for pas in range(max(1, int(max_passes))):
        todo = [t for t in order if (t in sums) == (pas > 0)]
        if pas == 0:
            todo = [t for t in order if np.unique(inst.Y[:, t]).size >= 2]
        for t in todo:
            if time.time() > deadline:
                break
            r = one_pass(t, pas)
            if r is None:
                continue
            if t in sums:
                sums[t][0] += r[0]
                sums[t][1] += r[1]
                sums[t][2] += 1
            else:
                sums[t] = [r[0].copy(), r[1].copy(), 1]
                done[t] = True
                if status is not None:
                    status["items"] = status.get("items", 0) + 1
            n = sums[t][2]
            oofs[t], tests[t] = sums[t][0] / n, sums[t][1] / n
            if status is not None:
                status["fits"] = status.get("fits", 0) + 1
        if status is not None:
            status["passes"] = pas + 1 if any(sums[t][2] > pas for t in sums) else pas
        if time.time() > deadline:
            break
    return oofs, tests, done
