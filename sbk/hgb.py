"""Per-item gradient-boosted trees (sklearn HistGradientBoostingClassifier,
pre-installed in the hosted image). GIVEN columns are native categoricals.
Cross-fitted on the shared folds; test = mean of the fold models.
"""
import time
import warnings

import numpy as np


def hgb_learner(inst, lr=0.1, max_iter=200, leaves=15, l2=1.0, min_leaf=20,
                deadline=None, items=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    deadline = deadline or (time.time() + 1e9)
    XF, XH = inst.XF, inst.XH
    cat = np.ones(XF.shape[1], dtype=bool)
    T = len(inst.targets)
    items = range(T) if items is None else items
    oofs, tests = [None] * T, [None] * T
    for t in items:
        if time.time() > deadline:
            break
        y = inst.Y[:, t]
        k = int(inst.K[t])
        if np.unique(y).size < 2:
            continue
        oof = np.zeros((y.size, k))
        test = np.zeros((XH.shape[0], k))
        ok = True
        for f in range(inst.n_folds):
            tr, va = inst.folds != f, inst.folds == f
            if np.unique(y[tr]).size < 2:
                ok = False
                break
            m = None
            for early in (True, False):
                try:
                    m = HistGradientBoostingClassifier(
                        learning_rate=lr, max_iter=max_iter if early else 60,
                        max_leaf_nodes=leaves, l2_regularization=l2,
                        min_samples_leaf=min_leaf, categorical_features=cat,
                        early_stopping=early, validation_fraction=0.1,
                        n_iter_no_change=15, random_state=f)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m.fit(XF[tr], y[tr])
                    break
                except Exception:
                    m = None
            if m is None:
                ok = False
                break
            cls = m.classes_
            oof[np.ix_(va, cls)] = m.predict_proba(XF[va])
            test[:, cls] += m.predict_proba(XH) / inst.n_folds
        if ok:
            oofs[t], tests[t] = oof, test
    return oofs, tests
