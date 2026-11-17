"""SimulacraBench submission v011b.

Learners per instrument, all fitted on the visible rows inside predict(),
then blended separately for every PREDICT item by convex weights that
maximise the out-of-fold (OOF) log score after the grader's floor-and-mix:

  lad   Dirichlet backoff ladder over GIVEN columns (v001).
  knn   Neighbour smoother, weight exp(-lam * #differing GIVEN columns),
        shrunk toward the ladder; lam and shrinkage chosen per item (v002).
  knnw  Neighbour smoother with a learned per-column metric: column weights
        fitted by coordinate search on OOF log score, shared by all items of
        the instrument; per-item scale and shrinkage (v003).
  mlp   Multi-task softmax network, one-hot GIVEN block -> shared hidden
        layer -> one softmax head per item (v002).
  hgb   Per-item gradient-boosted trees, GIVEN columns as native
        categoricals, test = mean of fold models (v003).
  tab   TabPFN v2 classifier per item on the GPU (vendored tabpfn 2.0.9 and
        bundled v2 weights, Prior-Labs/TabPFN-v2-clf commit 275f19d2), run
        in a background thread alongside the CPU learners, hardest items
        first; items it does not reach keep the ladder (v005). From v006 it
        covers items with more than 10 options (top 9 classes + 'other',
        split by the ladder), uses its own 3 folds in Development (5 plus a
        full-data fit in the Final), and unused time from an instrument that
        finishes early passes to the ones still to run. From v007 the
        TabPFN ensemble size is set per instrument (larger for UNHCR), and
        optionally (TAB_NUMERIC) GIVEN columns whose labels are numeric bands
        (household counts, age, pay) or a numbered education scale are given
        to TabPFN as numbers instead of categories.
        Built with PriorLabs-TabPFN.
  icl   TabICL v2 classifier per item on the GPU (vendored tabicl 2.2.0,
        BSD-3-Clause; bundled weights jingang/TabICL commit 4dcd344e,
        tabicl-classifier-v2-20260212.ckpt, BSD-3-Clause), same protocol as
        tab: own folds, hardest items first, GIVEN columns as categories,
        all options handled natively (v008). It runs in its own GPU thread.
        v009: anytime passes (cover every item first, then add fresh
        ensemble members to the hardest items while the clock allows; each
        item's prediction is the mean over its completed passes) and one
        loaded model reused across fits.

Options in this file: DERIVE adds banded household-roster totals (UNHCR)
and a reservation-wage gap (World Bank) as extra GIVEN columns; POST applies
per-item temperature and ordinal smoothing chosen by OOF score; HEAVY raises
seeds, widths, metric-search size and tree iterations within the budget.

All learners use the same 5 folds. NA_GATED is an ordinary class. A clock
keeps the run inside the phase budget; a learner that fails or runs out of
time drops out of the blend (per item for hgb), and the v001 model and then
uniform vectors are the last resorts.

Configuration of this file: TabICL v2 anytime: ensemble 4 per pass, up to 4 passes; 3 folds in Development, 5 folds + full fit in the Final; shares UNHCR 0.56 / World Bank 0.29 / UNICEF 0.15
"""
import os
import sys
import threading
import time

_START = time.time()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from sbk import fallback  # noqa: E402
from sbk.core import Inst, fit_stack, mix, normalize  # noqa: E402
from sbk.hgb import hgb_learner  # noqa: E402
from sbk.knn import knn_learner  # noqa: E402
from sbk.knnw import knnw_learner  # noqa: E402
from sbk.ladder import ladder_learner  # noqa: E402
from sbk.mlp import mlp_learner  # noqa: E402
from sbk.post import postprocess  # noqa: E402
from sbk import tabpfn_l  # noqa: E402
from sbk import tabicl_l  # noqa: E402
from sbk import localstack  # noqa: E402

LEARNERS = ["lad", "knn", "knnw", "mlp", "hgb", "icl"]
DERIVE = False
POST = True
HEAVY = False
SEED = 20261122
BUDGET = {1: 900.0, 2: 3600.0}
USE = 0.85                       # share of the phase budget we plan to spend
SHARE = {213: 0.56, 76: 0.29, 12: 0.15}
# TabPFN ensemble size per phase and instrument (keyed by #PREDICT items).
TAB_EST = {1: {213: 4, 76: 2, 12: 2}, 2: {213: 4, 76: 4, 12: 4}}
TAB_NUMERIC = False             # numeric bands given to TabPFN as numbers
TAB_FOLDS = {1: 3, 2: 5}          # TabPFN's own cross-fitting folds
TAB_FULL = {1: False, 2: True}    # also fit TabPFN on all visible rows
# TabICL settings per phase and instrument (keyed by #PREDICT items).
ICL_EST = {1: {213: 4, 76: 4, 12: 4}, 2: {213: 4, 76: 4, 12: 4}}
ICL_FOLDS = {1: {213: 3, 76: 3, 12: 3}, 2: {213: 5, 76: 5, 12: 5}}
ICL_FULL = {1: False, 2: True}
# Weight of the all-rows TabICL model in the hidden-row prediction (None =
# equal weight with the fold models).
ICL_FULLW = {1: None, 2: None}
ICL_TEMP = 0.9
# Anytime TabICL: up to this many passes per item (each pass adds
# ICL_EST members on the same folds; items are covered first, then revisited
# hardest-first while the clock allows).
ICL_PASSES = {1: 4, 2: 4}
# Blend weights fitted separately by distance to the nearest visible GIVEN
# profile (0 / 1 / 2+ differing columns), shrunk toward the global weights.
LOOKALIKE = True
LOOKALIKE_CUTS = (0, 1)
# Per item, the grouped weights are used only if a two-way split of the
# visible rows shows they beat the global weights out of sample; seconds per
# item reserved inside the clock for that check.
LOOKALIKE_GATE = True
LOOKALIKE_RESERVE = 0.12
_SEEN = set()                    # instruments already started in this run


def _phase(inst, schema):
    """2 when TRAIN and DEV are visible (Final), else 1 (Development).

    Primary rule: the visible rows number exactly n_train + n_dev from the
    schema's split block. Backup, used only if the visible count matches
    neither phase (a split block that disagrees with the delivered rows): the
    ratio of hidden to visible rows, 0.075-0.100 in Development and
    0.163-0.212 in the Final for all three instruments."""
    split = schema.get("split", {}) or {}
    n_tr, n_dev = split.get("n_train", -1), split.get("n_dev", -1)
    nF, nH = len(inst.F), len(inst.H)
    if n_dev > 0 and nF == n_tr + n_dev:
        return 2
    if nF == n_tr:
        return 1
    return 2 if nF > 0 and nH / float(nF) > 0.13 else 1


def _emit(inst, preds):
    """Canonical order: rows top to bottom, items in schema order."""
    pos = {h: i for i, h in enumerate(inst.H)}
    out = []
    for r in inst.H:
        row = inst.hidden_mask[r]
        i = pos[r]
        for c in range(len(inst.targets)):
            if row[c]:
                out.append(preds[c][i].tolist())
    return out


def _uniform(frame, schema):
    items = [n for n, r in schema["items"].items()
             if r.get("class") == "PREDICT" and n in frame.columns]
    grid = frame[items].isna().to_numpy()
    width = [len(schema["items"][n]["values"])
             + (1 if schema["items"][n].get("gate") else 0) for n in items]
    return [[1.0] * width[c] for r in range(grid.shape[0])
            for c in range(len(items)) if grid[r, c]]


def predict(frame, schema):
    try:
        return _predict(frame, schema)
    except Exception:
        pass
    try:
        return fallback.predict(frame, schema)
    except Exception:
        return _uniform(frame, schema)


def _ok(lists):
    return all(x is not None and np.isfinite(x).all() for x in lists)


def _predict(frame, schema):
    t0 = time.time()
    inst = Inst(frame, schema, n_folds=5, seed=SEED, derive=DERIVE)
    if len(inst.F) < 50 or len(inst.H) == 0 or not inst.given:
        return fallback.predict(frame, schema)
    phase = _phase(inst, schema)
    budget = BUDGET[phase] * USE
    # Split what is left of the budget among the instruments not yet run, so
    # time an earlier instrument did not use passes forward.
    key = len(inst.targets)
    left = [n for n in SHARE if n not in _SEEN]
    if key in SHARE and key in left:
        share = SHARE[key] / sum(SHARE[n] for n in left)
    else:
        share = 0.33
    _SEEN.add(key)
    remaining = max(30.0, _START + budget - t0)
    deadline = t0 + remaining * share
    if LOOKALIKE and LOOKALIKE_GATE:
        deadline -= min(0.1 * remaining * share,
                        LOOKALIKE_RESERVE * len(inst.targets))

    learners = {}
    lad_oof, lad_test, _ = ladder_learner(inst, deadline=deadline)
    learners["lad"] = (lad_oof, lad_test)
    # Least certain items first, so a clock cut-off costs least.
    loss = [-np.mean(np.log(np.maximum(
        lad_oof[t][np.arange(len(inst.F)), inst.Y[:, t]], 1e-12)))
        for t in range(len(inst.targets))]
    order = list(np.argsort(loss)[::-1])

    # TabPFN runs on the GPU in a background thread while the CPU learners
    # run; it checks the same deadline between folds.
    tab = {}

    def _tab():
        try:
            import torch
            torch.set_num_threads(2)
            tab["res"] = tabpfn_l.tabpfn_learner(
                inst, lad_oof, lad_test, deadline=deadline, order=order,
                n_estimators=TAB_EST[phase].get(len(inst.targets), 2),
                n_folds=TAB_FOLDS[phase], full_fit=TAB_FULL[phase],
                seed=SEED, status=tab, numeric=TAB_NUMERIC)
        except Exception as exc:
            tab["err"] = type(exc).__name__
    worker = None
    if "tab" in LEARNERS and tabpfn_l.available("cuda"):
        worker = threading.Thread(target=_tab, daemon=True)
        worker.start()

    icl = {}

    def _icl():
        try:
            import torch
            torch.set_num_threads(2)
            icl["res"] = tabicl_l.tabicl_learner(
                inst, lad_oof, lad_test, deadline=deadline, order=order,
                n_estimators=ICL_EST[phase].get(len(inst.targets), 4),
                n_folds=ICL_FOLDS[phase].get(len(inst.targets), 3),
                full_fit=ICL_FULL[phase], seed=SEED, status=icl,
                temperature=ICL_TEMP, full_weight=ICL_FULLW[phase],
                max_passes=ICL_PASSES[phase])
        except Exception as exc:
            icl["err"] = type(exc).__name__
    worker2 = None
    if "icl" in LEARNERS and tabicl_l.available("cuda"):
        worker2 = threading.Thread(target=_icl, daemon=True)
        worker2.start()

    def run(name, fn):
        if name not in LEARNERS or time.time() > deadline - 5:
            return
        try:
            o, t = fn()
            if _ok(o) and _ok(t):
                learners[name] = (o, t)
        except Exception:
            pass

    run("knnw", lambda: knnw_learner(inst, lad_oof, lad_test,
                                     deadline=deadline, seed=SEED,
                                     q=3000 if HEAVY else 2000)[:2])
    run("knn", lambda: knn_learner(inst, lad_oof, lad_test,
                                   deadline=deadline)[:2])
    seeds = (0,) if phase == 1 else (0, 1, 2)
    if HEAVY:
        seeds = (0, 1) if phase == 1 else (0, 1, 2, 3, 4)
    run("mlp", lambda: mlp_learner(inst, hidden=(256,) if HEAVY else (128,),
                                   drop=0.3 if HEAVY else 0.2, wd=1e-5,
                                   seeds=seeds, deadline=deadline))

    def _hgb():
        o, t = hgb_learner(inst, deadline=deadline, items=order,
                           max_iter=300 if HEAVY else 200)
        o = [x if x is not None else lad_oof[i] for i, x in enumerate(o)]
        t = [x if x is not None else lad_test[i] for i, x in enumerate(t)]
        return o, t
    run("hgb", _hgb)

    if worker is not None:
        worker.join(timeout=max(1.0, deadline - time.time() + 15.0))
        if not worker.is_alive() and "res" in tab:
            o, t, done = tab["res"]
            if any(done) and _ok(o) and _ok(t):
                learners["tab"] = (o, t)
    if worker2 is not None:
        worker2.join(timeout=max(1.0, deadline - time.time() + 15.0))
        if not worker2.is_alive() and "res" in icl:
            o, t, done = icl["res"]
            if any(done) and _ok(o) and _ok(t):
                learners["icl"] = (o, t)

    names = list(learners)
    bF = bH = None
    if LOOKALIKE and len(names) > 1:
        try:
            dF, dH = localstack.nn_distance(inst)
            bF = localstack.bins_of(dF, LOOKALIKE_CUTS)
            bH = localstack.bins_of(dH, LOOKALIKE_CUTS)
        except Exception:
            bF = bH = None
    preds = []
    for t in range(len(inst.targets)):
        y = inst.Y[:, t]
        oofs = [learners[n][0][t] for n in names]
        tests = [learners[n][1][t] for n in names]
        try:
            w = fit_stack(oofs, y) if len(names) > 1 else np.ones(1)
            bo, p = normalize(mix(oofs, w)), normalize(mix(tests, w))
            if bF is not None:
                try:
                    if LOOKALIKE_GATE:
                        bo, p, _ = localstack.gated_blend(
                            oofs, tests, y, bF, bH, w, inst.folds)
                    else:
                        bo, p, _ = localstack.local_blend(
                            oofs, tests, y, bF, bH, w)
                except Exception:
                    bo, p = normalize(mix(oofs, w)), normalize(mix(tests, w))
            if POST:
                p, _ = postprocess(bo, p, y, inst.opts[t])
        except Exception:
            p = normalize(lad_test[t])
        preds.append(normalize(p))
    return _emit(inst, preds)
