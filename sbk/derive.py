"""Derived GIVEN features, appended as extra categorical columns.

UNHCR: the household roster arrives as eight banded count columns
(s2_3_<sex>_<age>). Several answers depend on combinations of them
(e.g. s6_12, children's schooling, is asked of households with any child
aged 6-17 of either sex), which a tree or a backoff ladder needs several
splits to express. We add banded totals: household size, children 0-5,
children 6-17, adults 18-59, people 60+, and whether any adult man is
present.

World Bank: the gap, in bands, between the lowest acceptable pay and the
typical wage of similar people (both GIVEN), a reservation-wage signal.

Columns that are absent leave the instrument unchanged.
"""
import re

import numpy as np
import pandas as pd


def _mid(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip()
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    m = re.match(r"^(\d+)\s*\+$", s)
    if m:
        return float(m.group(1)) + 1.0
    m = re.match(r"^(\d+)$", s)
    if m:
        return float(m.group(1))
    return np.nan


def _band(x, edges, labels):
    out = np.full(x.shape, "NA", dtype=object)
    ok = ~np.isnan(x)
    idx = np.searchsorted(np.asarray(edges, float), x[ok], side="right")
    out[ok] = np.asarray(labels, dtype=object)[idx]
    return out


def derived_columns(frame):
    """Return {name: array of string levels} for the derived columns."""
    out = {}
    roster = {c: frame[c].map(_mid).to_numpy(dtype=float)
              for c in frame.columns if str(c).startswith("s2_3_")}
    if len(roster) >= 4:
        def total(keys):
            cols = [roster[k] for k in keys if k in roster]
            if not cols:
                return None
            return np.sum(np.stack(cols, 1), axis=1)
        allc = total(list(roster))
        out["d_hh_size"] = _band(allc, [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 9.5, 12.5],
                                 ["1", "2", "3", "4", "5", "6", "7", "8-9",
                                  "10-12", "13+"])
        for name, keys, edges, labels in [
                ("d_kids_0_5", ["s2_3_boys_0_5", "s2_3_girls_0_5"],
                 [0.5, 1.5, 2.5], ["0", "1", "2", "3+"]),
                ("d_kids_6_17", ["s2_3_boys_6_17", "s2_3_girls_6_17"],
                 [0.5, 1.5, 2.5, 3.5], ["0", "1", "2", "3", "4+"]),
                ("d_adults", ["s2_3_men_18_59", "s2_3_women_18_59"],
                 [0.5, 1.5, 2.5, 3.5], ["0", "1", "2", "3", "4+"]),
                ("d_elderly", ["s2_3_men_60plus", "s2_3_women_60plus"],
                 [0.5, 1.5], ["0", "1", "2+"])]:
            v = total(keys)
            if v is not None:
                out[name] = _band(v, edges, labels)
        men = total(["s2_3_men_18_59", "s2_3_men_60plus"])
        if men is not None:
            out["d_adult_man"] = _band(men, [0.5], ["no", "yes"])
    lo, wg = "user.lowestAcceptablePay", "user.wageOfSimilarEmployees"
    if lo in frame.columns and wg in frame.columns:
        order = ["0", "1-1999", "2000-2999", "3000-3999", "4000-4999",
                 "5000-6999", "7000-9999", "10000-14999", "15000-24999",
                 "25000+"]
        pos = {v: i for i, v in enumerate(order)}
        a = frame[lo].map(pos).to_numpy(dtype=float)
        b = frame[wg].map(pos).to_numpy(dtype=float)
        out["d_pay_gap"] = _band(a - b, [-1.5, -0.5, 0.5, 1.5],
                                 ["<=-2", "-1", "0", "+1", ">=+2"])
    return out
