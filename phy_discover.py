#!/usr/bin/env python3
"""
PHYDISCOVER V4 - hypothesis discovery engine
=============================================

Finds simple, accurate, robust, non-redundant relationships hidden in
observations. It is never told which variable is the "answer" and never told
that the data is physics.

    RAW DATA (several experiments)
        -> DATA ANALYZER             positivity, ranges, which columns vary
        -> DIMENSION ANALYZER        (optional) dimensionless groups
        -> POWER-LAW / INVARIANT     log-space null-space search over variable
                                     subsets:  prod v_i^p_i = C
        -> IMPLICIT SPARSE SEARCH    transform library + beam search for
                                     sparse  F(terms) = 0  (sums, ratios, logs)
        -> ROBUST TESTING            trimmed fits, bootstrap SEs, train/val/test,
                                     per-experiment consistency
        -> CONSTANT SNAPPING         2.7498 +- 0.0003 -> 2.75 (learned) ...
        -> EQUIVALENCE + NOVELTY     KNOWN / DERIVED / REDUNDANT / NOVEL
        -> INTERESTINGNESS           accuracy, simplicity, generality,
                                     robustness, novelty
        -> ACTIVE EXPERIMENT DESIGN  query-by-committee over rival hypotheses
        -> NEW DATA -> re-run

Benchmark: `--challenge` generates data from secret laws, gives the engine only
the observations, then scores the discovery against the truth on fresh data.

Requires: numpy only.
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from dataclasses import dataclass
from functools import reduce

import numpy as np


# =============================================================================
# 1. CONFIG
# =============================================================================
@dataclass
class Config:
    seed: int = 42
    split: tuple = (0.6, 0.2, 0.2)    # train / validation / test (per experiment)
    max_subset: int = 4               # max variables in a power-law relation
    log_fit_tol: float = 0.02         # robust log-residual for a candidate power law
    accept_err: float = 0.02          # median relative error to accept a hypothesis
    max_terms: int = 3                # max terms in an implicit sparse relation
    beam: int = 30
    n_boot: int = 40
    trim_k: float = 3.5               # outlier trimming, in robust sigmas
    max_library: int = 900
    committee: int = 24
    verbose: bool = True


def say(cfg, *a):
    if cfg.verbose:
        print(*a)


# =============================================================================
# 2. NUMERIC HELPERS
# =============================================================================
def fmt(x):
    return f"{x:.6g}"


def mad_sigma(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return 0.0
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def trim_mask(r, k, floor=1e-6):
    r = np.asarray(r)
    s = max(mad_sigma(r), floor)
    return np.abs(r - np.median(r)) <= k * s


def _nrows(cols):
    return len(next(iter(cols.values())))


# =============================================================================
# 3. CONSTANT / EXPONENT SNAPPING
#    Learned constants are distinguished from grammar constants.
# =============================================================================
NAMED = [("pi", math.pi), ("2*pi", 2 * math.pi), ("4*pi^2", 4 * math.pi ** 2),
         ("pi/2", math.pi / 2), ("e", math.e), ("sqrt(2)", math.sqrt(2)),
         ("sqrt(3)", math.sqrt(3)), ("1/pi", 1 / math.pi),
         ("1/(2*pi)", 1 / (2 * math.pi))]
GRAMMAR = [0.1, 0.2, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 9.8, 9.81,
           1 / 3, 2 / 3] + [v for _, v in NAMED]


@dataclass
class ConstInfo:
    value: float
    raw: float
    se: float
    origin: str            # 'grammar' | 'learned' | 'fitted'
    label: str = ""
    inv: float | None = None

    def describe(self):
        shown = self.label or fmt(abs(self.value))
        if self.inv:
            shown = f"1/{fmt(self.inv)}"
        if self.origin == "grammar":
            return f"{shown} [grammar constant]"
        if self.origin == "learned":
            return (f"{shown} [LEARNED, snapped from "
                    f"{fmt(self.raw)} +- {self.se:.2g}]")
        return f"{shown} [fitted, +- {self.se:.2g}]"


def snap_const(value, se_rel):
    sign = -1.0 if value < 0 else 1.0
    v = abs(value)
    se_abs = se_rel * v
    tol = min(max(3 * se_rel, 1e-6), 0.05)
    best = [None]

    def consider(cost, shown, inv, label):
        eff = 1.0 / shown if inv else shown
        dev = abs(eff - v) / v
        if dev <= tol:
            cand = (cost, dev, shown, inv, label)
            if best[0] is None or cand[:2] < best[0][:2]:
                best[0] = cand

    for inv in (False, True):
        x = 1.0 / v if inv else v
        extra = 0.7 if inv else 0.0
        for d in range(1, 6):
            s = float(f"{x:.{d - 1}e}")
            if s > 0:
                consider(d + extra, s, inv, fmt(s))
        if not inv:
            for q in range(2, 13):
                p = round(x * q)
                if 1 <= p <= 60 and math.gcd(p, q) == 1:
                    consider(1 + len(str(p)) + len(str(q)), p / q, False, f"{p}/{q}")
            for name, val in NAMED:
                consider(2.0, val, False, name)
    if best[0] is None:
        return ConstInfo(sign * float(f"{v:.6g}"), value, se_abs, "fitted")
    _, _, shown, inv, label = best[0]
    eff = 1.0 / shown if inv else shown
    origin = "grammar" if any(abs(eff - g) < 1e-9 * max(1, g) for g in GRAMMAR) else "learned"
    return ConstInfo(sign * eff, value, se_abs, origin, label, shown if inv else None)


def snap_exponent(q, se):
    tol = min(max(3 * se, 1e-6), 0.08)
    cands = []
    for den, cost in ((1, 1.0), (2, 1.5), (3, 2.0), (4, 2.2)):
        v = round(q * den) / den
        cands.append((cost, abs(v - q), v))
    for d, cost in ((1, 2.5), (2, 3.0), (3, 3.5)):
        v = round(q, d)
        cands.append((cost, abs(v - q), v))
    ok = [c for c in cands if c[1] <= tol]
    if ok:
        return min(ok)[2], True
    return round(q, 3), False


def exp_complexity(e):
    a = abs(e)
    if abs(a - 1) < 1e-9:
        return 1.0
    if abs(a - round(a)) < 1e-9:
        return 1.4
    if abs(a * 2 - round(a * 2)) < 1e-9:
        return 1.8
    if abs(a * 3 - round(a * 3)) < 1e-9 or abs(a * 4 - round(a * 4)) < 1e-9:
        return 2.2
    return 3.0


def _simple_cost(q):
    for den in (1, 2, 3, 4):
        v = round(q * den) / den
        if abs(v - q) < 0.04:
            return exp_complexity(v)
    return 3.5


# =============================================================================
# 4. FEATURES AND RELATIONS
#    A relation is   lhs = sum_k coef_k * feature_k    (implicit form F = 0)
# =============================================================================
class Feature:
    __slots__ = ("kind", "exps", "var")

    def __init__(self, kind, exps=None, var=None):
        self.kind, self.exps, self.var = kind, exps or {}, var

    @staticmethod
    def mono(exps):
        return Feature("m", {v: float(e) for v, e in exps.items() if abs(e) > 1e-12})

    @staticmethod
    def log(var):
        return Feature("log", var=var)

    def key(self):
        if self.kind == "m":
            return tuple(sorted((v, round(e, 3)) for v, e in self.exps.items()))
        return ((f"log({self.var})", 1.0),)

    def vec(self):
        return dict(self.exps) if self.kind == "m" else {"log:" + self.var: 1.0}

    def vars(self):
        return set(self.exps) if self.kind == "m" else {self.var}

    def eval(self, cols):
        n = _nrows(cols)
        with np.errstate(all="ignore"):
            if self.kind == "m":
                out = np.ones(n)
                for v, e in self.exps.items():
                    out = out * (cols[v] if e == 1.0 else np.power(cols[v], e))
                return out
            return np.log(cols[self.var])

    def complexity(self):
        if self.kind == "log":
            return 3.0
        return 1.0 + sum(exp_complexity(e) for e in self.exps.values())

    def text(self):
        if self.kind == "log":
            return f"log({self.var})"
        return mono_text(self.exps)


def _pow(v, e):
    return v if abs(e - 1) < 1e-9 else f"{v}^{fmt(e)}"


def mono_text(exps):
    num = [_pow(v, e) for v, e in sorted(exps.items()) if e > 0]
    den = [_pow(v, -e) for v, e in sorted(exps.items()) if e < 0]
    n = "*".join(num) if num else "1"
    if den:
        d = "*".join(den)
        return f"{n}/({d})" if len(den) > 1 else f"{n}/{d}"
    return n


class Relation:
    def __init__(self, lhs, rhs, cinfo=None):
        self.lhs = lhs
        self.rhs = list(rhs)                       # [(coef, Feature)]
        self.cinfo = cinfo or [None] * len(self.rhs)

    def terms(self):
        return [(1.0, self.lhs)] + [(-c, f) for c, f in self.rhs]

    def variables(self):
        out = set(self.lhs.vars())
        for _, f in self.rhs:
            out |= f.vars()
        return out

    def rhs_value(self, cols):
        out = np.zeros(_nrows(cols))
        for c, f in self.rhs:
            out = out + c * f.eval(cols)
        return out

    def residual(self, cols):
        with np.errstate(all="ignore"):
            L = self.lhs.eval(cols)
            return (L - self.rhs_value(cols)) / np.abs(L)

    def is_explicit(self):
        if self.lhs.kind != "m" or len(self.lhs.exps) != 1:
            return False
        (v, e), = self.lhs.exps.items()
        return e == 1.0 and all(v not in f.vars() for _, f in self.rhs)

    def lhs_var(self):
        return next(iter(self.lhs.exps))

    def with_coefs(self, coefs):
        return Relation(self.lhs, [(c, f) for c, (_, f) in zip(coefs, self.rhs)], self.cinfo)

    def complexity(self):
        total = self.lhs.complexity()
        for (c, f), info in zip(self.rhs, self.cinfo):
            total += f.complexity()
            if abs(abs(c) - 1) < 1e-9:
                continue
            if info is None or info.origin == "learned":
                total += 1.0
            elif info.origin == "grammar":
                total += 0.5
            else:
                total += 2.0
        return total

    def _term(self, c, f, info):
        body = f.text()
        if info is not None and info.inv:
            return f"{body}/{fmt(info.inv)}"
        if abs(abs(c) - 1) < 1e-9:
            return body
        label = info.label if (info is not None and info.label) else fmt(abs(c))
        return label if body == "1" else f"{label}*{body}"

    def text(self):
        s = ""
        for i, ((c, f), info) in enumerate(zip(self.rhs, self.cinfo)):
            t = self._term(c, f, info)
            if i == 0:
                s = ("-" if c < 0 else "") + t
            else:
                s += (" - " if c < 0 else " + ") + t
        return f"{self.lhs.text()} = {s}"

    def implicit_text(self):
        s = self.lhs.text()
        for (c, f), info in zip(self.rhs, self.cinfo):
            s += (" + " if c < 0 else " - ") + self._term(c, f, info)
        return f"{s} = 0"


# ---- equivalence ------------------------------------------------------------
def _reps(terms):
    terms = [(c, f) for c, f in terms if abs(c) > 1e-12]
    allm = all(f.kind == "m" for _, f in terms)
    reps = []
    for ck, fk in terms:
        rep = []
        for c, f in terms:
            vf, vk = f.vec(), (fk.vec() if allm else {})
            e = {v: vf.get(v, 0) - vk.get(v, 0) for v in set(vf) | set(vk)}
            rep.append(({v: x for v, x in e.items() if abs(x) > 1e-9}, c / ck))
        reps.append(rep)
    return reps


def _rep_equal(a, b, te, tc):
    if len(a) != len(b):
        return False
    used = set()
    for va, ca in a:
        hit = False
        for j, (vb, cb) in enumerate(b):
            if j in used or set(va) != set(vb):
                continue
            if all(abs(va[v] - vb[v]) <= te for v in va) and \
               abs(ca - cb) <= tc * max(abs(ca), abs(cb)):
                used.add(j)
                hit = True
                break
        if not hit:
            return False
    return True


def _mono_law(terms):
    """two-term monomial relation c1*m1 + c2*m2 = 0  ->  (exponent vector, log C)."""
    if len(terms) != 2 or any(f.kind != "m" for _, f in terms):
        return None
    (c1, f1), (c2, f2) = terms
    if c1 * c2 >= 0:
        return None
    p = {v: f1.exps.get(v, 0) - f2.exps.get(v, 0) for v in set(f1.exps) | set(f2.exps)}
    return p, math.log(-c2 / c1)


def terms_match(a, b, te=0.03, tc=0.015):
    """Same underlying relation? Invariant to monomial multiplication,
    rescaling, which term is moved to the left, and (for pure power laws)
    raising both sides to a power:  A=BC, A/B=C, A/(BC)=1, A-BC=0 all match;
    so do  x = 10.24*Y^2/z^4  and  Y = sqrt(x)*z^2/3.2."""
    ma, mb = _mono_law(a), _mono_law(b)
    if ma and mb:
        keys = sorted(set(ma[0]) | set(mb[0]))
        u = np.array([ma[0].get(k, 0) for k in keys])
        w = np.array([mb[0].get(k, 0) for k in keys])
        if u @ u > 0 and w @ w > 0:
            s = (u @ w) / (w @ w)
            if np.linalg.norm(u - s * w) < 0.03 * np.linalg.norm(u) and \
               abs(ma[1] - s * mb[1]) < 0.02 + 0.01 * abs(ma[1]):
                return True
    ra = _reps(a)
    if not ra:
        return False
    return any(_rep_equal(ra[0], rb, te, tc) for rb in _reps(b))


# =============================================================================
# 5. DATA
# =============================================================================
class Experiment:
    def __init__(self, name, cols):
        self.name = name
        self.cols = {k: np.asarray(v, float) for k, v in cols.items()}
        self.n = len(next(iter(self.cols.values())))


class Stack:
    def __init__(self, cols, exp_id, n_exp):
        self.cols, self.exp_id, self.n_exp = cols, exp_id, n_exp
        self.n = len(exp_id)

    def take(self, idx):
        return Stack({k: v[idx] for k, v in self.cols.items()}, self.exp_id[idx], self.n_exp)


def stack_from_cols(cols):
    n = _nrows(cols)
    return Stack({k: np.asarray(v, float) for k, v in cols.items()}, np.zeros(n, int), 1)


class DataBundle:
    def __init__(self, experiments, cfg, dims=None, base_dims=None, outputs=None):
        self.experiments, self.cfg = experiments, cfg
        self.variables = list(experiments[0].cols.keys())
        self.dims, self.base_dims = dims, base_dims
        self.outputs = list(outputs or [])
        rng = np.random.default_rng(cfg.seed)
        parts = {"train": [], "val": [], "test": []}
        for e, ex in enumerate(experiments):
            perm = rng.permutation(ex.n)
            a = int(cfg.split[0] * ex.n)
            b = a + int(cfg.split[1] * ex.n)
            for nm, idx in zip(("train", "val", "test"), (perm[:a], perm[a:b], perm[b:])):
                parts[nm].append((e, ex, idx))
        self.train, self.val, self.test = (self._stack(parts[k]) for k in ("train", "val", "test"))
        self.full = self._stack([(e, ex, np.arange(ex.n)) for e, ex in enumerate(experiments)])

    def _stack(self, parts):
        cols = {v: np.concatenate([ex.cols[v][idx] for _, ex, idx in parts]) for v in self.variables}
        eid = np.concatenate([np.full(len(idx), e) for e, _, idx in parts])
        return Stack(cols, eid, len(self.experiments))


def load_csv(paths, cfg):
    exps = []
    for p in paths:
        arr = np.genfromtxt(p, delimiter=",", names=True)
        exps.append(Experiment(p, {n: arr[n] for n in arr.dtype.names}))
    return DataBundle(exps, cfg)


def analyze(bundle, cfg):
    info = {}
    say(cfg, "\n[DATA ANALYZER]")
    say(cfg, f"  {'var':8s} {'min':>11s} {'max':>11s} {'decades':>8s}  flags")
    for v in bundle.variables:
        a = bundle.full.cols[v]
        pos = bool(np.all(a > 0))
        varying = False
        for e in range(bundle.full.n_exp):
            s = a[bundle.full.exp_id == e]
            varying |= (np.std(np.log(s)) > 1e-3) if pos else (np.std(s) > 1e-9)
        span = math.log10(a.max() / a.min()) if pos else float("nan")
        info[v] = dict(positive=pos, varying=bool(varying))
        flags = ("log-able " if pos else "has<=0 ") + ("" if varying else "CONSTANT")
        say(cfg, f"  {v:8s} {a.min():11.4g} {a.max():11.4g} {span:8.2f}  {flags}")
    return info


# =============================================================================
# 6. ERROR, EVALUATION, ENTRIES
# =============================================================================
def median_err(rel, stack):
    with np.errstate(all="ignore"):
        r = np.abs(rel.residual(stack.cols))
    fin = np.isfinite(r)
    if fin.mean() < 0.9:
        return float("inf")
    return float(np.median(r[fin]))


def evaluate_relation(rel, bundle):
    st = {k: median_err(rel, getattr(bundle, k)) for k in ("train", "val", "test")}
    full = bundle.full
    ex = [median_err(rel, full.take(np.where(full.exp_id == e)[0])) for e in range(full.n_exp)]
    st["per_exp"], st["max_exp"] = ex, max(ex)
    return st


class Entry:
    def __init__(self, rel, source, stats, spread, params, committee):
        self.rel, self.source, self.stats = rel, source, stats
        self.spread, self.params, self.committee = spread, params, committee
        self.cls, self.aliases, self.scores, self.dim = None, [], {}, "n/a"


def accept(stats, cfg):
    a = cfg.accept_err
    return stats["train"] <= a and stats["val"] <= a and stats["max_exp"] <= a


# =============================================================================
# 7. DIMENSION ANALYZER
# =============================================================================
def dim_status(rel, bundle):
    dims = bundle.dims
    if not dims or not rel.variables() <= set(dims):
        return "n/a"
    nb = len(next(iter(dims.values())))

    def dv(f):
        vec = np.zeros(nb)
        if f.kind == "m":
            for v, e in f.exps.items():
                vec += e * np.array(dims[v], float)
        elif np.any(np.array(dims[f.var]) != 0):
            return None
        return vec

    dl = dv(rel.lhs)
    dr = [dv(f) for _, f in rel.rhs]
    if dl is None or any(d is None for d in dr):
        return "INCONSISTENT (log of a dimensional quantity)"
    if all(np.allclose(d, dl) for d in dr):
        return "consistent"
    if len(rel.rhs) == 1:
        u = dl - dr[0]
        names = bundle.base_dims or [f"d{i}" for i in range(nb)]
        txt = " ".join(f"{n}^{fmt(x)}" for n, x in zip(names, u) if abs(x) > 1e-9)
        return f"constant carries units [{txt}]"
    return "INCONSISTENT (terms differ in dimension)"


# =============================================================================
# 8. POWER-LAW / INVARIANT FINDER   (prod v^p = C   <=>   log-linear null space)
# =============================================================================
class PowerLawFinder:
    def __init__(self, bundle, cfg, loggable):
        self.b, self.cfg = bundle, cfg
        tr = bundle.train
        self.vars = [v for v in bundle.variables if v in loggable]
        self.logs = {v: np.log(tr.cols[v]) for v in self.vars}
        self.cent = {}
        for v in self.vars:
            c = self.logs[v].copy()
            for e in range(tr.n_exp):
                m = tr.exp_id == e
                if m.any():
                    c[m] -= c[m].mean()
            self.cent[v] = c
        self.rng = np.random.default_rng(cfg.seed + 1)
        self.hints = []
        self.order = {v: i for i, v in enumerate(bundle.variables)}

    @staticmethod
    def _null(X):
        _, s, vt = np.linalg.svd(X, full_matrices=False)
        return vt[-1], s

    def fit_subset(self, sub):
        X = np.column_stack([self.cent[v] for v in sub])
        mask = np.ones(len(X), bool)
        for _ in range(4):
            p, _ = self._null(X[mask])
            p = p / np.abs(p).max()
            keep = trim_mask(X @ p, self.cfg.trim_k)
            if keep.sum() < 0.8 * len(X):
                return None
            if (keep == mask).all():
                break
            mask = keep
        p, s = self._null(X[mask])
        p = p / np.abs(p).max()
        sig = mad_sigma(X[mask] @ p)
        su = s / math.sqrt(mask.sum())
        if sig > self.cfg.log_fit_tol or np.abs(p).min() < 0.04:
            return None
        if len(su) >= 2 and su[-2] < max(5 * su[-1], 0.05):
            return None            # null space not 1-D: smaller subsets cover it
        return dict(sub=sub, p=p, mask=mask, X=X)

    def pick_lhs(self, sub, p):
        best = None
        for i in range(len(sub)):
            cost = sum(_simple_cost(-p[j] / p[i]) for j in range(len(sub)) if j != i)
            if sub[i] in self.b.outputs:
                cost -= 2.0
            key = (round(cost, 6), -self.order[sub[i]])
            if best is None or key < best[0]:
                best = (key, i)
        return best[1]

    def fit_const(self, lhs, q, mask):
        tr = self.b.train
        y = self.logs[lhs] - sum(q[v] * self.logs[v] for v in q)
        c = float(np.median(y[mask]))
        res = y - c
        n = int(mask.sum())
        se = 1.2533 * max(mad_sigma(res[mask]), 1e-9) / math.sqrt(n)
        per = []
        for e in range(tr.n_exp):
            m = mask & (tr.exp_id == e)
            if m.sum() >= 5:
                per.append((float(np.median(y[m])),
                            1.2533 * max(mad_sigma(res[m]), 1e-9) / math.sqrt(m.sum())))
        return c, se, per

    def refine(self, cand):
        sub, p, mask, X = cand["sub"], cand["p"], cand["mask"], cand["X"]
        tr = self.b.train
        i = self.pick_lhs(sub, p)
        lhs = sub[i]
        rest = [v for j, v in enumerate(sub) if j != i]
        qf = {sub[j]: -p[j] / p[i] for j in range(len(sub)) if j != i}
        idx = np.where(mask)[0]
        QB = []
        for _ in range(self.cfg.n_boot):
            rows = self.rng.choice(idx, size=len(idx))
            pb, _ = self._null(X[rows])
            if pb @ p < 0:
                pb = -pb
            if abs(pb[i]) < 1e-6:
                continue
            QB.append([-pb[j] / pb[i] for j in range(len(sub)) if j != i])
        if len(QB) < 5:
            return None
        QB = np.array(QB)
        se = {v: float(QB[:, k].std()) for k, v in enumerate(rest)}

        lhs_f = Feature.mono({lhs: 1})
        c_f, _, _ = self.fit_const(lhs, qf, mask)
        err_free = median_err(Relation(lhs_f, [(math.exp(c_f), Feature.mono(qf))]), tr)
        snap_ok = {v: snap_exponent(qf[v], se[v]) for v in rest}
        qs = {v: snap_ok[v][0] for v in rest}
        c_s, se_s, per_s = self.fit_const(lhs, qs, mask)
        err_snap = median_err(Relation(lhs_f, [(math.exp(c_s), Feature.mono(qs))]), tr)
        if err_snap <= err_free * 1.25 + 3e-4:
            q, c, se_c, per = qs, c_s, se_s, per_s
        else:
            q = {v: round(qf[v], 3) for v in rest}
            c, se_c, per = self.fit_const(lhs, q, mask)
        if len(per) > 1:
            spread = max(abs(ce - c) for ce, _ in per)
            if spread > 5 * max(s for _, s in per) + 1e-4:
                self.hints.append((lhs, q, [round(math.exp(ce), 4) for ce, _ in per]))
                return None
        info = snap_const(math.exp(c), se_c)
        rel = Relation(lhs_f, [(info.value, Feature.mono(q))], [info])

        committee = []
        for row in QB[: self.cfg.committee]:
            qb = dict(zip(rest, row))
            cb, _, _ = self.fit_const(lhs, qb, mask)
            committee.append(Relation(lhs_f, [(math.exp(cb), Feature.mono(qb))]))
        used_snap = q is qs
        exp_info = [f"exponent {v}: {fmt(q[v])} "
                    f"[{'snapped' if used_snap and snap_ok[v][1] else 'fitted'}"
                    f", raw {qf[v]:.4f} +- {se[v]:.2g}]" for v in rest]
        spread = max(max(se.values()), se_c)
        return rel, spread, exp_info, committee

    def dimensionless(self, found_subs):
        dims = self.b.dims
        out = []
        if not dims:
            return out
        vs = [v for v in self.vars if v in dims]
        for size in range(2, self.cfg.max_subset + 1):
            for sub in itertools.combinations(vs, size):
                if sub in found_subs:
                    continue
                D = np.array([dims[v] for v in sub], float).T
                if not np.any(D):
                    continue
                P = np.array(list(itertools.product([-3, -2, -1, 1, 2, 3], repeat=size)))
                P = P[np.abs(P @ D.T).max(axis=1) < 1e-9]
                X = np.column_stack([self.cent[v] for v in sub])
                for p in P:
                    if p[0] < 0 or reduce(math.gcd, [int(abs(x)) for x in p]) != 1:
                        continue
                    pf = p / np.abs(p).max()
                    r = X @ pf
                    mask = trim_mask(r, self.cfg.trim_k)
                    if mask.mean() > 0.8 and mad_sigma(r[mask]) <= self.cfg.log_fit_tol:
                        out.append(dict(sub=sub, p=pf, mask=mask, X=X))
        return out

    def run(self):
        entries, found = [], set()
        cands = []
        for size in range(2, self.cfg.max_subset + 1):
            for sub in itertools.combinations(self.vars, size):
                c = self.fit_subset(sub)
                if c:
                    cands.append(c)
                    found.add(sub)
        cands += self.dimensionless(found)
        for c in cands:
            res = self.refine(c)
            if res is None:
                continue
            rel, spread, exp_info, committee = res
            st = evaluate_relation(rel, self.b)
            if accept(st, self.cfg):
                kind = "power-law"
                entries.append(Entry(rel, kind, st, spread, exp_info, committee))
        return entries


# =============================================================================
# 9. IMPLICIT SPARSE FINDER  (transform library + beam search)
# =============================================================================
def beam_search(Phi, t, kmax, beam):
    G = Phi.T @ Phi
    b = Phi.T @ t
    tt = float(t @ t)
    diag = np.diag(G).copy()
    L = Phi.shape[1]
    rss1 = tt - b ** 2 / np.maximum(diag, 1e-30)
    order = np.argsort(rss1)[:beam]
    cur = [(float(rss1[j]), (int(j),)) for j in order]
    models = {1: cur[0]}
    for k in range(2, kmax + 1):
        cand = {}
        for rss_s, S in cur:
            idx = list(S)
            try:
                inv = np.linalg.inv(G[np.ix_(idx, idx)] + 1e-12 * np.eye(len(idx)))
            except np.linalg.LinAlgError:
                continue
            GjS = G[:, idx]
            rb = b - GjS @ (inv @ b[idx])
            d = diag - np.einsum("ij,jk,ik->i", GjS, inv, GjS)
            new = rss_s - rb ** 2 / np.maximum(d, 1e-12)
            new[idx] = np.inf
            new[d < 1e-8] = np.inf
            for j in np.argsort(new)[:beam]:
                if not np.isfinite(new[j]):
                    continue
                T = tuple(sorted(S + (int(j),)))
                r = float(new[j])
                if T not in cand or r < cand[T]:
                    cand[T] = r
        if not cand:
            break
        cur = sorted(((r, T) for T, r in cand.items()))[:beam]
        models[k] = cur[0]
    return models


class ImplicitFinder:
    def __init__(self, bundle, cfg, basis, err_cap=None):
        self.b, self.cfg, self.basis = bundle, cfg, basis
        self.err_cap = err_cap or cfg.accept_err
        self.rng = np.random.default_rng(cfg.seed + 2)
        tr = bundle.train
        feats, cols = [], []
        for f in self._library():
            col = f.eval(tr.cols)
            if np.all(np.isfinite(col)):
                feats.append(f)
                cols.append(col)
        self.feats, self.F = feats, np.column_stack(cols)

    def _library(self):
        B, cfg = self.basis, self.cfg
        S1 = (-2, -1, -0.5, 0.5, 1, 2, 3)

        def build(P, triples):
            fs = [Feature.mono({})]
            for v in B:
                fs += [Feature.mono({v: e}) for e in S1] + [Feature.log(v)]
            for a, b in itertools.combinations(B, 2):
                for ea in P:
                    for eb in P:
                        fs.append(Feature.mono({a: ea, b: eb}))
            if triples:
                for a, b, c in itertools.combinations(B, 3):
                    for ea, eb, ec in itertools.product((-1, 1), repeat=3):
                        fs.append(Feature.mono({a: ea, b: eb, c: ec}))
            return fs

        for P, tri in [((-2, -1, -0.5, 0.5, 1, 2), True), ((-2, -1, 1, 2), True),
                       ((-2, -1, 1, 2), False), ((-1, 1), False)]:
            fs = build(P, tri)
            if len(fs) <= cfg.max_library:
                break
        return fs

    def fit_lhs(self, lhs):
        cfg, tr = self.cfg, self.b.train
        L = lhs.eval(tr.cols)
        ok = np.isfinite(L) & (np.abs(L) > 1e-12)
        if ok.sum() < 50:
            return None
        cols = [j for j, f in enumerate(self.feats) if not (f.kind == "m" and f.key() == lhs.key())]
        w = 1.0 / np.abs(L[ok])
        t = np.sign(L[ok])
        Phi = self.F[ok][:, cols] * w[:, None]
        nrm = np.linalg.norm(Phi, axis=0)
        nrm[nrm < 1e-12] = 1e12

        def normed(A):
            n = np.linalg.norm(A, axis=0)
            n[n < 1e-12] = 1e12
            return A / n

        K = cfg.max_terms
        m1 = beam_search(Phi / nrm, t, K, cfg.beam)
        S = list(m1[max(m1)][1])
        c = np.linalg.lstsq((Phi / nrm)[:, S], t, rcond=None)[0]
        inl = trim_mask(t - (Phi / nrm)[:, S] @ c, cfg.trim_k)
        n2 = int(inl.sum())
        m2 = beam_search(normed(Phi[inl]), t[inl], K, cfg.beam)
        nL = Phi.shape[1]
        bic = {k: n2 * math.log(max(r, 0) / n2 + 1e-12) + k * (math.log(n2) + 2 * math.log(nL))
               for k, (r, _) in m2.items()}
        S = list(m2[min(bic, key=bic.get)][1])

        def robust(S):
            mask = np.ones(len(t), bool)
            for _ in range(5):
                A = Phi[mask][:, S]
                sc = np.linalg.norm(A, axis=0)
                c = np.linalg.lstsq(A / sc, t[mask], rcond=None)[0] / sc
                r = t - Phi[:, S] @ c
                new = trim_mask(r, cfg.trim_k)
                if (new == mask).all():
                    break
                mask = new
            return c, mask

        def boot(S, mask):
            idx = np.where(mask)[0]
            C = []
            for _ in range(cfg.n_boot):
                rows = self.rng.choice(idx, size=len(idx))
                A = Phi[rows][:, S]
                sc = np.linalg.norm(A, axis=0)
                C.append(np.linalg.lstsq(A / sc, t[rows], rcond=None)[0] / sc)
            C = np.array(C)
            return C.std(0), C

        while True:
            c, mask = robust(S)
            se, C = boot(S, mask)
            tstat = np.abs(c) / np.maximum(se, 1e-15)
            if tstat.min() < 6:
                if len(S) == 1:
                    return None
                S.pop(int(np.argmin(tstat)))
                continue
            break

        fs = [self.feats[cols[j]] for j in S]
        free = Relation(lhs, list(zip(c, fs)))
        err_free = median_err(free, tr)
        coefs, infos = list(c), [None] * len(c)
        for j in np.argsort(-np.abs(c)):
            info = snap_const(c[j], se[j] / abs(c[j]))
            trial = list(coefs)
            trial[j] = info.value
            if median_err(Relation(lhs, list(zip(trial, fs))), tr) <= err_free * 1.3 + 3e-4:
                coefs, infos[j] = trial, info
            else:
                infos[j] = ConstInfo(c[j], c[j], se[j], "fitted")
        rel = Relation(lhs, list(zip(coefs, fs)), infos)
        committee = [Relation(lhs, list(zip(row, fs))) for row in C[: cfg.committee]]
        params = [f"coef of {f.text()}: {i.describe()}" for f, i in zip(fs, infos)]
        return rel, float(np.max(se / np.abs(c))), params, committee

    def run(self):
        out = []
        lhs_list = [Feature.mono({})] + [Feature.mono({v: 1}) for v in self.basis]
        for lhs in lhs_list:
            res = self.fit_lhs(lhs)
            if res is None:
                continue
            rel, spread, params, committee = res
            st = evaluate_relation(rel, self.b)
            capped = Config(**{**self.cfg.__dict__, "accept_err": self.err_cap})
            if accept(st, capped):
                out.append(Entry(rel, "implicit", st, spread, params, committee))
        return out


# =============================================================================
# 10. ARCHIVE: KNOWN / DERIVED / REDUNDANT / NOVEL
# =============================================================================
class Archive:
    def __init__(self, known, variables):
        self.vars, self.known = variables, list(known)
        self.entries, self.all = [], []
        self.mono = []

    @staticmethod
    def _mono_vec(rel):
        if len(rel.rhs) != 1 or rel.lhs.kind != "m" or rel.rhs[0][1].kind != "m":
            return None
        c, f = rel.rhs[0]
        if c <= 0:
            return None
        p = dict(rel.lhs.exps)
        for v, e in f.exps.items():
            p[v] = p.get(v, 0) - e
        return p, math.log(c)

    def _mono_class(self, p, logc):
        if not self.mono:
            return "NOVEL", None
        vp = np.array([p.get(v, 0) for v in self.vars])
        M = np.array([[q.get(v, 0) for v in self.vars] for q, _ in self.mono])
        for k, (q, lc) in enumerate(self.mono):
            u = M[k]
            s = (vp @ u) / (u @ u)
            if np.linalg.norm(vp - s * u) < 0.03 * np.linalg.norm(vp) and abs(logc - s * lc) < 0.03:
                return "REDUNDANT", k
        coef, *_ = np.linalg.lstsq(M.T, vp, rcond=None)
        if np.linalg.norm(M.T @ coef - vp) < 0.03 * np.linalg.norm(vp):
            lc = sum(c * l for c, (_, l) in zip(coef, self.mono))
            if abs(logc - lc) < 0.05:
                return "DERIVED", None
        return "NOVEL", None

    def classify(self, entry):
        rel = entry.rel
        for old in self.entries:
            if terms_match(rel.terms(), old.rel.terms()):
                old.aliases.append(rel.text())
                entry.cls = "REDUNDANT"
                self.all.append(entry)
                return
        cls = None
        for k in self.known:
            if terms_match(rel.terms(), k.terms()):
                cls = "KNOWN"
        mv = self._mono_vec(rel)
        if cls is None:
            cls = "NOVEL"
            if mv is not None:
                cls, par = self._mono_class(*mv)
                if cls == "REDUNDANT":
                    self.entries[0].aliases.append(rel.text())
                    entry.cls = cls
                    self.all.append(entry)
                    return
        entry.cls = cls
        self.entries.append(entry)
        self.all.append(entry)
        if mv is not None and cls in ("NOVEL", "KNOWN"):
            self.mono.append(mv)

    def ranked(self):
        return sorted(self.entries, key=lambda e: -e.scores.get("interestingness", 0))


# =============================================================================
# 11. INTERESTINGNESS
# =============================================================================
def score_entry(e, bundle):
    st = e.stats
    acc = 1 / (1 + (st["test"] / 0.01) ** 2)
    simp = math.exp(-e.rel.complexity() / 16)
    gap = 1 / (1 + (max(0, st["test"] - st["train"]) / 0.01) ** 2)
    gen = (1 / (1 + (st["max_exp"] / 0.02) ** 2)) * gap * (1.0 if bundle.full.n_exp > 1 else 0.75)
    rob = 1 / (1 + (e.spread / 0.01) ** 2)
    nov = {"NOVEL": 1.0, "DERIVED": 0.4, "KNOWN": 0.15, "REDUNDANT": 0.0}[e.cls]
    w = dict(acc=0.30, simp=0.20, gen=0.20, rob=0.15, nov=0.15)
    inter = float(np.prod([max(v, 1e-6) ** w[k] for k, v in
                           dict(acc=acc, simp=simp, gen=gen, rob=rob, nov=nov).items()]))
    e.scores = dict(accuracy=acc, simplicity=simp, generality=gen, robustness=rob,
                    novelty=nov, interestingness=inter)


# =============================================================================
# 12. DISCOVERER (pipeline)
# =============================================================================
class Discoverer:
    def __init__(self, bundle, cfg, known=(), name=""):
        self.b, self.cfg, self.known, self.name = bundle, cfg, known, name
        self.archive = Archive(known, bundle.variables)
        self.hints = []

    def run(self):
        b, cfg = self.b, self.cfg
        t0 = time.time()
        info = analyze(b, cfg)
        loggable = {v for v, i in info.items() if i["positive"] and i["varying"]}
        varying = [v for v, i in info.items() if i["varying"]]

        say(cfg, "\n[POWER LAWS / INVARIANTS] log-space null-space search"
                 + (" + dimensionless groups" if b.dims else ""))
        pl = PowerLawFinder(b, cfg, loggable)
        pool = pl.run()
        self.hints = pl.hints
        pool.sort(key=lambda e: (round(e.rel.complexity(), 1), e.stats["train"]))
        say(cfg, f"  {len(pool)} accepted candidates")
        for e in pool:
            self.archive.classify(e)

        eliminated = set()
        for e in self.archive.entries:
            if e.rel.is_explicit():
                eliminated.add(e.rel.lhs_var())
        basis = [v for v in varying if v not in eliminated]
        say(cfg, f"\n[BASIS REDUCTION] variables explained by laws: "
                 f"{sorted(eliminated) or 'none'}; independent basis: {basis}")

        # noise floor: what the cleanest accepted laws achieve. A sum-of-terms
        # relation that is far worse than that is an approximation, not a law.
        floor = min([e.stats["train"] for e in self.archive.entries], default=None)
        cap = min(cfg.accept_err, max(3 * floor, 0.004)) if floor is not None else None
        imp = ImplicitFinder(b, cfg, basis, cap)
        say(cfg, f"\n[IMPLICIT SEARCH] transform library: {len(imp.feats)} features "
                 f"(powers, ratios, products, logs) over {basis}; beam search, "
                 f"<= {cfg.max_terms} terms")
        pool2 = imp.run()
        pool2.sort(key=lambda e: (round(e.rel.complexity(), 1), e.stats["train"]))
        say(cfg, f"  {len(pool2)} accepted candidates")
        for e in pool2:
            self.archive.classify(e)

        for e in self.archive.all:
            e.dim = dim_status(e.rel, b)
            score_entry(e, b)
        say(cfg, f"\n[done in {time.time() - t0:.1f}s]")
        return self.archive

    def report(self, top=20):
        A = self.archive
        counts = {c: sum(1 for e in A.all if e.cls == c)
                  for c in ("NOVEL", "DERIVED", "KNOWN", "REDUNDANT")}
        print("\n" + "=" * 76)
        print(f"DISCOVERY ARCHIVE  {self.name}")
        print("  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
        print("=" * 76)
        for i, e in enumerate(A.ranked()[:top], 1):
            s, st = e.scores, e.stats
            print(f"\n[{i}] {e.cls:8s} interestingness {s['interestingness']:.2f}   ({e.source})")
            print(f"    {e.rel.text()}")
            print(f"    implicit form: {e.rel.implicit_text()}")
            for line in e.params:
                print(f"      - {line}")
            for inf in e.rel.cinfo:
                if inf is not None and e.source == "power-law":
                    print(f"      - constant {inf.describe()}")
            print(f"    accuracy {s['accuracy']:.2f}  simplicity {s['simplicity']:.2f}  "
                  f"generality {s['generality']:.2f}  novelty {s['novelty']:.2f}  "
                  f"robustness {s['robustness']:.2f}")
            print(f"    median rel. error  train {st['train']:.4f} | val {st['val']:.4f} | "
                  f"test {st['test']:.4f} | per-experiment "
                  f"{[round(x, 4) for x in st['per_exp']]}")
            print(f"    dimensions: {e.dim}")
            if e.aliases:
                print(f"    equivalent forms collapsed: "
                      f"{len(e.aliases)} e.g. {e.aliases[0]}")
        red = [e for e in A.all if e.cls == "REDUNDANT"]
        if red:
            print(f"\n(collapsed {len(red)} algebraically equivalent restatements)")
        if self.hints:
            print("\nHINTS - relations whose constant differs between experiments "
                  "(a hidden variable?):")
            for lhs, q, consts in self.hints[:5]:
                print(f"  {lhs} ~ C_exp * {mono_text(q)}; C per experiment = {consts}")


# =============================================================================
# 13. ACTIVE EXPERIMENT DESIGN
# =============================================================================
class ActiveDesigner:
    def __init__(self, bundle, cfg, controllable):
        self.b, self.cfg, self.ctrl = bundle, cfg, list(controllable)
        self.rng = np.random.default_rng(cfg.seed + 7)

    def _rivals(self, e):
        rel, tr = e.rel, self.b.train
        if len(rel.rhs) != 1 or rel.rhs[0][1].kind != "m" or not rel.is_explicit():
            return []
        f = rel.rhs[0][1]
        tgt = rel.lhs_var()
        out = []
        for step in (0.25, 0.5, 1.0):
            q = {v: round(x / step) * step for v, x in f.exps.items()}
            if all(abs(q[v] - f.exps[v]) < 1e-9 for v in q) or any(abs(x) < 1e-9 for x in q.values()):
                continue
            y = np.log(tr.cols[tgt]) - sum(x * np.log(tr.cols[v]) for v, x in q.items())
            r = Relation(rel.lhs, [(math.exp(np.median(y)), Feature.mono(q))])
            err = median_err(r, tr)
            out.append((r, err))
        return out

    def propose(self, archive, k=6, replicates=10):
        cands = []
        for e in archive.ranked():
            r = e.rel
            if e.cls in ("NOVEL", "KNOWN") and r.is_explicit():
                tgt = r.lhs_var()
                used = r.variables() - {tgt}
                if tgt not in self.ctrl and used <= set(self.ctrl):
                    cands.append(e)
        if not cands:
            print("\n[ACTIVE] no explicit hypothesis over controllable variables; skipping")
            return None
        e = cands[0]
        tgt = e.rel.lhs_var()
        used = sorted(e.rel.variables() - {tgt})
        members = [e.rel] + list(e.committee)
        base_err = e.stats["train"]
        alive = []
        print(f"\n[ACTIVE EXPERIMENT DESIGN] testing  {e.rel.text()}")
        seen = set()
        for r, err in self._rivals(e):
            if r.text() in seen:
                continue
            seen.add(r.text())
            keep = err <= base_err * 1.5 + 0.002
            print(f"   rival {r.text():40s} fit error {err:.4f} -> "
                  f"{'STILL COMPETITIVE' if keep else 'already rejected by existing data'}")
            if keep:
                alive.append(r)
        members += alive
        full = self.b.full
        N = 4000
        pts = {}
        for v in self.ctrl:
            lo, hi = full.cols[v].min() / 3, full.cols[v].max() * 3
            pts[v] = np.exp(self.rng.uniform(math.log(lo), math.log(hi), N))
        with np.errstate(all="ignore"):
            preds = np.array([m.rhs_value(pts) for m in members])
        preds = np.log(np.abs(preds) + 1e-300)
        ok = np.all(np.isfinite(preds), axis=0)
        spread = np.where(ok, preds.max(0) - preds.min(0), -1)
        sigma = max(1.4826 * base_err, 1e-3)
        sep = spread / sigma
        Z = np.column_stack([np.log(pts[v]) for v in used]) if used else np.zeros((N, 1))
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
        chosen = [int(np.argmax(sep))]
        while len(chosen) < k:
            d = np.min([np.linalg.norm(Z - Z[c], axis=1) for c in chosen], axis=0)
            score = sep * (1 - np.exp(-(d ** 2) / (2 * 0.8 ** 2)))
            chosen.append(int(np.argmax(score)))
        print(f"  committee = {len(members)} competing parameter sets "
              f"(bootstrap + surviving rivals); noise ~ {sigma:.4f}")
        print(f"  most informative new measurements "
              f"(separation = hypothesis disagreement in noise units):")
        for c in chosen:
            print("    " + "  ".join(f"{v}={pts[v][c]:.4g}" for v in self.ctrl)
                  + f"   -> separation {sep[c]:.1f} sigma")
        sel = {v: np.repeat(pts[v][chosen], replicates) for v in self.ctrl}
        return sel, e


# =============================================================================
# 14. HIDDEN-LAW CHALLENGE
# =============================================================================
def T(coef, **exps):
    return (coef, Feature.mono(exps))


class HiddenLaw:
    def __init__(self, name, desc, inputs, outputs, truths, dims=None, base_dims=None,
                 target_outputs=None):
        self.name, self.desc, self.inputs, self.outputs = name, desc, inputs, outputs
        self.truths = truths
        self.dims, self.base_dims = dims, base_dims
        self.target_outputs = target_outputs or list(outputs)

    def lab(self, points, rng, noise=0.003, outlier=0.01):
        cols = dict(points)
        for nm, fn in self.outputs.items():
            y = fn(points) * (1 + noise * rng.standard_normal(_nrows(points)))
            bad = rng.random(len(y)) < outlier
            y = np.where(bad, y * rng.uniform(1.3, 2.0, len(y)), y)
            cols[nm] = y
        return cols

    def sample_inputs(self, rng, n, shift=1.0, stretch=1.0):
        pts = {}
        for v, (lo, hi) in self.inputs.items():
            lo2, hi2 = lo * shift / stretch, hi * shift * stretch
            pts[v] = np.exp(rng.uniform(math.log(lo2), math.log(hi2), n))
        return pts

    def experiments(self, rng, n_exp=3, n=250):
        shifts = [1.0, 3.0, 0.4, 1.7][:n_exp]
        return [Experiment(f"exp{i + 1}", self.lab(self.sample_inputs(rng, n, s), rng))
                for i, s in enumerate(shifts)]


LAWS = {
    "law1": HiddenLaw("law1", "Y = 4.7 * x^1.83 * z",
                      {"x": (0.5, 20), "y": (0.5, 15), "z": (0.5, 10)},
                      {"Y": lambda p: 4.7 * p["x"] ** 1.83 * p["z"]},
                      [("Y = 4.7*x^1.83*z", [T(1, Y=1), T(-4.7, x=1.83, z=1)])]),
    "law2": HiddenLaw("law2", "Y = sqrt(x) * z^2 / 3.2",
                      {"x": (0.5, 20), "y": (0.5, 15), "z": (0.5, 10)},
                      {"Y": lambda p: np.sqrt(p["x"]) * p["z"] ** 2 / 3.2},
                      [("Y = sqrt(x)*z^2/3.2", [T(1, Y=1), T(-1 / 3.2, x=0.5, z=2)])]),
    "law3": HiddenLaw("law3", "Y = (x*y)/(z+2)",
                      {"x": (0.5, 20), "y": (0.5, 15), "z": (0.3, 10)},
                      {"Y": lambda p: p["x"] * p["y"] / (p["z"] + 2)},
                      [("Y*(z+2) = x*y", [T(1, Y=1, z=1), T(2, Y=1), T(-1, x=1, y=1)])]),
    "law4": HiddenLaw("law4", "Y = 7.3 * x^1.73  (scaling law, y and z irrelevant)",
                      {"x": (0.5, 20), "y": (0.5, 15), "z": (0.5, 10)},
                      {"Y": lambda p: 7.3 * p["x"] ** 1.73},
                      [("Y = 7.3*x^1.73", [T(1, Y=1), T(-7.3, x=1.73)])]),
    "suite": HiddenLaw("suite", "five simultaneous hidden laws (the V3 data)",
                       {"x": (0.5, 20), "y": (0.5, 15), "z": (0.5, 10)},
                       {"A": lambda p: 2.75 * p["x"] * p["y"],
                        "B": lambda p: 0.5 * p["x"] * p["y"] ** 2,
                        "C": lambda p: 3.0 * np.sqrt(p["x"]) * p["z"],
                        "D": lambda p: 4.2 * p["x"] ** 2 / p["y"],
                        "E": lambda p: 1.7 * p["x"] * p["y"] + 2.0 * p["z"]},
                       [("A = 2.75*x*y", [T(1, A=1), T(-2.75, x=1, y=1)]),
                        ("B = 0.5*x*y^2", [T(1, B=1), T(-0.5, x=1, y=2)]),
                        ("C = 3*sqrt(x)*z", [T(1, C=1), T(-3, x=0.5, z=1)]),
                        ("D = 4.2*x^2/y", [T(1, D=1), T(-4.2, x=2, y=-1)]),
                        ("E = 1.7*x*y + 2*z", [T(1, E=1), T(-1.7, x=1, y=1), T(-2, z=1)])]),
    "pendulum": HiddenLaw("pendulum", "T = 2*pi*sqrt(L/g)  (physics = one application)",
                          {"L": (0.1, 5), "g": (1, 25), "m": (0.1, 10)},
                          {"T": lambda p: 2 * math.pi * np.sqrt(p["L"] / p["g"])},
                          [("T = 2*pi*sqrt(L/g)", [T(1, T=1), T(-2 * math.pi, L=0.5, g=-0.5)])],
                          dims={"L": (0, 1, 0), "g": (0, 1, -2), "m": (1, 0, 0), "T": (0, 0, 1)},
                          base_dims=["M", "L", "T"], target_outputs=["T"]),
}


def run_challenge(law, cfg, active=True, known=()):
    rng = np.random.default_rng(cfg.seed + 100)
    print("\n" + "#" * 76)
    print(f"# HIDDEN-LAW CHALLENGE: {law.name}  (the engine is NOT told: {law.desc})")
    print("#" * 76)
    exps = law.experiments(rng)
    bundle = DataBundle(exps, cfg, law.dims, law.base_dims, law.target_outputs)
    eng = Discoverer(bundle, cfg, known, law.name)
    arch = eng.run()
    eng.report(top=12)

    # ---- score against truth on fresh, extrapolated data ----------------------
    fresh = stack_from_cols(law.lab(law.sample_inputs(rng, 400, stretch=3.0), rng, outlier=0))
    print("\n" + "-" * 76)
    print("SCOREBOARD  (truth vs discovery, fresh data drawn 3x beyond the training range)")
    print("-" * 76)
    ok_all = True
    for label, terms in law.truths:
        truth = Relation(terms[0][1], [(-c, f) for c, f in terms[1:]])
        match = [e for e in arch.all if terms_match(e.rel.terms(), terms)]
        if match:
            m = sorted(match, key=lambda e: e.cls == "REDUNDANT")[0]
            err = median_err(m.rel, fresh)
            print(f"  TRUE  {label}")
            print(f"  FOUND {m.rel.text()}   [{m.cls}]")
            print(f"        structure+constants match: YES | fresh-data error {err:.4f} | "
                  f"complexity found {m.rel.complexity():.1f} vs true {truth.complexity():.1f} | "
                  f"test-split error {m.stats['test']:.4f}")
        else:
            ok_all = False
            print(f"  TRUE  {label}\n  FOUND no matching relation")
    print(f"  => {'ALL HIDDEN LAWS RECOVERED' if ok_all else 'some laws missed'}")

    # ---- active experimentation ----------------------------------------------
    if active:
        prop = ActiveDesigner(bundle, cfg, list(law.inputs)).propose(arch)
        if prop:
            sel, e0 = prop
            new = Experiment("active_1", law.lab(sel, rng))
            b2 = DataBundle(exps + [new], cfg, law.dims, law.base_dims, law.target_outputs)
            cfg2 = Config(**{**cfg.__dict__, "verbose": False})
            arch2 = Discoverer(b2, cfg2, known, law.name).run()
            match = [e for e in arch2.entries if terms_match(e.rel.terms(), e0.rel.terms())]
            print(f"\n[ACTIVE] ran {new.n} new measurements as experiment 'active_1', re-ran discovery")
            if match:
                m = match[0]
                print(f"  hypothesis survived: {m.rel.text()}")
                print(f"  parameter uncertainty  before {e0.spread:.5f}  ->  after {m.spread:.5f}")
            else:
                print("  hypothesis did NOT survive the new experiment; new top relation:")
                for e in arch2.ranked()[:2]:
                    print("    " + e.rel.text())
    return ok_all


# =============================================================================
# 15. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="PhyDiscover V4")
    ap.add_argument("--challenge", default="all",
                    help="law1 law2 law3 law4 suite pendulum | all")
    ap.add_argument("--csv", nargs="+", help="discover on your own CSVs (one file = one experiment)")
    ap.add_argument("--no-active", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    cfg = Config(seed=a.seed, verbose=not a.quiet)

    if a.csv:
        eng = Discoverer(load_csv(a.csv, cfg), cfg, name="csv data")
        eng.run()
        eng.report()
        return
    names = list(LAWS) if a.challenge == "all" else [a.challenge]
    results = {}
    for n in names:
        known = [Relation(Feature.mono({"A": 1}), [(2.75, Feature.mono({"x": 1, "y": 1}))])] \
            if n == "suite" else ()
        results[n] = run_challenge(LAWS[n], cfg, active=not a.no_active, known=known)
    print("\n" + "=" * 76)
    print("BENCHMARK SUMMARY")
    for n, ok in results.items():
        print(f"  {n:9s} {'RECOVERED' if ok else 'MISSED'}   ({LAWS[n].desc})")


if __name__ == "__main__":
    main()
