#!/usr/bin/env python3
"""
PHY_DISCOVER - single-file hypothesis-discovery engine + autonomous agent
=========================================================================

Part 1 (sections 1-14): the engine. Given observations from several experiments
it finds simple, accurate, robust, non-redundant relationships: power laws,
invariants, sparse implicit laws, snapped constants, dimensionless groups,
equivalence detection (KNOWN/DERIVED/REDUNDANT/NOVEL), interestingness scores,
and active experiment design.

Laws it can assemble: power laws, sparse sums/ratios/logs, exp/Arrhenius/Gaussian factors,
and (damped) sin/cos of a measured input (frequency, decay, amplitude and phase are estimated).

Memory: pass a JSON file (`memory="phy_memory.json"` / `--memory`) and the engine
re-validates remembered laws instead of rediscovering them, reuses learned constants,
exponents and invented variables, remembers falsified hypotheses, and keeps notes.

Part 4 (autopilot): scans for data, infers experiments/units/noise, writes reports, asks for the
next measurements, watches a folder, and practises on worlds it invents when there is no data.

Part 2 (agent): a lab-driven agent that designs its own experiments, INVENTS new
variables (single-index search), learns exponents back into its vocabulary,
falsifies its own hypotheses on fresh far-range measurements, and invents latent
per-experiment variables.

Python:
    from phy_discover import discover, explore
    res = discover("data.csv", experiment="run_id");  print(res.summary());  res.save("report.md")
    res = explore(knobs, measure)                    # autonomous agent on an instrument

Command line:
    python phy_discover.py                      # AUTOPILOT: finds data in this folder, detects experiments,
                                                #   units and noise level itself, discovers, and writes
                                                #   phy_memory.json / phy_report.md / phy_report.json /
                                                #   phy_next_experiments_*.csv
    python phy_discover.py --watch              # keep running, re-analyse when files change
    python phy_discover.py --practice 12        # self-directed practice on random worlds it invents
    python phy_discover.py --csv a.csv b.csv    # explicit data;   --lab my_lab.py = run an instrument
    python phy_discover.py --lab-template > my_lab.py
    python phy_discover.py --challenge all      # built-in hidden-law self-test (simulated worlds)

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


# Built-in constant priors. V4 default: on. V5 turns them OFF so every constant
# is learned from data alone (e.g. 2*pi is reported as 6.2832, not recognised).
PRIORS = {"named": True, "grammar": True}
MEM_CONSTS = []      # constants confirmed in >=2 earlier runs (filled by Memory)


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
        extra = 1.2 if inv else 0.0
        for d in range(1, 6):
            s = float(f"{x:.{d - 1}e}")
            if s > 0:
                consider(d + extra, s, inv, fmt(s))
        if not inv:
            for q in range(2, 13):
                p = round(x * q)
                if 1 <= p <= 60 and math.gcd(p, q) == 1:
                    consider(1 + len(str(p)) + len(str(q)), p / q, False, f"{p}/{q}")
            if PRIORS["named"]:
                for name, val in NAMED:
                    consider(2.0, val, False, name)
            if tol <= 0.01:                    # remembered constants: reuse across datasets
                for val in MEM_CONSTS:
                    consider(1.6, val, False, fmt(val))
    if best[0] is None:
        return ConstInfo(sign * float(f"{v:.6g}"), value, se_abs, "fitted")
    _, _, shown, inv, label = best[0]
    eff = 1.0 / shown if inv else shown
    gram = GRAMMAR if PRIORS["grammar"] else [1]
    origin = "grammar" if any(abs(eff - g) < 1e-9 * max(1, g) for g in gram) else "learned"
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
        if abs(v - q) < 0.02:
            return exp_complexity(v)
    return 3.5


# =============================================================================
# 4. FEATURES AND RELATIONS
#    A relation is   lhs = sum_k coef_k * feature_k    (implicit form F = 0)
# =============================================================================
class Feature:
    """kind 'm'  : monomial  prod v^e
       kind 'log': log(v)
       kind 'em' : monomial * exp(sum q * v^k)   (exponential growth/decay, Arrhenius, Gaussian)"""
    __slots__ = ("kind", "exps", "var", "ex", "w", "fn")

    def __init__(self, kind, exps=None, var=None, ex=None, w=None, fn=None):
        self.kind, self.exps, self.var, self.ex = kind, exps or {}, var, ex or {}
        self.w, self.fn = w, fn

    @staticmethod
    def trig(fn, var, w, decay=0.0):
        """sin(w*var) or cos(w*var), optionally damped by exp(-decay*var)."""
        ex = {(var, 1): -float(decay)} if decay and abs(decay) > 1e-12 else {}
        return Feature("trig", var=var, w=float(w), fn=fn, ex=ex)

    @staticmethod
    def mono(exps):
        return Feature("m", {v: float(e) for v, e in exps.items() if abs(e) > 1e-12})

    @staticmethod
    def log(var):
        return Feature("log", var=var)

    @staticmethod
    def em(exps, ex):
        norm = {}
        for key, val in ex.items():            # accepts {(v, k): q} or {v: (k, q)}
            (v, k), q = (key, val) if isinstance(key, tuple) else ((key, val[0]), val[1])
            norm[(v, int(k))] = float(q)
        ex = {key: q for key, q in norm.items() if abs(q) > 1e-12}
        if not ex:
            return Feature.mono(exps)
        return Feature("em", {v: float(e) for v, e in exps.items() if abs(e) > 1e-12}, None, ex)

    def key(self):
        if self.kind == "m":
            return tuple(sorted((v, round(e, 3)) for v, e in self.exps.items()))
        if self.kind == "em":
            return tuple(sorted((v, round(e, 3)) for v, e in self.exps.items())) + \
                tuple(sorted((f"exp:{v}^{k}", round(q, 4)) for (v, k), q in self.ex.items()))
        if self.kind == "trig":
            return ((f"{self.fn}({self.var})", round(self.w, 4)),) + \
                tuple(sorted((f"exp:{v}^{k}", round(q, 4)) for (v, k), q in self.ex.items()))
        return ((f"log({self.var})", 1.0),)

    def vec(self):
        if self.kind == "m":
            return dict(self.exps)
        if self.kind == "em":
            d = dict(self.exps)
            for (v, k), q in self.ex.items():
                d[f"exp:{v}^{k}"] = q
            return d
        if self.kind == "trig":
            d = {f"{self.fn}:{self.var}": self.w}
            for (v, k), q in self.ex.items():
                d[f"exp:{v}^{k}"] = q
            return d
        return {"log:" + self.var: 1.0}

    def vars(self):
        if self.kind == "m":
            return set(self.exps)
        if self.kind == "em":
            return set(self.exps) | {v for v, _ in self.ex}
        return {self.var}

    def eval(self, cols):
        n = _nrows(cols)
        with np.errstate(all="ignore"):
            if self.kind in ("m", "em"):
                out = np.ones(n)
                for v, e in self.exps.items():
                    out = out * (cols[v] if e == 1.0 else np.power(cols[v], e))
                if self.kind == "em":
                    arg = np.zeros(n)
                    for (v, k), q in self.ex.items():
                        arg = arg + q * (cols[v] if k == 1 else np.power(cols[v], k))
                    out = out * np.exp(np.clip(arg, -700, 700))
                return out
            if self.kind == "trig":
                arg = self.w * cols[self.var]
                out = np.sin(arg) if self.fn == "sin" else np.cos(arg)
                for (v, k), q in self.ex.items():
                    out = out * np.exp(np.clip(q * (cols[v] if k == 1 else np.power(cols[v], k)), -700, 700))
                return out
            return np.log(cols[self.var])

    def complexity(self):
        if self.kind == "log":
            return 3.0
        if self.kind == "trig":
            return 4.0 + 2.0 * len(self.ex)
        base = 1.0 + sum(exp_complexity(e) for e in self.exps.values())
        return base + 2.0 * len(self.ex)

    def text(self):
        if self.kind == "log":
            return f"log({self.var})"
        if self.kind == "trig":
            core = f"{self.fn}({fmt(self.w)}*{self.var})"
            env = "".join(f"exp({fmt(q)}*{v})*" for (v, k), q in sorted(self.ex.items()))
            return env + core
        base = mono_text(self.exps) if self.exps else ""
        if self.kind == "m":
            return base or "1"
        facs = []
        for (v, k), q in sorted(self.ex.items()):
            if k == 1:
                facs.append(f"exp({fmt(q)}*{v})")
            elif k == -1:
                facs.append(f"exp({fmt(q)}/{v})")
            else:
                facs.append(f"exp({fmt(q)}*{v}^2)")
        return "*".join(([base] if base else []) + facs)


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


def is_single_variable(terms):
    """True if, after cancelling any common monomial factor, the relation involves
    at most one variable (e.g. y = y*g(Y) is really 1 = g(Y): not a law)."""
    for rep in _reps(terms):
        vs = set()
        for vec, _ in rep:
            vs |= {k.replace("log:", "") for k in vec}
        if len(vs) <= 1:
            return True
    return False


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
        within = False
        for e in range(bundle.full.n_exp):
            s = a[bundle.full.exp_id == e]
            within |= (np.std(np.log(s)) > 1e-3) if pos else (np.std(s) > 1e-9)
        between = (np.std(np.log(a)) > 1e-3) if pos else (np.std(a) > 1e-9)
        varying = bool(within or between)      # a value set once per experiment still varies
        span = math.log10(a.max() / a.min()) if pos else float("nan")
        info[v] = dict(positive=pos, varying=varying, between_only=bool(between and not within))
        flags = ("log-able " if pos else "has<=0 ") + ("" if varying else "CONSTANT") + \
            ("set once per experiment" if info[v]["between_only"] else "")
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
        self.recalled, self.memory_id, self.warn = False, None, []


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
        if f.kind in ("m", "em"):
            for v, e in f.exps.items():
                vec += e * np.array(dims[v], float)
            if f.kind == "em" and any(np.any(np.array(dims[v]) != 0) for v, _ in f.ex):
                return None
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
        # globally centred logs: keep BETWEEN-experiment variation, so a variable that is set once per
        # experiment (e.g. gravity per planet, temperature per run) can take part in a law
        self.cent_g = {v: self.logs[v] - self.logs[v].mean() for v in self.vars}
        self.rng = np.random.default_rng(cfg.seed + 1)
        self.hints = []
        self.order = {v: i for i, v in enumerate(bundle.variables)}

    @staticmethod
    def _null(X):
        _, s, vt = np.linalg.svd(X, full_matrices=False)
        return vt[-1], s

    def fit_subset(self, sub, cent=None):
        X = np.column_stack([(cent or self.cent)[v] for v in sub])
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

    def const_only(self, lhs, q, mask):
        y = self.logs[lhs] - sum(q[v] * self.logs[v] for v in q)
        return float(np.median(y[mask]))

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
            cb = self.const_only(lhs, qb, mask)
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
                c = self.fit_subset(sub, self.cent_g) or self.fit_subset(sub)
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
            # A power law is one relation however it is written (g = 40 L/T^2 is T = 0.16 sqrt(L g^-1)...),
            # but its relative error is larger when the left side has a large exponent on the noisy
            # variable. Judge it in the most noise-sensitive orientation, not the one we happened to print.
            big = max([1.0] + [abs(e) for e in rel.rhs[0][1].exps.values()])
            if big > 1.0:
                for key in ("train", "val", "test", "max_exp"):
                    st[key] = st[key] / big
                st["per_exp"] = [x / big for x in st["per_exp"]]
                exp_info = list(exp_info) + [f"(error shown in the most noise-sensitive orientation, "
                                             f"i.e. divided by {big:g} as for this equation written with "
                                             f"the largest-exponent variable on the left)"]
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


def boot_lstsq(A, t, idx, B, rng):
    """Bootstrap least squares via multinomial row weights on the Gram matrix
    (no fancy-indexing copies of the big matrix)."""
    Ai, ti = A[idx], t[idx]
    n, k = Ai.shape
    out = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        G = Ai.T @ (Ai * w[:, None])
        try:
            out.append(np.linalg.solve(G + 1e-12 * np.eye(k), Ai.T @ (ti * w)))
        except np.linalg.LinAlgError:
            continue
    return np.array(out)


class ImplicitFinder:
    def __init__(self, bundle, cfg, basis, err_cap=None, extra_exps=(), trig=()):
        self.b, self.cfg, self.basis = bundle, cfg, basis
        self.extra_exps = tuple(extra_exps)
        self.trig = [(v, w, k) for v, w, k in trig if v in basis]
        self.kmax = cfg.max_terms + (1 if self.trig else 0)
        self.err_cap = err_cap or cfg.accept_err
        self.rng = np.random.default_rng(cfg.seed + 2)
        tr = bundle.train
        feats, cols, seen = [], [], {}
        for f in self._library():
            col = f.eval(tr.cols)
            if not np.all(np.isfinite(col)):
                continue
            sig = tuple(np.round(col[:40] / (np.linalg.norm(col[:40]) + 1e-300), 6))
            if sig in seen:                     # numerically identical feature
                j = seen[sig]                   # (e.g. an invented variable u == x*y)
                if f.complexity() < feats[j].complexity():
                    feats[j], cols[j] = f, col
                continue
            seen[sig] = len(feats)
            feats.append(f)
            cols.append(col)
        self.feats, self.F = feats, np.column_stack(cols)

    def _library(self):
        B, cfg = self.basis, self.cfg
        S1 = tuple(sorted(set((-2, -1, -0.5, 0.5, 1, 1.5, 2, 3)) | set(self.extra_exps)))

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
        for v, w, k in self.trig:
            fs += [Feature.trig("sin", v, w, k), Feature.trig("cos", v, w, k)]
        return fs

    def fit_lhs(self, lhs, allow_self=True):
        cfg, tr = self.cfg, self.b.train
        L = lhs.eval(tr.cols)
        ok = np.isfinite(L) & (np.abs(L) > 1e-12)
        if lhs.exps and ok.any():     # relative-error weights blow up where lhs crosses 0
            ok &= np.abs(L) > 0.02 * np.median(np.abs(L[ok]))
        if ok.sum() < 50:
            return None
        lv = lhs.vars()
        cols = [j for j, f in enumerate(self.feats)
                if not (f.kind == "m" and f.key() == lhs.key())
                and (allow_self or not (f.vars() & lv))]
        if len(cols) < 3:
            return None
        w = 1.0 / np.abs(L[ok])
        t = np.sign(L[ok])
        Phi = self.F[ok][:, cols] * w[:, None]
        nrm = np.linalg.norm(Phi, axis=0)
        nrm[nrm < 1e-12] = 1e12

        def normed(A):
            n = np.linalg.norm(A, axis=0)
            n[n < 1e-12] = 1e12
            return A / n

        K = self.kmax
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
            A = Phi[:, S]
            sc = np.linalg.norm(A[mask], axis=0)
            C = boot_lstsq(A / sc, t, np.where(mask)[0], cfg.n_boot, self.rng) / sc
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
        for (v, w, kd) in {(f.var, f.w, -sum(f.ex.values())) for f in fs if f.kind == "trig"}:
            same = lambda f: f.kind == "trig" and f.var == v and f.w == w and abs(-sum(f.ex.values()) - kd) < 1e-12
            a = sum(c for c, f in zip(coefs, fs) if same(f) and f.fn == "sin")
            b_ = sum(c for c, f in zip(coefs, fs) if same(f) and f.fn == "cos")
            env = f"exp({fmt(-kd)}*{v})*" if kd else ""
            params.append(f"{'damped ' if kd else ''}periodic in {v}: amplitude {math.hypot(a, b_):.4g}, "
                          f"angular frequency {fmt(w)} (period {2 * math.pi / w:.4g}), "
                          + (f"decay rate {fmt(kd)}, " if kd else "")
                          + f"phase {math.atan2(b_, a):.4g} rad  [= amplitude*{env}sin({fmt(w)}*{v} + phase)]")
        return rel, float(np.max(se / np.abs(c))), params, committee

    def run(self, lhs_vars=None):
        out = []
        lhs_list = ([Feature.mono({v: 1}) for v in lhs_vars] if lhs_vars is not None
                    else [Feature.mono({})] + [Feature.mono({v: 1}) for v in self.basis])
        capped = Config(**{**self.cfg.__dict__, "accept_err": self.err_cap})
        for lhs in lhs_list:
            # Occam order: a model that does not use the quantity it predicts is
            # preferred, unless the self-referential (implicit) one is clearly
            # better. Self-referential terms can otherwise fit the noise of lhs.
            options = []
            for allow_self in ((False, True) if lhs.exps else (True,)):
                res = self.fit_lhs(lhs, allow_self)
                if res is None:
                    continue
                rel, spread, params, committee = res
                if len(rel.variables()) < 2 or is_single_variable(rel.terms()):
                    continue                   # a variable explaining itself is not a law
                options.append((evaluate_relation(rel, self.b), rel, spread, params, committee))
            if not options:
                continue
            best = options[0]
            for o in options[1:]:
                # noise in the predicted quantity appears on both sides of a self-referential fit,
                # which alone makes it look ~2x better; demand a much larger margin
                if o[0]["train"] < 0.3 * best[0]["train"]:
                    best = o
            st, rel, spread, params, committee = best
            if accept(st, capped):
                out.append(Entry(rel, "implicit", st, spread, params, committee))
        return out


# =============================================================================
# 9b. SEMI-LOG FINDER   y = C * prod v^p * exp(sum q * v^k),  k in {1, -1, 2}
#     Exponential growth/decay, Arrhenius rates, Gaussians. Sparse regression of
#     log y on {log v, v, 1/v, v^2}; an exponential factor must beat the best
#     pure power law by 2x to be accepted.
# =============================================================================
KIND_K = {"lin": 1, "inv": -1, "sq": 2}


class SemiLogFinder:
    def __init__(self, bundle, cfg, targets, loggable, varying):
        self.b, self.cfg = bundle, cfg
        self.rng = np.random.default_rng(cfg.seed + 3)
        tr = bundle.train
        self.targets = [t for t in targets if t in loggable]
        self.cols, raw = [], []
        for v in bundle.variables:
            if v not in varying:
                continue
            a = tr.cols[v]
            opts = [("lin", a), ("sq", a ** 2)]
            if v in loggable:
                opts = [("log", np.log(a)), ("inv", 1.0 / a)] + opts
            for kind, arr in opts:
                self.cols.append((v, kind))
                raw.append(arr)
        self.raw = np.column_stack(raw) if raw else np.zeros((tr.n, 0))
        self.cent = self._center(self.raw)
        self.ylog = {t: np.log(tr.cols[t]) for t in self.targets}
        self.ycent = {t: self._center(self.ylog[t][:, None])[:, 0] for t in self.targets}

    def _center(self, M):
        out = M.copy()
        tr = self.b.train
        for e in range(tr.n_exp):
            m = tr.exp_id == e
            if m.any():
                out[m] -= out[m].mean(axis=0)
        return out

    def _select(self, Xn, y):
        cfg = self.cfg
        K = cfg.max_terms
        m1 = beam_search(Xn, y, K, cfg.beam)
        S = list(m1[max(m1)][1])
        c = np.linalg.lstsq(Xn[:, S], y, rcond=None)[0]
        inl = trim_mask(y - Xn[:, S] @ c, cfg.trim_k)
        n2, L = int(inl.sum()), Xn.shape[1]
        m2 = beam_search(Xn[inl], y[inl], K, cfg.beam)
        bic = {k: n2 * math.log(max(r, 0) / n2 + 1e-12) + k * (math.log(n2) + 2 * math.log(L))
               for k, (r, _) in m2.items()}
        return list(m2[min(bic, key=bic.get)][1])

    def _fit_model(self, t, eligible):
        cfg = self.cfg
        if len(eligible) < 2:
            return None
        y = self.ycent[t]
        X = self.cent[:, eligible]
        nrm = np.linalg.norm(X, axis=0)
        nrm[nrm < 1e-12] = 1e12
        Xn = X / nrm
        S = self._select(Xn, y)
        while True:
            mask = np.ones(len(y), bool)
            for _ in range(5):
                c = np.linalg.lstsq(Xn[mask][:, S], y[mask], rcond=None)[0]
                new = trim_mask(y - Xn[:, S] @ c, cfg.trim_k)
                if (new == mask).all():
                    break
                mask = new
            C = boot_lstsq(Xn[:, S], y, np.where(mask)[0], cfg.n_boot, self.rng)
            se = C.std(0)
            tst = np.abs(c) / np.maximum(se, 1e-15)
            if tst.min() < 6:
                if len(S) == 1:
                    return None
                S.pop(int(np.argmin(tst)))
                continue
            break
        cols = [eligible[j] for j in S]
        return cols, c / nrm[S], se / nrm[S], C / nrm[S], mask

    def _build(self, t, cols, beta, mask):
        exps, ex = {}, {}
        for j, b in zip(cols, beta):
            v, kind = self.cols[j]
            if kind == "log":
                exps[v] = exps.get(v, 0) + b
            else:
                ex[(v, KIND_K[kind])] = b
        pred = self.raw[:, cols] @ np.asarray(beta)
        c = float(np.median((self.ylog[t] - pred)[mask]))
        f = Feature.em(exps, ex)
        return Relation(Feature.mono({t: 1}), [(math.exp(c), f)]), c

    def _fit_target(self, t, eligible):
        res = self._fit_model(t, eligible)
        if res is None:
            return None
        cols, beta, se, C, mask = res
        tr = self.b.train
        rel, c = self._build(t, cols, beta, mask)
        if rel is None:
            return None
        err_free = median_err(rel, tr)
        beta_s, info = list(beta), [None] * len(beta)
        for j in np.argsort(-np.abs(beta)):
            kind = self.cols[cols[j]][1]
            if kind == "log":
                val, ok = snap_exponent(beta[j], se[j])
            else:
                ci = snap_const(beta[j], se[j] / abs(beta[j]))
                val, ok = ci.value, ci.origin != "fitted"
            trial = list(beta_s)
            trial[j] = val
            r2, _ = self._build(t, cols, trial, mask)
            if r2 is not None and median_err(r2, tr) <= err_free * 1.25 + 3e-4:
                beta_s = trial
                info[j] = "snapped" if ok else "fitted"
            else:
                info[j] = "fitted"
        rel, c = self._build(t, cols, beta_s, mask)
        if rel is None:
            return None
        res_c = (self.ylog[t] - self.raw[:, cols] @ np.asarray(beta_s) - c)[mask]
        se_c = 1.2533 * max(mad_sigma(res_c), 1e-9) / math.sqrt(mask.sum())
        ci = snap_const(math.exp(c), se_c)
        rel = Relation(rel.lhs, [(ci.value, rel.rhs[0][1])], [ci])
        params = []
        for j, bj, sj, ij in zip(cols, beta_s, se, info):
            v, kind = self.cols[j]
            what = f"exponent of {v}" if kind == "log" else f"rate in exp({v}^{KIND_K[kind]})"
            params.append(f"{what}: {fmt(bj)} [{ij}, +- {sj:.2g}]")
        params.append(f"constant {ci.describe()}")
        committee = []
        for row in C[: self.cfg.committee]:
            r2, _ = self._build(t, cols, list(row), mask)
            if r2 is not None:
                committee.append(r2)
        rel_spread = max([sj if self.cols[j][1] == "log" else sj / max(abs(bj), 1e-12)
                          for j, bj, sj in zip(cols, beta_s, se)] + [se_c])
        return rel, rel_spread, params, committee

    def run(self):
        out = []
        for t in self.targets:
            elig_all = [j for j, (v, k) in enumerate(self.cols) if v != t]
            elig_log = [j for j in elig_all if self.cols[j][1] == "log"]
            semi = self._fit_target(t, elig_all)
            if semi is None or semi[0].rhs[0][1].kind != "em":
                continue
            st = evaluate_relation(semi[0], self.b)
            pure = self._fit_target(t, elig_log)
            err_p = median_err(pure[0], self.b.train) if pure else float("inf")
            if st["train"] < 0.5 * err_p and accept(st, self.cfg):
                rel, spread, params, committee = semi
                out.append(Entry(rel, "semilog", st, spread, params, committee))
        return out


# =============================================================================
# 9c. PERIODIC SCAN: find frequencies w such that a reading contains sin(w v)/cos(w v)
#     Slow trends are projected out first; every frequency on a log grid is tested in
#     closed form (vectorised); peaks are refined, their uncertainty estimated from
#     the RSS curve, and snapped like any constant. Found frequencies become features
#     of the sparse search, so  P = 5 + 3*sin(2.1*x) + 2*z  can be assembled.
# =============================================================================
class PeriodicScan:
    """Finds (variable, w, decay) such that a reading contains exp(-decay*v)*sin/cos(w*v)."""

    K_GRID = (0.3, 0.6, 1.0, 1.6, 2.5, 4.0, 6.0, 10.0, 16.0, 25.0)   # decay, in e-folds across the core range

    def __init__(self, bundle, cfg, targets, varying, max_peaks=2):
        self.b, self.cfg, self.max_peaks = bundle, cfg, max_peaks
        self.targets = [t for t in targets if t in varying]
        self.varying = list(varying)
        outs = set(bundle.outputs)
        self.sources = [v for v in varying if v not in outs] or list(varying)

    @staticmethod
    def _rss_grid(ws, ks, x, yr, Q, wgt):
        """RSS of  y ~ baseline + a*e^(-k x) sin(w x) + b*e^(-k x) cos(w x)  for paired (w, k)."""
        env = np.exp(-np.outer(ks, x - x.min())) if np.any(ks) else 1.0
        arg = np.outer(ws, x)
        S = np.sin(arg) * env * wgt
        C = np.cos(arg) * env * wgt
        S -= (S @ Q) @ Q.T
        C -= (C @ Q) @ Q.T
        ss, cc, sc = (S * S).sum(1), (C * C).sum(1), (S * C).sum(1)
        sy, cy = S @ yr, C @ yr
        det = np.maximum(ss * cc - sc * sc, 1e-18)
        a = (sy * cc - cy * sc) / det
        b = (cy * ss - sy * sc) / det
        return (yr @ yr) - a * sy - b * cy

    @staticmethod
    def _interval(grid, rss, n):
        if not np.all(np.isfinite(rss)):
            return float("inf")
        rmin = rss.min()
        inside = grid[rss <= rmin * (1 + 1.0 / max(n - 7, 1))]
        if inside.size == 0:
            return float("inf")
        return max((inside.max() - inside.min()) / 2, 1e-12)

    def _peaks(self, x, y, max_peaks, extra):
        wgt = 1.0 / np.abs(y)
        R = x.max() - x.min()
        Rc = max(np.percentile(x, 95) - np.percentile(x, 5), 1e-12)   # core range (robust to far outliers)
        n = len(y)
        xs = (x - x.mean()) / (x.std() + 1e-300)
        base = [np.ones_like(xs), xs, xs ** 2] + list(extra)
        peaks = []
        for _ in range(max_peaks):
            Q, _ = np.linalg.qr(np.column_stack(base) * wgt[:, None])
            yw = y * wgt
            yr = yw - Q @ (Q.T @ yw)
            rss0 = float(yr @ yr)
            lo, hi = math.pi / R, min(40 * math.pi / R, 0.4 * n / R)
            if hi <= lo * 1.5:
                break
            ws = np.exp(np.linspace(math.log(lo), math.log(hi), 320))
            zero = np.zeros_like(ws)
            rss = self._rss_grid(ws, zero, x, yr, Q, wgt)
            if not np.all(np.isfinite(rss)):
                break
            k0 = int(np.argmin(rss))
            if rss[k0] > 0.9 * rss0:                         # nothing periodic here: stop early
                break
            best = (float(rss[k0]), ws[k0], 0.0)
            # damped candidates (need >= 1.5 periods so decay and frequency are separable)
            wmin_d = 2 * math.pi * 1.5 / Rc
            sel = ws >= wmin_d
            if sel.any():
                for c in self.K_GRID:
                    kk = c / Rc
                    r = self._rss_grid(ws[sel], np.full(sel.sum(), kk), x, yr, Q, wgt)
                    j = int(np.argmin(r))
                    if r[j] < best[0] * 0.5:                  # damping must clearly help
                        best = (float(r[j]), ws[sel][j], kk)
            rmin, w0, k = best
            if k == 0.0 and k0 in (0, len(ws) - 1):
                break
            for it in range(3):                               # alternate zooms on w and k
                half = (0.02, 0.003, 0.0006)[it]
                fine = np.linspace(w0 * (1 - half), w0 * (1 + half), 201)
                r2 = self._rss_grid(fine, np.full(201, k), x, yr, Q, wgt)
                w0 = fine[int(np.argmin(r2))]
                if k > 0:
                    kf = np.linspace(k * 0.5, k * 1.6, 201)
                    r4 = self._rss_grid(np.full(201, w0), kf, x, yr, Q, wgt)
                    k = kf[int(np.argmin(r4))]
            fine = np.linspace(w0 * 0.995, w0 * 1.005, 801)
            r3 = self._rss_grid(fine, np.full(801, k), x, yr, Q, wgt)
            rmin = float(r3.min())
            w0 = fine[int(np.argmin(r3))]
            se_w = self._interval(fine, r3, n)
            if not math.isfinite(se_w):
                break
            se_k = 0.0
            if k > 0:
                kf = np.linspace(k * 0.7, k * 1.3, 601)
                r5 = self._rss_grid(np.full(601, w0), kf, x, yr, Q, wgt)
                se_k = self._interval(kf, r5, n)
            if not (rmin < 0.5 * rss0 and n * math.log(rss0 / max(rmin, 1e-300)) > 80):
                break
            peaks.append((w0, se_w, k, se_k))
            env = np.exp(-k * (x - x.min()))
            base += [np.sin(w0 * x) * env, np.cos(w0 * x) * env]
        return peaks

    def _scan_pair(self, t, v):
        tr = self.b.train
        y, x = tr.cols[t], tr.cols[v]
        ok = np.isfinite(y) & np.isfinite(x) & (np.abs(y) > 1e-12)
        if ok.sum() < 80 or np.ptp(x[ok]) <= 0 or len(np.unique(np.round(x[ok], 12))) < 15:
            return []          # periodicity needs a continuously varied input (>= 15 distinct values)
        extra = []
        for u in self.varying:
            if u in (t, v):
                continue
            a = tr.cols[u][ok]
            if np.std(a) <= 0:
                continue
            z = (a - a.mean()) / a.std()
            extra += [z, z ** 2]
            if np.all(a > 0):
                lg = np.log(a)
                extra.append((lg - lg.mean()) / (lg.std() + 1e-300))
        y, x = y[ok], x[ok]
        peaks = self._peaks(x, y, self.max_peaks, extra)
        if not peaks:
            return []
        xs = (x - x.mean()) / x.std()
        cols = [np.ones_like(x), xs, xs ** 2] + list(extra)
        for w, _, k, _ in peaks:
            env = np.exp(-k * (x - x.min()))
            cols += [np.sin(w * x) * env, np.cos(w * x) * env]
        A = np.column_stack(cols) / np.abs(y)[:, None]
        coef = np.linalg.lstsq(A, np.sign(y), rcond=None)[0]
        keep = trim_mask(np.sign(y) - A @ coef, self.cfg.trim_k)      # drop outliers, rescan
        if keep.sum() >= 80:
            x, y = x[keep], y[keep]
            extra = [e[keep] for e in extra]
            peaks = self._peaks(x, y, self.max_peaks, extra) or peaks
        out = []
        for w0, se_w, k, se_k in peaks:
            ci = snap_const(w0, se_w / w0)
            w = ci.value if ci.origin != "fitted" else float(f"{w0:.6g}")
            kk = 0.0
            if k > 0:
                ck = snap_const(k, se_k / k)
                kk = ck.value if ck.origin != "fitted" else float(f"{k:.6g}")
            out.append((w, kk))
        return out

    def run(self):
        found = {}
        for t in self.targets:
            for v in self.sources:
                if v == t:
                    continue
                for w, k in self._scan_pair(t, v):
                    lst = found.setdefault(v, [])
                    if not any(abs(w - w2) / w2 < 2e-3 and abs(k - k2) <= 0.05 * max(k, k2, 1e-9)
                               for w2, k2 in lst):
                        lst.append((w, k))
        return [(v, w, k) for v, lst in found.items() for w, k in lst]


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
        if len(rel.rhs) != 1 or rel.lhs.kind != "m" or rel.rhs[0][1].kind not in ("m", "em"):
            return None
        c, f = rel.rhs[0]
        if c <= 0:
            return None
        p = dict(rel.lhs.exps)
        for v, e in f.exps.items():
            p[v] = p.get(v, 0) - e
        if f.kind == "em":                      # exp(q v^k) factors are extra "log-space" coordinates
            for (v, k), q in f.ex.items():
                p[f"exp:{v}^{k}"] = p.get(f"exp:{v}^{k}", 0) - q
        return p, math.log(c)

    def _mono_class(self, p, logc):
        if not self.mono:
            return "NOVEL", None
        keys = sorted(set(p) | {k for q, _ in self.mono for k in q})
        scale = np.array([max([abs(p.get(k, 0))] + [abs(q.get(k, 0)) for q, _ in self.mono]) or 1.0
                          for k in keys])                 # exp-rates (~1000) and exponents (~1) comparable
        vp = np.array([p.get(k, 0) for k in keys]) / scale
        M = np.array([[q.get(k, 0) for k in keys] for q, _ in self.mono]) / scale
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

    def remove(self, entry):
        """Withdraw an entry (a superseded approximation) and rebuild the law lattice."""
        self.entries = [e for e in self.entries if e is not entry]
        self.all = [e for e in self.all if e is not entry]
        self.mono = []
        for e in self.entries:
            mv = self._mono_vec(e.rel)
            if mv and e.cls in ("NOVEL", "KNOWN", "DEFINITION"):
                self.mono.append(mv)

    def add_recalled(self, entry):
        entry.cls, entry.recalled = "KNOWN", True
        self.entries.append(entry)
        self.all.append(entry)
        mv = self._mono_vec(entry.rel)
        if mv:
            self.mono.append(mv)

    def ranked(self):
        return sorted((e for e in self.entries if e.cls != "DEFINITION"),
                      key=lambda e: -e.scores.get("interestingness", 0))


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


import time as _time
# =============================================================================
# 11b. PERSISTENT MEMORY  (a JSON file the engine reads at start and writes at end)
#
#   laws        validated relations (confirmed / failed counts) -> recalled and
#               re-validated in milliseconds instead of rediscovered
#   falsified   hypotheses that died on out-of-range data -> never trusted blindly again
#   constants   learned constants seen in >= 2 runs -> reused when snapping
#   exponents   exponents seen in >= 2 runs -> added to the search vocabulary
#   inventions  invented variables (index vectors) -> re-adopted after a 1-line check
#   stages      cost/yield per search stage -> unproductive stages are sampled less
#   notes       the engine's own log: what it found, what failed, lessons, hints
#   history     one line per run
# =============================================================================
import datetime as _dt
import json as _json
import os as _os


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def feature_to_dict(f):
    return {"k": f.kind, "e": dict(f.exps), "v": f.var, "w": f.w, "fn": f.fn,
            "x": [[v, k, q] for (v, k), q in f.ex.items()]}


def feature_from_dict(d):
    if d["k"] == "m":
        return Feature.mono(d["e"])
    if d["k"] == "log":
        return Feature.log(d["v"])
    if d["k"] == "trig":
        return Feature.trig(d["fn"], d["v"], d["w"], decay=-sum(q for _, _, q in d["x"]))
    return Feature.em(d["e"], {(v, int(k)): q for v, k, q in d["x"]})


def relation_to_dict(rel):
    def ci(c):
        return None if c is None else dict(value=c.value, raw=c.raw, se=c.se, origin=c.origin,
                                           label=c.label, inv=c.inv)
    return {"lhs": feature_to_dict(rel.lhs),
            "rhs": [{"c": float(c), "f": feature_to_dict(f), "ci": ci(i)}
                    for (c, f), i in zip(rel.rhs, rel.cinfo)]}


def relation_from_dict(d):
    rhs, infos = [], []
    for t in d["rhs"]:
        rhs.append((t["c"], feature_from_dict(t["f"])))
        infos.append(None if t["ci"] is None else ConstInfo(**t["ci"]))
    return Relation(feature_from_dict(d["lhs"]), rhs, infos)


class Memory:
    VERSION = 1
    MAX_NOTES, MAX_HISTORY = 400, 200
    BASE_EXPS = {-2.0, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0}

    def __init__(self, path="phy_memory.json", domain="", autosave=True):
        self.path, self.domain, self.autosave = path, domain, autosave
        self.data = self._fresh()
        self.session = dict(recalled=0, new=0, failed=0, notes=0, falsified=0)
        if path and _os.path.exists(path):
            self._load()
        self._next = 1 + max([r["id"] for r in self.data["laws"] + self.data["retired"]], default=0)
        self._refresh_constants()

    # ---- persistence ------------------------------------------------------
    @staticmethod
    def _now():
        return _dt.datetime.now().isoformat(timespec="seconds")

    def _fresh(self):
        t = self._now()
        return {"version": self.VERSION, "created": t, "updated": t, "runs": 0,
                "laws": [], "retired": [], "falsified": [], "constants": [],
                "constant_relations": [], "exponents": {}, "inventions": [],
                "stages": {}, "notes": [], "history": [], "skills": {}}

    def _load(self):
        try:
            with open(self.path) as f:
                d = _json.load(f)
            base = self._fresh()
            base.update({k: v for k, v in d.items() if k in base})
            self.data = base
        except Exception as ex:           # corrupt file: keep it, start clean, say so
            bad = f"{self.path}.corrupt-{int(_time.time())}"
            try:
                _os.replace(self.path, bad)
            except OSError:
                pass
            self.data = self._fresh()
            self.note("warning", f"memory file was unreadable ({type(ex).__name__}); "
                                 f"moved to {_os.path.basename(bad)} and started fresh")

    def save(self):
        if not self.path:
            return
        self.data["updated"] = self._now()
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            _json.dump(_plain(self.data), f, indent=1)
        _os.replace(tmp, self.path)       # atomic: a crash never leaves a half-written file

    # ---- notes -------------------------------------------------------------
    def note(self, kind, text, **data):
        for n in self.data["notes"]:
            if n["kind"] == kind and n["text"] == text:
                n["count"] += 1
                n["t"] = self._now()
                return
        self.data["notes"].append(dict(t=self._now(), kind=kind, text=text, count=1, **_plain(data)))
        self.session["notes"] += 1
        if len(self.data["notes"]) > self.MAX_NOTES:
            drop = next((i for i, n in enumerate(self.data["notes"]) if n["count"] == 1), 0)
            self.data["notes"].pop(drop)

    # ---- laws ---------------------------------------------------------------
    def recall(self, variables):
        vs = set(variables)
        out = []
        for rec in self.data["laws"]:
            if self.domain and rec.get("domain", "") != self.domain:
                continue
            if set(rec["vars"]) <= vs:
                try:
                    out.append((rec, relation_from_dict(rec["rel"])))
                except Exception:
                    continue
        return out

    def _find_law(self, rel):
        vs = set(rel.variables())
        for rec in self.data["laws"]:
            if set(rec["vars"]) == vs:
                try:
                    if terms_match(relation_from_dict(rec["rel"]).terms(), rel.terms()):
                        return rec
                except Exception:
                    pass
        return None

    def remember_law(self, entry, variables, ranges, dataset):
        rel, t = entry.rel, self._now()
        rec = self._find_law(rel)
        if rec:
            rec["seen"] += 1
            rec["confirmed"] += 1
            rec["last_seen"] = t
            for v, (lo, hi) in ranges.items():
                if v in rec["ranges"]:
                    rec["ranges"][v] = [min(lo, rec["ranges"][v][0]), max(hi, rec["ranges"][v][1])]
            if entry.spread < rec.get("spread", 1e9):      # keep the better-determined constants
                rec.update(rel=relation_to_dict(rel), text=rel.text(), spread=float(entry.spread),
                           err=float(entry.stats["test"]))
            return False
        self.data["laws"].append(dict(
            id=self._next, text=rel.text(), rel=relation_to_dict(rel),
            vars=sorted(rel.variables()), context=sorted(variables), domain=self.domain,
            source=entry.source, seen=1, confirmed=1, failed=0, first_seen=t, last_seen=t,
            spread=float(entry.spread), err=float(entry.stats["test"]),
            ranges={v: ranges[v] for v in rel.variables() if v in ranges}, dataset=dataset))
        self._next += 1
        self.session["new"] += 1
        self._unfalsify(rel)
        self.note("discovery", f"learned: {rel.text()}  (test error {entry.stats['test']:.2%}, "
                               f"dataset '{dataset}')")
        return True

    def apply_validation(self, rid, ok, jaccard, dataset):
        rec = next((r for r in self.data["laws"] if r["id"] == rid), None)
        if rec is None:
            return
        if ok:
            rec["confirmed"] += 1
            rec["seen"] += 1
            rec["last_seen"] = self._now()
            self.session["recalled"] += 1
        elif jaccard >= 0.5:          # only blame the law if the dataset really is its context
            rec["failed"] += 1
            self.session["failed"] += 1
            self.note("lesson", f"remembered law failed on '{dataset}': {rec['text']}")
            if rec["failed"] >= 3 and rec["failed"] > rec["confirmed"]:
                self.data["laws"].remove(rec)
                self.data["retired"].append(rec)
                self.note("lesson", f"retired law (failed {rec['failed']}x, confirmed "
                                    f"{rec['confirmed']}x): {rec['text']}")

    def falsify(self, rel, reason, variables):
        for f in self.data["falsified"]:
            try:
                if set(f["vars"]) == set(rel.variables()) and \
                   terms_match(relation_from_dict(f["rel"]).terms(), rel.terms()):
                    f["count"] += 1
                    f["t"] = self._now()
                    return
            except Exception:
                pass
        self.data["falsified"].append(dict(text=rel.text(), rel=relation_to_dict(rel),
                                           vars=sorted(rel.variables()), reason=reason,
                                           t=self._now(), count=1))
        self.session["falsified"] += 1
        self.note("falsified", f"{rel.text()}  - {reason}")

    def is_falsified(self, rel, variables):
        vs = set(variables)
        for f in self.data["falsified"]:
            if set(f["vars"]) <= vs and set(f["vars"]) == set(rel.variables()):
                try:
                    if terms_match(relation_from_dict(f["rel"]).terms(), rel.terms()):
                        return f
                except Exception:
                    pass
        return None

    def _unfalsify(self, rel):
        keep = []
        for f in self.data["falsified"]:
            try:
                same = set(f["vars"]) == set(rel.variables()) and \
                    terms_match(relation_from_dict(f["rel"]).terms(), rel.terms())
            except Exception:
                same = False
            if same:
                self.note("lesson", f"previously falsified law now holds on new data: {f['text']}")
            else:
                keep.append(f)
        self.data["falsified"] = keep

    # ---- constants -----------------------------------------------------------
    @staticmethod
    def _simple(v):
        """<= 2 significant digits (2, 0.5, 4.2, 30000 ...): trivial to snap, not worth remembering."""
        return abs(float(f"{v:.1e}") - v) <= 1e-9 * abs(v)

    def remember_constants(self, items):
        for value, se_rel, text in items:
            v = abs(float(value))
            if not math.isfinite(v) or v <= 0 or self._simple(v):
                continue
            se = max(float(se_rel), 1e-7)
            for c in self.data["constants"]:
                if abs(c["value"] - v) / v <= min(max(3 * max(se, c["se_rel"]), 1e-6), 0.01):
                    c["count"] += 1
                    if se < c["se_rel"]:
                        c["value"], c["se_rel"] = v, se
                    if text not in c["where"]:
                        c["where"] = (c["where"] + [text])[-4:]
                    break
            else:
                self.data["constants"].append(dict(value=v, se_rel=se, count=1, where=[text]))
        self._refresh_constants()
        self._mine_constant_relations()

    def confirmed_constants(self):
        return [c["value"] for c in self.data["constants"]
                if c["count"] >= 2 and c["se_rel"] <= 0.003]

    def _refresh_constants(self):
        MEM_CONSTS[:] = self.confirmed_constants()

    def _mine_constant_relations(self):
        good = [c for c in self.data["constants"] if c["count"] >= 2 and c["se_rel"] <= 5e-4]
        forms = [("{b} = {a}^2", lambda a: a * a), ("{b} = sqrt({a})", math.sqrt),
                 ("{b} = 2*{a}", lambda a: 2 * a), ("{b} = {a}/2", lambda a: a / 2),
                 ("{b} = 1/{a}", lambda a: 1 / a)]
        have = {r["text"] for r in self.data["constant_relations"]}
        for ca in good:
            for cb in good:
                if ca is cb:
                    continue
                for pat, fn in forms:
                    tol = min(4 * max(ca["se_rel"], cb["se_rel"]) + 1e-6, 1e-3)
                    if abs(fn(ca["value"]) - cb["value"]) / cb["value"] <= tol:
                        txt = pat.format(a=fmt(ca["value"]), b=fmt(cb["value"]))
                        if txt not in have:
                            have.add(txt)
                            self.data["constant_relations"].append(dict(text=txt, t=self._now()))
                            self.note("constant-relation", f"two learned constants are related: {txt}")

    # ---- vocabulary / inventions / stages ---------------------------------------
    def remember_exponents(self, values):
        for x in values:
            if abs(x) > 4 or abs(x) < 1e-9:
                continue
            k = f"{x:g}"
            self.data["exponents"][k] = self.data["exponents"].get(k, 0) + 1

    def top_exponents(self, min_count=2, k=6):
        items = [(c, float(x)) for x, c in self.data["exponents"].items()
                 if c >= min_count and float(x) not in self.BASE_EXPS]
        return [x for _, x in sorted(items, reverse=True)[:k]]

    def remember_invention(self, name, exps, knobs, target, rough):
        for r in self.data["inventions"]:
            if r["exps"] == exps and set(r["knobs"]) == set(knobs) and r["target"] == target:
                r["count"] += 1
                r["last"] = self._now()
                return
        self.data["inventions"].append(dict(name=name, exps=exps, knobs=sorted(knobs),
                                            target=target, rough=float(rough), count=1,
                                            last=self._now()))
        self.note("discovery", f"invented variable {name} := {mono_text(exps)} "
                               f"(makes '{target}' a 1-D function)")

    def recall_inventions(self, knobs, outputs):
        return [r for r in self.data["inventions"]
                if set(r["knobs"]) == set(knobs) and r["target"] in outputs]

    def record_stage(self, name, seconds, accepted):
        s = self.data["stages"].setdefault(name, dict(runs=0, accepted=0, seconds=0.0))
        s["runs"] += 1
        s["accepted"] += int(accepted)
        s["seconds"] += float(seconds)

    def should_skip(self, name):
        """Only a stage that is EXPENSIVE (>= 1 s per run) and has produced nothing in 8+ runs is
        sampled every 5th run. Cheap stages are never skipped: a stage that only fires on certain
        kinds of data (periodic, exponential) looks useless until the data that needs it arrives."""
        s = self.data["stages"].get(name)
        if (s and s["runs"] >= 8 and s["accepted"] == 0 and s["seconds"] / s["runs"] >= 1.0
                and (self.data["runs"] % 5) != 0):
            return True
        return False

    def record_skill(self, family, ok, seconds, measurements, noise):
        s = self.data["skills"].setdefault(family, dict(attempts=0, successes=0, seconds=0.0,
                                                        measurements=0, failed_noise=[]))
        s["attempts"] += 1
        s["successes"] += int(ok)
        s["seconds"] += float(seconds)
        s["measurements"] += int(measurements)
        if not ok:
            s["failed_noise"] = (s["failed_noise"] + [round(float(noise), 4)])[-8:]
        if s["attempts"] >= 4 and s["successes"] / s["attempts"] < 0.75:
            self.note("weakness", f"finds '{family}' laws only {s['successes']}/{s['attempts']} times "
                                  f"(failed at noise {s['failed_noise']}) - treat such results with care")

    def record_run(self, **info):
        self.data["runs"] += 1
        self.data["history"].append(dict(t=self._now(), **_plain(info)))
        self.data["history"] = self.data["history"][-self.MAX_HISTORY:]

    # ---- human-readable --------------------------------------------------------
    def describe(self, notes=12):
        d = self.data
        L = [f"memory file: {self.path}   (runs: {d['runs']}, updated {d['updated']})",
             f"  laws remembered: {len(d['laws'])}   retired: {len(d['retired'])}   "
             f"falsified: {len(d['falsified'])}   constants: {len(d['constants'])} "
             f"({len(self.confirmed_constants())} confirmed)   inventions: {len(d['inventions'])}"]
        for r in sorted(d["laws"], key=lambda r: -r["confirmed"])[:15]:
            L.append(f"   [{r['id']}] {r['text']}   (confirmed {r['confirmed']}x, failed {r['failed']}x,"
                     f" err {r['err']:.2%})")
        for r in d["inventions"]:
            L.append(f"   invention {r['name']} := {mono_text(r['exps'])} for '{r['target']}' "
                     f"(seen {r['count']}x)")
        for r in d["constant_relations"]:
            L.append(f"   constant relation: {r['text']}")
        if d["stages"]:
            L.append("   stage stats: " + "; ".join(
                f"{k}: {v['accepted']} accepted / {v['runs']} runs / {v['seconds']:.1f}s"
                for k, v in d["stages"].items()))
        if d.get("skills"):
            L.append("   skills (own practice): " + "; ".join(
                f"{k} {v['successes']}/{v['attempts']}" for k, v in sorted(d["skills"].items())))
        L.append("  recent notes:")
        for n in d["notes"][-notes:]:
            L.append(f"   - [{n['kind']}] {n['text']}" + (f"  (x{n['count']})" if n["count"] > 1 else ""))
        return "\n".join(L)

    def forget(self, text):
        n0 = len(self.data["laws"])
        self.data["laws"] = [r for r in self.data["laws"] if text not in r["text"]]
        self.data["falsified"] = [r for r in self.data["falsified"] if text not in r["text"]]
        self.note("lesson", f"forgot laws matching '{text}' (user request)")
        return n0 - len(self.data["laws"])


def as_memory(m):
    if m is None or isinstance(m, Memory):
        return m
    return Memory(str(m))


# =============================================================================
# 12. DISCOVERER (pipeline)
# =============================================================================
class Discoverer:
    def __init__(self, bundle, cfg, known=(), name="", defs=(), extra_exps=(), memory=None):
        self.b, self.cfg, self.known, self.name = bundle, cfg, known, name
        self.memory = memory
        self.extra_exps = tuple(sorted(set(extra_exps) | (set(memory.top_exponents()) if memory else set())))
        self.pending, self.stage_log, self.memory_summary = [], {}, ""
        self.recalled_n = 0
        self.archive = Archive(known, bundle.variables)
        self.hints = []
        for d in defs:      # definitions of invented variables are bookkeeping, not discoveries
            e = Entry(d, "definition", dict(train=0, val=0, test=0, per_exp=[0], max_exp=0), 0, [], [])
            e.cls = "DEFINITION"
            self.archive.entries.append(e)
            self.archive.all.append(e)
            mv = Archive._mono_vec(d)
            if mv:
                self.archive.mono.append(mv)

    def run(self):
        b, cfg = self.b, self.cfg
        t0 = time.time()
        info = analyze(b, cfg)
        loggable = {v for v, i in info.items() if i["positive"] and i["varying"]}
        varying = [v for v, i in info.items() if i["varying"]]

        mem = self.memory
        recalled_lhs, fast = set(), False
        if mem:
            MEM_CONSTS[:] = mem.confirmed_constants()
            for rec, rel in mem.recall(b.variables):
                st = evaluate_relation(rel, b)
                ok = accept(st, cfg)
                jac = len(set(rec["context"]) & set(b.variables)) / max(1, len(set(rec["context"]) | set(b.variables)))
                self.pending.append((rec["id"], ok, jac))
                if not ok:
                    continue
                e = Entry(rel, "memory", st, rec.get("spread", 0.0),
                          [f"remembered law (confirmed {rec['confirmed']}x, first seen {rec['first_seen'][:10]})"], [])
                e.memory_id = rec["id"]
                self.archive.add_recalled(e)
                self.recalled_n += 1
                if rel.is_explicit():
                    recalled_lhs.add(rel.lhs_var())
            if self.recalled_n:
                say(cfg, f"\n[MEMORY] re-validated {self.recalled_n} remembered law(s) on this data "
                         f"in {time.time() - t0:.2f}s; readings already explained: {sorted(recalled_lhs)}")
            outs = [o for o in b.outputs if o in varying]
            fast = bool(outs) and all(o in recalled_lhs for o in outs)
        if fast:
            say(cfg, "[MEMORY] every reading is explained by a remembered law that still holds -> "
                     "skipping search")
        else:
            self._search(loggable, varying, recalled_lhs)

        for e in self.archive.all:
            if e.cls == "DEFINITION":
                continue
            e.dim = dim_status(e.rel, b)
            score_entry(e, b)
            if mem and not e.recalled:
                f = mem.is_falsified(e.rel, b.variables)
                if f:
                    e.warn.append(f"previously FALSIFIED on out-of-range data ({f['reason']}); "
                                  f"treat with suspicion")
                    e.scores["interestingness"] *= 0.5
        say(cfg, f"\n[done in {time.time() - t0:.1f}s]")
        return self.archive


    def _stage(self, name, t0, accepted):
        self.stage_log[name] = (time.time() - t0, int(accepted))

    def _search(self, loggable, varying, recalled_lhs):
        b, cfg, mem = self.b, self.cfg, self.memory
        say(cfg, "\n[POWER LAWS / INVARIANTS] log-space null-space search"
                 + (" + dimensionless groups" if b.dims else ""))
        t_ = time.time()
        pl = PowerLawFinder(b, cfg, loggable)
        pool = pl.run()
        self.hints = pl.hints
        self._stage("power-law", t_, len(pool))
        targets = [o for o in b.outputs if o in loggable] or sorted(loggable)
        targets = [t for t in targets if t not in recalled_lhs]
        if mem and mem.should_skip("semilog"):
            say(cfg, "\n[SEMI-LOG LAWS] skipped: this stage has never paid off in memory "
                     "(sampled every 5th run)")
            mem.note("lesson", "semilog stage skipped: 0 accepted laws in 8+ earlier runs "
                               "(re-sampled every 5th run)")
        else:
            say(cfg, "\n[SEMI-LOG LAWS] exponential / Arrhenius / Gaussian dependence")
            t_ = time.time()
            sl_pool = SemiLogFinder(b, cfg, targets, loggable, set(varying)).run()
            self._stage("semilog", t_, len(sl_pool))
            pool += sl_pool
        # Per-variable noise floor: a relation far worse than the best law that
        # involves the same variables is an approximation of a true law, not a law.
        floor = {}
        for e in pool:
            for v in e.rel.variables():
                floor[v] = min(floor.get(v, 1e9), e.stats["train"])
        pool = [e for e in pool
                if e.stats["train"] <= max(3 * max(floor[v] for v in e.rel.variables()), 0.004)]
        pool.sort(key=lambda e: (round(e.rel.complexity(), 1), e.stats["train"]))
        say(cfg, f"  {len(pool)} accepted candidates (power-law + semi-log)")
        for e in pool:
            self.archive.classify(e)

        eliminated = set()
        for e in self.archive.entries:
            if e.cls != "DEFINITION" and e.rel.is_explicit():
                eliminated.add(e.rel.lhs_var())
        basis = [v for v in varying if v not in eliminated]
        say(cfg, f"\n[BASIS REDUCTION] variables explained by laws: "
                 f"{sorted(eliminated) or 'none'}; independent basis: {basis}")

        # noise floor: what the cleanest accepted laws achieve. A sum-of-terms
        # relation that is far worse than that is an approximation, not a law.
        floor = min([e.stats["train"] for e in self.archive.entries], default=None)
        cap = min(cfg.accept_err, max(3 * floor, 0.004)) if floor is not None else None
        trig = []
        if mem and mem.should_skip("periodic"):
            say(cfg, "\n[PERIODIC SCAN] skipped: never paid off in memory (sampled every 5th run)")
        else:
            t_ = time.time()
            trig = PeriodicScan(b, cfg, [o for o in (b.outputs or varying) if o not in recalled_lhs],
                                varying).run()
            self._stage("periodic", t_, len(trig))
            if trig:
                say(cfg, "\n[PERIODIC SCAN] oscillation found: "
                         + ", ".join(f"{v}: w={fmt(w)} (period {2 * math.pi / w:.4g})"
                                     + (f", damping {fmt(k)}" if k else "") for v, w, k in trig))
        imp = ImplicitFinder(b, cfg, basis, cap, self.extra_exps, trig=trig)
        say(cfg, f"\n[IMPLICIT SEARCH] transform library: {len(imp.feats)} features "
                 f"(powers, ratios, products, logs) over {basis}; beam search, "
                 f"<= {cfg.max_terms} terms")
        t_ = time.time()
        pool2 = imp.run()
        self._stage("implicit", t_, len(pool2))
        pool2.sort(key=lambda e: (round(e.rel.complexity(), 1), e.stats["train"]))
        if b.dims:        # units known: a sum of terms with different dimensions is not physics
            bad = [e for e in pool2 if dim_status(e.rel, b).startswith("INCONSISTENT")]
            if bad:
                say(cfg, f"  [UNITS] rejected {len(bad)} relation(s) whose terms have different dimensions")
                if mem:
                    mem.note("lesson", f"rejected dimensionally inconsistent relation(s), e.g. {bad[0].rel.text()}")
                pool2 = [e for e in pool2 if e not in bad]
        say(cfg, f"  {len(pool2)} accepted candidates")
        for e in pool2:
            if not e.rel.lhs.exps:      # "1 = f(...)" over exactly the variables of an accepted law
                vs = e.rel.variables()  # is a noise-exploiting restatement, not a new law
                if any(a.cls in ("NOVEL", "KNOWN") and a.rel.variables() == vs
                       for a in self.archive.entries):
                    continue
            self.archive.classify(e)

        # refinement: an explicit law that is far worse than the noise level may only be an
        # approximation (e.g. a power law fitted to a log curve). Its variable was removed from the
        # search, so look for a better relation for it and keep that only if >= 2x more accurate.
        for e in list(self.archive.entries):
            if (e.cls not in ("NOVEL", "KNOWN") or e.recalled or e.source not in ("power-law", "semilog")
                    or not e.rel.is_explicit() or e.stats["train"] <= 0.005):
                continue
            v = e.rel.lhs_var()
            t_ = time.time()
            better = [r for r in ImplicitFinder(b, cfg, list(varying), None, self.extra_exps,
                                                trig=trig).run(lhs_vars=[v])
                      if r.stats["train"] < 0.5 * e.stats["train"] and v in r.rel.variables()]
            self._stage("refine", t_, len(better))
            if better:
                r = min(better, key=lambda x: (x.rel.complexity(), x.stats["train"]))
                say(cfg, f"[REFINE] {e.rel.text()} (error {e.stats['train']:.2%}) was only an approximation; "
                         f"replaced by {r.rel.text()} (error {r.stats['train']:.2%})")
                if mem:
                    mem.note("lesson", f"approximation {e.rel.text()} at {e.stats['train']:.2%} error was "
                                       f"superseded by {r.rel.text()} at {r.stats['train']:.2%}")
                self.archive.remove(e)
                self.archive.classify(r)

        # chained search: a reading may be built from OTHER readings (heat from rate and
        # a knob). Basis reduction hid those, so retry unexplained readings with
        # every variable available as a building block.
        seen = set()
        for e in self.archive.entries:
            if e.cls != "DEFINITION":
                seen |= e.rel.variables()
        missing = [o for o in (b.outputs or varying) if o in varying and o not in seen]
        if missing and eliminated:
            say(cfg, f"\n[CHAINED SEARCH] still unexplained: {missing}; retrying with all "
                     f"variables (including already-explained readings) as building blocks")
            imp2 = ImplicitFinder(b, cfg, list(varying), cap, self.extra_exps, trig=trig)
            t_ = time.time()
            pool3 = imp2.run(lhs_vars=missing)
            self._stage("chained", t_, len(pool3))
            pool3.sort(key=lambda e: (round(e.rel.complexity(), 1), e.stats["train"]))
            say(cfg, f"  {len(pool3)} accepted candidates")
            for e in pool3:
                e.source = "implicit (chained)"
                self.archive.classify(e)

    def commit_memory(self, context=None, notes=()):
        """Write what this run learned into the memory file. Call once, at the end."""
        mem, b = self.memory, self.b
        if mem is None:
            return ""
        ctx = dict(context or {})
        ranges = {v: [float(b.full.cols[v].min()), float(b.full.cols[v].max())] for v in b.variables}
        mem.session.update(recalled=0, new=0, failed=0)
        for rid, ok, jac in self.pending:
            mem.apply_validation(rid, ok, jac, self.name or "data")
        consts, exps = [], []
        for e in self.archive.entries:
            if e.cls == "DEFINITION":
                continue
            if e.cls in ("NOVEL", "KNOWN") and not e.recalled:
                mem.remember_law(e, b.variables, ranges, self.name or "data")
            if e.cls in ("NOVEL", "KNOWN", "DERIVED"):
                for (c, f), ci in zip(e.rel.rhs, e.rel.cinfo):
                    if ci is not None and ci.origin in ("learned", "fitted") and ci.value:
                        consts.append((ci.value, abs(ci.se / ci.value) if ci.value else 1.0, e.rel.text()))
                    if f.kind in ("m", "em"):
                        exps += list(f.exps.values())
        mem.remember_constants(consts)
        mem.remember_exponents(exps)
        for st, (sec, acc) in self.stage_log.items():
            mem.record_stage(st, sec, acc)
        for h in self.hints[:3]:
            mem.note("hint", f"{h[0]} ~ C_experiment * {mono_text(h[1])}; constants differ by experiment "
                             f"{h[2]} (hidden variable?)")
        for n in notes:
            mem.note("run", n)
        novel = sum(1 for e in self.archive.entries if e.cls == "NOVEL" and not e.recalled)
        total_t = sum(v[0] for v in self.stage_log.values())
        mem.record_run(dataset=self.name or "data", variables=b.variables, experiments=b.full.n_exp,
                       rows=b.full.n, recalled=self.recalled_n, new=novel,
                       search_seconds=round(total_t, 2), **{k: v for k, v in ctx.items()
                                                            if isinstance(v, (int, float, str, list))})
        mem.note("run", f"run {mem.data['runs']} on '{self.name or 'data'}': {self.recalled_n} law(s) recalled, "
                        f"{novel} new, search {total_t:.1f}s")
        mem.save()
        self.memory_summary = (f"memory '{mem.path}': {self.recalled_n} recalled, {novel} new law(s) learned; "
                               f"{len(mem.data['laws'])} laws, {len(mem.data['notes'])} notes stored")
        return self.memory_summary

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
    def __init__(self, bundle, cfg, controllable, augment=None, extra=(), bounds=None):
        self.b, self.cfg, self.ctrl = bundle, cfg, list(controllable)
        self.bounds = bounds or {}
        self.augment, self.extra = augment, set(extra)
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

    def propose(self, archive, k=6, replicates=10, only=None):
        cands = []
        self.last_sep = []
        for e in ([only] if only is not None else archive.ranked()):
            r = e.rel
            if e.cls in ("NOVEL", "KNOWN") and r.is_explicit():
                tgt = r.lhs_var()
                used = r.variables() - {tgt}
                if tgt not in self.ctrl and used <= (set(self.ctrl) | self.extra):
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
            if v in self.bounds:              # never design beyond 3x the instrument's real range
                lo, hi = max(lo, self.bounds[v][0] / 3), min(hi, self.bounds[v][1] * 3)
            pts[v] = np.exp(self.rng.uniform(math.log(lo), math.log(hi), N))
        if self.augment:
            pts = self.augment(pts)
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
        self.last_sep = [float(sep[c]) for c in chosen for _ in range(replicates)]
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


def run_hidden_law_challenge(law, cfg, active=True, known=()):
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




# #############################################################################
# PART 2: AUTONOMOUS AGENT
# #############################################################################
# =============================================================================
# 1. LAB (the only thing that knows the world) AND DERIVED VARIABLES
# =============================================================================
class Lab:
    """knobs: {name: (lo, hi)}.  fn(points)->dict of ALL columns (knobs + readings)."""

    def __init__(self, knobs, fn):
        self.knobs, self.fn, self.count = dict(knobs), fn, 0

    def measure(self, pts):
        self.count += len(next(iter(pts.values())))
        return self.fn(pts)

    def sample(self, rng, n, a=0.0, b=1.0, stretch=1.0):
        out = {}
        for k, (lo, hi) in self.knobs.items():
            l, h = math.log(lo), math.log(hi)
            lo2 = l + (h - l) * a - math.log(stretch)
            hi2 = l + (h - l) * b + math.log(stretch)
            out[k] = np.exp(rng.uniform(lo2, hi2, n))
        return out

    def center(self):
        return {k: math.sqrt(lo * hi) for k, (lo, hi) in self.knobs.items()}


class DerivedVar:
    def __init__(self, name, exps):
        self.name, self.exps = name, dict(exps)

    def apply(self, cols):
        out = np.ones(len(next(iter(cols.values()))))
        for v, e in self.exps.items():
            out = out * np.power(cols[v], e)
        return out

    def text(self):
        return f"{self.name} := {mono_text(self.exps)}"

    def relation(self):
        return Relation(Feature.mono({self.name: 1}), [(1.0, Feature.mono(self.exps))])


def expand_terms(rel, derived):
    """Substitute invented variables by their definitions (for scoring / display)."""
    dmap = {d.name: d for d in derived}
    out = {}

    def add(c, f):
        k = f.key()
        if k in out:
            out[k] = (out[k][0] + c, f)
        else:
            out[k] = (c, f)

    for c, f in rel.terms():
        if f.kind == "m":
            exps = {}
            for v, e in f.exps.items():
                if v in dmap:
                    for w, pw in dmap[v].exps.items():
                        exps[w] = exps.get(w, 0) + e * pw
                else:
                    exps[v] = exps.get(v, 0) + e
            add(c, Feature.mono(exps))
        elif f.var in dmap:
            for w, pw in dmap[f.var].exps.items():
                add(c * pw, Feature.log(w))
        else:
            add(c, f)
    return [(c, f) for c, f in out.values() if abs(c) > 1e-12]


# =============================================================================
# 2. VARIABLE INVENTION: single-index search
#    Find u = prod knob^p (integer exponent vector, direction only) such that a
#    measured output is a smooth 1-D function of u. Sorting rows by u and
#    measuring local roughness is invariant to any monotone f, so f need not be
#    known, a power law, or even simple.
# =============================================================================
def roughness(ulog, ylog, w=7):
    order = np.argsort(ulog, kind="stable")
    y = ylog[order]
    win = np.lib.stride_tricks.sliding_window_view(y, w)
    med = np.median(win, axis=1)
    return float(np.median(np.abs(y[w // 2: w // 2 + len(med)] - med)))


def index_search(cols, inputs, target, thr=0.012, ratio=0.35, max_support=3):
    if not np.all(cols[target] > 0):
        return None
    y = np.log(cols[target])
    logs = {v: np.log(cols[v]) for v in inputs if np.all(cols[v] > 0)}
    ins = list(logs)
    if len(ins) < 2:
        return None
    base = min(roughness(logs[v], y) for v in ins)
    results = []
    for size in range(2, min(max_support, len(ins)) + 1):
        for sub in itertools.combinations(ins, size):
            for p in itertools.product(range(-3, 4), repeat=size):
                if 0 in p or p[0] < 0 or reduce(math.gcd, map(abs, p)) != 1:
                    continue
                u = sum(pi * logs[v] for pi, v in zip(p, sub))
                results.append((roughness(u, y), size, sum(map(abs, p)), sub, p))
    if not results:
        return None
    rmin = min(r[0] for r in results)
    if rmin > thr or rmin > ratio * base:
        return None
    good = [r for r in results if r[0] <= 1.3 * rmin]
    r, size, sa, sub, p = min(good, key=lambda t: (t[1], t[2], t[0]))
    return dict(exps=dict(zip(sub, p)), rough=r, base=base)


# =============================================================================
# 3. THE AGENT
# =============================================================================
class Agent:
    def __init__(self, lab, cfg, rng, n0=200, max_rounds=4, quiet_cfg=True, memory=None, name="agent"):
        self.lab, self.cfg, self.rng = lab, cfg, rng
        self.memory, self.name = memory, name
        self.events = []
        self.n0, self.max_rounds = n0, max_rounds
        self.qcfg = Config(**{**cfg.__dict__, "verbose": False})
        self.raw, self.derived, self.outputs = [], [], set()
        self.extra_exps = set()
        self.eng, self.bundle = None, None

    def say(self, *a):
        print(*a)

    # ---- data handling --------------------------------------------------------
    def augment(self, cols):
        cols = dict(cols)
        for d in self.derived:
            cols[d.name] = d.apply(cols)
        return cols

    def experiments(self):
        return [Experiment(n, self.augment(c)) for n, c in self.raw]

    def measure(self, name, pts):
        cols = self.lab.measure(pts)
        self.outputs |= set(cols) - set(self.lab.knobs)
        self.raw.append((name, cols))
        return cols

    # ---- 1. explore ------------------------------------------------------------
    def explore(self):
        L, rng = self.lab, self.rng
        self.measure("explore_full", L.sample(rng, self.n0))
        self.measure("explore_high", L.sample(rng, self.n0, 0.5, 1.0))
        self.measure("explore_low", L.sample(rng, self.n0, 0.0, 0.5))
        c = L.center()
        parts = []
        for k in L.knobs:
            sw = {j: np.full(25, c[j]) for j in L.knobs}
            lo, hi = L.knobs[k]
            sw[k] = np.exp(np.linspace(math.log(lo), math.log(hi), 25))
            parts.append(sw)
        pts = {j: np.concatenate([p[j] for p in parts]) for j in L.knobs}
        self.measure("one_factor_sweeps", pts)
        self.say(f"[EXPLORE] designed {len(self.raw)} experiments "
                 f"({self.lab.count} measurements): full range, high regime, low regime, "
                 f"one-factor-at-a-time sweeps over knobs {list(L.knobs)}")

    # ---- 2. discover -----------------------------------------------------------
    def discover(self):
        self.bundle = DataBundle(self.experiments(), self.qcfg, outputs=sorted(self.outputs))
        eng = Discoverer(self.bundle, self.qcfg, name=self.name,
                         defs=[d.relation() for d in self.derived],
                         extra_exps=sorted(self.extra_exps), memory=self.memory)
        eng.run()
        self.eng = eng
        # grammar growth: exponents seen in accepted laws join the search vocabulary
        for e in eng.archive.entries:
            if e.cls == "DEFINITION":
                continue
            for _, f in e.rel.rhs:
                if f.kind == "m":
                    for x in f.exps.values():
                        if abs(x) <= 4 and abs(x) > 1e-9:
                            self.extra_exps.add(round(x, 3))
        return eng

    def explained(self, eng):
        """A reading is explained only if some hypothesis ties it, directly or
        through other explained readings, back to the knobs."""
        grounded = set(self.lab.knobs) | {d.name for d in self.derived}
        rels = [e.rel.variables() for e in eng.archive.entries if e.cls != "DEFINITION"]
        changed = True
        while changed:
            changed = False
            for vs in rels:
                if vs & grounded and not vs <= grounded:
                    # a relation that touches grounded vars grounds the rest only
                    # if it is a function of grounded vars plus one unknown
                    unknown = vs - grounded
                    if len(unknown) == 1:
                        grounded |= unknown
                        changed = True
                elif len(vs - grounded) == 1 and len(vs) > 1 and vs & grounded:
                    pass
        return grounded

    # ---- memory: re-adopt earlier inventions after a cheap validity check -----------
    def _free_name(self):
        used = {d.name for d in self.derived}
        k = 1
        while f"u{k}" in used:
            k += 1
        return f"u{k}"

    def recall_inventions(self):
        mem = self.memory
        if not mem:
            return
        cols = {k: np.concatenate([c[k] for _, c in self.raw]) for k in self.raw[0][1]}
        for rec in mem.recall_inventions(list(self.lab.knobs), self.outputs):
            if rec["name"] in {d.name for d in self.derived} or \
               any(d.exps == rec["exps"] for d in self.derived):
                continue
            t = rec["target"]
            if t not in cols or not np.all(cols[t] > 0):
                continue
            u = sum(p * np.log(cols[v]) for v, p in rec["exps"].items())
            r = roughness(u, np.log(cols[t]))
            if r <= 0.012:
                d = DerivedVar(rec["name"], rec["exps"])
                self.derived.append(d)
                rec["count"] += 1
                self.say(f"[MEMORY] re-adopted invented variable {d.text()} "
                         f"(roughness of '{t}' vs {d.name}: {r:.4f}) - no search needed")
            else:
                mem.note("lesson", f"remembered invention {rec['name']} := {mono_text(rec['exps'])} "
                                   f"no longer makes '{t}' a 1-D function (roughness {r:.3f})")

    # ---- 3. invent -------------------------------------------------------------
    def invent(self, eng):
        seen = self.explained(eng)
        todo = sorted(o for o in self.outputs if o not in seen)
        made = False
        full = self.bundle.full.cols
        knobs = list(self.lab.knobs)
        for t in todo:
            self.say(f"[INVENT] output '{t}' is not explained by any hypothesis -> "
                     f"searching for a variable u (product of knob powers) such that "
                     f"{t} = f(u) for some unknown f")
            res = index_search(full, knobs, t)
            if res is None:
                self.say(f"         no single-index structure found for '{t}'")
                continue
            same = [d for d in self.derived if d.exps == res["exps"]]
            if same:
                continue
            name = self._free_name()
            d = DerivedVar(name, res["exps"])
            self.derived.append(d)
            if self.memory:
                self.memory.remember_invention(name, res["exps"], knobs, t, res["rough"])
            made = True
            self.say(f"         INVENTED  {d.text()}   (roughness of {t} vs {name}: "
                     f"{res['rough']:.4f}; best single knob: {res['base']:.3f})")
        return made

    # ---- 4/5. falsify and design ---------------------------------------------
    def falsify(self, eng, tag):
        pts = self.lab.sample(self.rng, 150, stretch=3.0)
        cols = self.measure(f"falsify_{tag}", pts)
        st = stack_from_cols(self.augment(cols))
        bad = []
        self.say(f"[FALSIFY] 150 fresh measurements up to 3x outside the explored range")
        for e in eng.archive.ranked():
            err = median_err(e.rel, st)
            lim = max(3 * e.stats["train"], 0.02)
            ok = err <= lim
            self.say(f"   {'survives' if ok else 'FALSIFIED'}  err {err:.4f} (limit {lim:.4f})  "
                     f"{e.rel.text()}")
            if not ok:
                bad.append(e)
                if self.memory:
                    self.memory.falsify(e.rel, f"error {err:.3f} on measurements 3x outside the range "
                                               f"(limit {lim:.3f})", list(st.cols))
                if e.recalled:
                    self.events.append(f"remembered law failed extrapolation: {e.rel.text()}")
        return bad

    def design(self, eng, tag):
        d = ActiveDesigner(self.bundle, self.qcfg, list(self.lab.knobs), self.augment,
                           [x.name for x in self.derived], bounds=self.lab.knobs)
        prop = d.propose(eng.archive, k=6, replicates=10)
        if prop:
            sel, _ = prop
            self.measure(f"active_{tag}", sel)
            return True
        return False

    # ---- main loop -------------------------------------------------------------
    def run(self):
        self.explore()
        self.recall_inventions()
        prev = None
        for r in range(self.max_rounds):
            self.say(f"\n--- round {r} " + "-" * 60)
            eng = self.discover()
            for _ in range(3):                       # chained invention
                if self.invent(eng):
                    eng = self.discover()
                else:
                    break
            sig = sorted(e.rel.text() for e in eng.archive.entries if e.cls != "DEFINITION")
            self.say(f"[DISCOVER] {len(sig)} hypotheses in archive; "
                     f"{len(self.derived)} invented variable(s); "
                     f"{self.lab.count} measurements so far")
            bad = self.falsify(eng, r)
            ents = [e for e in eng.archive.entries if e.cls != "DEFINITION"]
            if not bad and ents and all(e.recalled for e in ents):
                self.say("[STOP] every hypothesis was already in memory and survived fresh "
                         "out-of-range measurements - nothing new to learn")
                break
            if not bad and sig == prev:
                self.say("[STOP] hypotheses unchanged and none falsified")
                break
            if self.memory:
                self.memory.save()
            prev = sig
            if not bad:
                self.design(eng, r)
        self.eng = self.discover()
        used = set()
        for e in self.eng.archive.entries:
            if e.cls != "DEFINITION":
                used |= e.rel.variables()
        self.derived = [d for d in self.derived if d.name in used]
        if self.memory:
            notes = [f"agent '{self.name}': {self.lab.count} lab measurements, "
                     f"{len(self.derived)} invented variable(s)"] + self.events
            self.eng.commit_memory(context=dict(measurements=self.lab.count,
                                                knobs=list(self.lab.knobs)), notes=notes)
        return self.eng


# =============================================================================
# 4. LATENT (per-experiment) VARIABLE INVENTION
# =============================================================================
def latent_analysis(bundle, cfg, hidden=None):
    cfg_q = Config(**{**cfg.__dict__, "verbose": False})
    eng = Discoverer(bundle, cfg_q, name="latent")
    eng.run()
    eng.report(top=5)
    for lhs, q, consts in eng.hints[:3]:
        print("\n[LATENT VARIABLE INVENTED]")
        print(f"  {lhs} = h_e * {mono_text(q)}  holds INSIDE every experiment, but the "
              f"constant differs between experiments:")
        h = np.array(consts)
        print(f"  h_e = {[round(x, 4) for x in consts]}   -> an unmeasured per-experiment factor")
        print(f"  relative to experiment 1 (h_e/h_1): {[round(float(x), 4) for x in h / h[0]]}")
        errs = []
        for e, ex in enumerate(bundle.experiments):
            r = Relation(Feature.mono({lhs: 1}), [(h[e], Feature.mono(q))])
            errs.append(median_err(r, stack_from_cols(ex.cols)))
        print(f"  per-experiment fit error with latent factor: {[round(x, 4) for x in errs]}")
        if hidden is not None:
            k = hidden["power"]
            truth = (np.array(hidden["values"]) / hidden["values"][0]) ** k
            print(f"  TRUTH (hidden quantity ^ {k}, relative to exp 1): "
                  f"{[round(float(x), 4) for x in truth]}")
            print("  => latent factor recovers the hidden per-experiment quantity up to scale"
                  if np.allclose(h / h[0], truth, rtol=0.01) else "  => mismatch")
    return eng


# =============================================================================
# 5. CHALLENGES (hidden laws live only inside the simulated lab)
# =============================================================================
def _u(p):
    return p["x"] * p["y"] / p["z"] ** 2


def _composite(p):
    u = _u(p)
    return 2.2 * u ** 1.5 + 4.0 * u


def _noisy(rng, y, noise=0.003, outlier=0.01):
    y = y * (1 + noise * rng.standard_normal(len(y)))
    bad = rng.random(len(y)) < outlier
    return np.where(bad, y * rng.uniform(1.3, 2.0, len(y)), y)


def make_lab(name, rng):
    knobs = {"x": (0.5, 20), "y": (0.5, 15), "z": (0.5, 10)}

    def wrap(outputs):
        def fn(p):
            cols = dict(p)
            for k, f in outputs.items():
                cols[k] = _noisy(rng, f(p))
            return cols
        return fn

    M = lambda c, **e: (c, Feature.mono(e))
    if name == "composite":
        return (Lab(knobs, wrap({"Y": _composite, "W": lambda p: 0.8 * _composite(p) ** 1.2})),
                [("Y = 2.2*u^1.5 + 4*u,  u = x*y/z^2",
                  [M(1, Y=1), M(-2.2, x=1.5, y=1.5, z=-3), M(-4, x=1, y=1, z=-2)]),
                 ("W = 0.8*Y^1.2", [M(1, W=1), M(-0.8, Y=1.2)])])
    if name == "logindex":
        return (Lab(knobs, wrap({"Y": lambda p: 30 + 3 * np.log(_u(p))})),
                [("Y = 30 + 3*log(x*y/z^2)",
                  [M(1, Y=1), M(-30), (-3, Feature.log("x")), (-3, Feature.log("y")),
                   (6, Feature.log("z"))])])
    if name == "decay":
        return (Lab({"x": (0.2, 12), "y": (0.5, 15), "z": (0.5, 10)},
                    wrap({"Y": lambda p: 5.0 * np.exp(-0.35 * p["x"]) * p["z"] ** 1.5})),
                [("Y = 5*z^1.5*exp(-0.35*x)",
                  [(1, Feature.mono({"Y": 1})),
                   (-5, Feature.em({"z": 1.5}, {"x": (1, -0.35)}))])])
    if name == "arrhenius":
        return (Lab({"T": (250, 600), "c": (0.1, 5)},
                    wrap({"R": lambda p: 3.0e4 * p["c"] ** 2 * np.exp(-2500.0 / p["T"])})),
                [("R = 30000*c^2*exp(-2500/T)",
                  [(1, Feature.mono({"R": 1})),
                   (-3.0e4, Feature.em({"c": 2}, {"T": (-1, -2500.0)}))])])
    if name == "wave":
        return (Lab({"x": (0.2, 8), "z": (0.5, 10)},
                    wrap({"P": lambda p: 5 + 3 * np.sin(2.1 * p["x"])})),
                [("P = 5 + 3*sin(2.1*x)",
                  [M(1, P=1), M(-5), (-3, Feature.trig("sin", "x", 2.1))])])
    if name == "damped":
        return (Lab({"x": (0.2, 8), "z": (0.5, 10)},
                    wrap({"S": lambda p: 4 + 3 * np.exp(-0.35 * (p["x"] - 0.2)) * np.sin(2.5 * p["x"] + 0.4)})),
                [("S = 4 + 3*exp(-0.35*(x-0.2))*sin(2.5*x + 0.4)",
                  [M(1, S=1), M(-4 - 0.0), (-3 * math.exp(0.07) * math.cos(0.4), Feature.trig("sin", "x", 2.5, 0.35)),
                   (-3 * math.exp(0.07) * math.sin(0.4), Feature.trig("cos", "x", 2.5, 0.35))])])
    if name == "wave2":
        ph = 0.7
        return (Lab({"x": (0.2, 8), "z": (0.5, 10)},
                    wrap({"Q": lambda p: 10 + 4 * np.sin(1.3 * p["x"] + ph) + 2 * p["z"]})),
                [("Q = 10 + 4*sin(1.3*x + 0.7) + 2*z",
                  [M(1, Q=1), M(-10), (-4 * math.cos(ph), Feature.trig("sin", "x", 1.3)),
                   (-4 * math.sin(ph), Feature.trig("cos", "x", 1.3)), M(-2, z=1)])])
    if name == "peak":
        return (Lab({"x": (0.2, 8), "z": (0.5, 10)},
                    wrap({"G": lambda p: 2.0 * np.exp(-(p["x"] - 3) ** 2 / 4)})),
                [("G = 2*exp(-(x-3)^2/4)",
                  [(1, Feature.mono({"G": 1})),
                   (-2 * math.exp(-2.25), Feature.em({}, {("x", 1): 1.5, ("x", 2): -0.25}))])])
    if name == "law1":
        return (Lab(knobs, wrap({"Y": lambda p: 4.7 * p["x"] ** 1.83 * p["z"]})),
                [("Y = 4.7*x^1.83*z", [M(1, Y=1), M(-4.7, x=1.83, z=1)])])
    if name == "law3":
        return (Lab({"x": (0.5, 20), "y": (0.5, 15), "z": (0.3, 10)},
                    wrap({"Y": lambda p: p["x"] * p["y"] / (p["z"] + 2)})),
                [("Y*(z+2) = x*y", [M(1, Y=1, z=1), M(2, Y=1), M(-1, x=1, y=1)])])
    if name == "suite":
        return (Lab(knobs, wrap({"A": lambda p: 2.75 * p["x"] * p["y"],
                                 "B": lambda p: 0.5 * p["x"] * p["y"] ** 2,
                                 "C": lambda p: 3 * np.sqrt(p["x"]) * p["z"],
                                 "D": lambda p: 4.2 * p["x"] ** 2 / p["y"],
                                 "E": lambda p: 1.7 * p["x"] * p["y"] + 2 * p["z"]})),
                [("A = 2.75*x*y", [M(1, A=1), M(-2.75, x=1, y=1)]),
                 ("B = 0.5*x*y^2", [M(1, B=1), M(-0.5, x=1, y=2)]),
                 ("C = 3*sqrt(x)*z", [M(1, C=1), M(-3, x=0.5, z=1)]),
                 ("D = 4.2*x^2/y", [M(1, D=1), M(-4.2, x=2, y=-1)]),
                 ("E = 1.7*x*y + 2*z", [M(1, E=1), M(-1.7, x=1, y=1), M(-2, z=1)])])
    if name == "pendulum":
        return (Lab({"L": (0.1, 5), "g": (1, 25), "m": (0.1, 10)},
                    wrap({"T": lambda p: 2 * math.pi * np.sqrt(p["L"] / p["g"])})),
                [("T = 2*pi*sqrt(L/g)", [M(1, T=1), M(-2 * math.pi, L=0.5, g=-0.5)])])
    raise KeyError(name)


CHALLENGES = ["law1", "law3", "suite", "pendulum", "composite", "logindex", "decay", "arrhenius", "peak", "wave", "wave2", "damped"]


def run_challenge(name, cfg, seed, memory=None):
    rng = np.random.default_rng(seed)
    lab, truths = make_lab(name, rng)
    print("\n" + "#" * 78)
    print(f"# AUTONOMOUS CHALLENGE: {name}")
    print("# The agent gets knob names + ranges only. No data, no formula, no target.")
    print("#" * 78)
    agent = Agent(lab, cfg, rng, memory=memory, name=name)
    eng = agent.run()
    print("\n[INVENTED VARIABLES]",
          "; ".join(d.text() for d in agent.derived) if agent.derived else "none needed")
    eng.cfg = cfg
    eng.report(top=14)

    fresh = lab.measure(lab.sample(np.random.default_rng(seed + 5), 400, stretch=3.0))
    fs = stack_from_cols(agent.augment(fresh))
    print("\n" + "-" * 78)
    print(f"SCOREBOARD (hidden truth vs discovery, fresh data 3x outside range; "
          f"{lab.count} lab measurements used)")
    print("-" * 78)
    ok_all = True
    ents = [e for e in eng.archive.all if e.cls != "DEFINITION"]
    for label, terms in truths:
        hit = [e for e in ents if terms_match(expand_terms(e.rel, agent.derived), terms)]
        if not hit and len(terms) == 2 and all(f.kind == "m" for _, f in terms):
            tm = Archive._mono_vec(Relation(terms[0][1], [(-terms[1][0], terms[1][1])]))
            if tm is not None and tm[0] and eng.archive.mono:
                cls, _ = eng.archive._mono_class(*tm)
                if cls in ("DERIVED", "REDUNDANT"):          # follows from archived laws
                    hit = [e for e in ents if e.rel.variables() & set(tm[0])][:1]
        if hit:
            m = sorted(hit, key=lambda e: e.cls == "REDUNDANT")[0]
            print(f"  TRUE  {label}\n  FOUND {m.rel.text()}   [{m.cls}]  "
                  f"fresh-data error {median_err(m.rel, fs):.4f}")
        else:
            ok_all = False
            print(f"  TRUE  {label}\n  FOUND no matching relation")
    print(f"  => {'ALL HIDDEN LAWS RECOVERED' if ok_all else 'some laws missed'}")
    return ok_all


def run_latent(cfg, seed):
    rng = np.random.default_rng(seed)
    g = [9.81, 3.71, 24.8, 1.62]
    exps = []
    for i, gi in enumerate(g):
        L = np.exp(rng.uniform(math.log(0.1), math.log(5), 200))
        m = np.exp(rng.uniform(math.log(0.1), math.log(10), 200))
        exps.append(Experiment(f"planet{i + 1}", {"L": L, "m": m,
                    "T": _noisy(rng, 2 * math.pi * np.sqrt(L / gi))}))
    print("\n" + "#" * 78)
    print("# LATENT-VARIABLE CHALLENGE: pendulum on 4 planets; gravity is NOT recorded")
    print("#" * 78)
    b = DataBundle(exps, cfg, outputs=["T"])
    latent_analysis(b, cfg, hidden=dict(values=g, power=-0.5))



# #############################################################################
# PART 3: PUBLIC API, FILE HANDLING, EXPORT, CLI
# #############################################################################
import csv
import importlib.util
import json
import os


def _read_csv_table(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    cols = {}
    for k in rows[0]:
        vals = [(r.get(k) or "").strip() for r in rows]
        parsed, bad = [], 0
        for v in vals:
            try:
                parsed.append(float(v) if v else np.nan)
            except ValueError:
                parsed.append(np.nan)
                bad += 1
        filled = sum(1 for v in vals if v)
        # numeric if >=80% of the filled cells parse (stray "n/a" cells become missing)
        if filled and bad <= 0.2 * filled:
            cols[k.strip()] = np.array(parsed)
        else:
            cols[k.strip()] = np.array(vals, dtype=object)
    return cols


def _table_from(obj):
    if isinstance(obj, str):
        return _read_csv_table(obj)
    if hasattr(obj, "to_dict") and hasattr(obj, "columns"):          # pandas DataFrame
        return {str(k): np.asarray(v) for k, v in obj.to_dict("list").items()}
    return {str(k): np.asarray(v) for k, v in dict(obj).items()}


def _clean(cols, name, min_rows=30):
    """Keep numeric columns that are mostly present; drop rows with missing/inf values."""
    num = {k: np.asarray(v, float) for k, v in cols.items() if np.asarray(v).dtype != object}
    if not num:
        raise ValueError(f"{name}: no numeric columns")
    n = len(next(iter(num.values())))
    keep = {k: v for k, v in num.items() if np.isfinite(v).mean() > 0.5}
    ok = np.ones(n, bool)
    for v in keep.values():
        ok &= np.isfinite(v)
    if ok.sum() < min_rows:
        raise ValueError(f"{name}: only {int(ok.sum())} usable rows (need >= {min_rows})")
    return {k: v[ok] for k, v in keep.items()}, int(n - ok.sum())


def to_experiments(data, experiment=None):
    """data: CSV path | list of CSV paths (one experiment each) | dict of arrays |
    DataFrame | list of dicts. `experiment`: column naming which experiment each row is in."""
    items = data if isinstance(data, (list, tuple)) else [data]
    exps, notes = [], []
    for idx, item in enumerate(items):
        table = _table_from(item)
        label = item if isinstance(item, str) else f"experiment{idx + 1}"
        ignored = [k for k, v in table.items() if np.asarray(v).dtype == object and k != experiment]
        if ignored:
            notes.append(f"{label}: ignored non-numeric column(s) {ignored}")
        if experiment and experiment in table:
            tags = np.asarray(table.pop(experiment))
            for tag in dict.fromkeys(tags.tolist()):
                m = tags == tag
                cols, dropped = _clean({k: np.asarray(v)[m] for k, v in table.items()}, f"{label}[{tag}]")
                exps.append(Experiment(f"{tag}", cols))
                if dropped:
                    notes.append(f"{label}[{tag}]: dropped {dropped} rows with missing values")
        else:
            cols, dropped = _clean(table, label)
            exps.append(Experiment(os.path.basename(label) if isinstance(item, str) else label, cols))
            if dropped:
                notes.append(f"{label}: dropped {dropped} rows with missing values")
    common = [k for k in exps[0].cols if all(k in e.cols for e in exps)]
    if len(common) < 2:
        raise ValueError("experiments share fewer than 2 numeric columns")
    exps = [Experiment(e.name, {k: e.cols[k] for k in common}) for e in exps]
    return exps, notes


class Result:
    """What `discover()` returns."""

    def __init__(self, engine, bundle, derived=(), notes=(), memory=None):
        self.engine, self.bundle, self.derived, self.notes = engine, bundle, list(derived), list(notes)
        self.memory = memory

    @property
    def entries(self):
        return self.engine.archive.ranked()

    def counts(self):
        return {c: sum(1 for e in self.engine.archive.all if e.cls == c)
                for c in ("NOVEL", "DERIVED", "KNOWN", "REDUNDANT")}

    def summary(self):
        n_exp = self.bundle.full.n_exp
        lines = [f"{len(self.entries)} independent relationship(s) found "
                 f"({self.counts()['REDUNDANT']} equivalent restatements collapsed), "
                 f"{n_exp} experiment(s), {self.bundle.full.n} rows."]
        for d in self.derived:
            lines.append(f"invented variable: {d.text()}")
        for i, e in enumerate(self.entries, 1):
            st = e.stats
            lines.append(f"{i}. {e.rel.text()}    [{e.cls}{' - remembered' if e.recalled else ''}; "
                         f"median error {st['test']:.2%} on held-out "
                         f"data; worst experiment {st['max_exp']:.2%}; interestingness "
                         f"{e.scores['interestingness']:.2f}"
                         + (f"; dimensions {e.dim}" if e.dim != "n/a" else "") + "]")
        for e in self.entries:
            for w in e.warn:
                lines.append(f"WARNING: {e.rel.text()}: {w}")
        lines += [f"note: {n}" for n in self.notes]
        if self.engine.memory_summary:
            lines.append(self.engine.memory_summary)
        return "\n".join(lines)

    def to_dict(self):
        out = []
        for e in self.entries:
            out.append(dict(
                relation=e.rel.text(), implicit_form=e.rel.implicit_text(), cls=e.cls,
                source=e.source, complexity=round(e.rel.complexity(), 2),
                scores={k: round(float(v), 4) for k, v in e.scores.items()},
                error=dict(train=e.stats["train"], val=e.stats["val"], test=e.stats["test"],
                           per_experiment=[float(x) for x in e.stats["per_exp"]]),
                dimensions=e.dim, parameters=list(e.params), remembered=bool(e.recalled),
                warnings=list(e.warn),
                constants=[c.describe() for c in e.rel.cinfo if c is not None],
                equivalent_forms=list(e.aliases)))
        return dict(counts=self.counts(), experiments=self.bundle.full.n_exp,
                    rows=int(self.bundle.full.n),
                    invented_variables=[d.text() for d in self.derived],
                    notes=self.notes, relations=out)

    def to_markdown(self):
        L = ["# Discovery report", "", self.summary().split("\n")[0], ""]
        if self.derived:
            L += ["**Invented variables**", ""] + [f"- `{d.text()}`" for d in self.derived] + [""]
        L += ["| # | Relation | Class | Test error | Worst experiment | Complexity | Interestingness |",
              "|---|---|---|---|---|---|---|"]
        for i, e in enumerate(self.entries, 1):
            L.append(f"| {i} | `{e.rel.text()}` | {e.cls} | {e.stats['test']:.3%} | "
                     f"{e.stats['max_exp']:.3%} | {e.rel.complexity():.1f} | "
                     f"{e.scores['interestingness']:.2f} |")
        L.append("")
        for i, e in enumerate(self.entries, 1):
            L += [f"## {i}. `{e.rel.text()}`", "",
                  f"- implicit form: `{e.rel.implicit_text()}`", f"- found by: {e.source}"]
            L += [f"- {p}" for p in e.params]
            L += [f"- constant: {c.describe()}" for c in e.rel.cinfo if c is not None and e.source == "power-law"]
            sc = e.scores
            L += [f"- scores: accuracy {sc['accuracy']:.2f}, simplicity {sc['simplicity']:.2f}, "
                  f"generality {sc['generality']:.2f}, robustness {sc['robustness']:.2f}, "
                  f"novelty {sc['novelty']:.2f}",
                  f"- median relative error: train {e.stats['train']:.4f}, val {e.stats['val']:.4f}, "
                  f"test {e.stats['test']:.4f}; per experiment "
                  f"{[round(float(x), 4) for x in e.stats['per_exp']]}"]
            if e.dim != "n/a":
                L.append(f"- dimensions: {e.dim}")
            if e.aliases:
                L.append(f"- equivalent forms collapsed: {len(e.aliases)}, e.g. `{e.aliases[0]}`")
            L.append("")
        return "\n".join(L)

    def save(self, path):
        with open(path, "w") as f:
            f.write(json.dumps(self.to_dict(), indent=2) if path.endswith(".json")
                    else self.to_markdown())
        return path

    def report(self, top=20):
        self.engine.report(top=top)


def discover(data, experiment=None, outputs=None, dims=None, base_dims=None, known=(),
             seed=42, verbose=False, priors=False, memory=None, name="data", domain=None,
             commit=True, **config):
    """Find relationships in observations.

        res = discover("run1.csv")                              # one experiment
        res = discover(["a.csv", "b.csv", "c.csv"])             # several experiments
        res = discover("all.csv", experiment="run_id")          # experiment column
        res = discover(df, outputs=["T"], dims={"L": (0,1,0), ...}, base_dims=["M","L","T"])
        print(res.summary());  res.save("report.md");  res.save("report.json")

    `outputs`: readings you want on the left-hand side (display preference only).
    `priors`: allow recognising pi, e, ... (default: constants are learned from data only).
    `memory`: path to a JSON file (or a Memory object). The engine re-validates what it learned
              in earlier runs instead of rediscovering it, then writes new laws, constants,
              falsified hypotheses and notes back to the file.
    """
    PRIORS["named"] = PRIORS["grammar"] = bool(priors)
    cfg = Config(seed=seed, verbose=verbose, **config)
    exps, notes = to_experiments(data, experiment)
    bundle = DataBundle(exps, cfg, dims, base_dims, outputs)
    mem = as_memory(memory)
    if mem is not None and domain is not None:
        mem.domain = domain
    eng = Discoverer(bundle, cfg, known, name, memory=mem)
    eng.run()
    if commit:
        eng.commit_memory()
    return Result(eng, bundle, notes=notes, memory=mem)


def explore(knobs, measure, seed=42, max_rounds=4, n0=200, priors=False, verbose=False,
            memory=None, name="lab", domain=None):
    """Autonomous discovery on a real or simulated instrument.

        knobs   = {"temperature": (250, 600), "concentration": (0.1, 5)}   # name: (min, max), > 0
        def measure(points): ...   # dict name->array of settings; return dict name->array of readings
        res = explore(knobs, measure)

    The agent chooses every setting itself, invents variables, falsifies its own
    hypotheses with out-of-range measurements, and stops when nothing changes."""
    PRIORS["named"] = PRIORS["grammar"] = bool(priors)
    rng = np.random.default_rng(seed)

    def fn(p):
        out = {k: np.asarray(v, float) for k, v in measure(dict(p)).items()}
        return {**{k: np.asarray(v, float) for k, v in p.items()}, **out}

    lab = Lab(knobs, fn)
    mem = as_memory(memory)
    if mem is not None and domain is not None:
        mem.domain = domain
    agent = Agent(lab, Config(seed=seed, verbose=verbose), rng, n0=n0, max_rounds=max_rounds,
                  memory=mem, name=name)
    eng = agent.run()
    res = Result(eng, agent.bundle, agent.derived,
                 [f"{lab.count} lab measurements used"], memory=mem)
    res.agent = agent
    return res


LAB_TEMPLATE = '''# my_lab.py  -  describe your instrument; the agent does the rest.
import numpy as np

KNOBS = {                      # settings the agent may choose: name: (min, max). All > 0.
    "temperature":   (250.0, 600.0),
    "concentration": (0.1, 5.0),
}

def measure(points):
    """points: dict name -> numpy array (one entry per requested measurement).
    Run your experiment at each setting and return a dict of numpy arrays, one per
    quantity you read out (e.g. {"rate": ..., "yield": ...}). Knobs are added for you."""
    T, c = points["temperature"], points["concentration"]
    rate = np.zeros_like(T)    # <- replace with a call to your hardware / simulator
    return {"rate": rate}
'''


def _load_lab(path):
    spec = importlib.util.spec_from_file_location("user_lab", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.KNOBS, mod.measure


# #############################################################################
# PART 4: AUTOPILOT - finds the data, works out its structure, and runs everything itself
#
#   python phy_discover.py            -> scans the current folder for data files, then
#                                        detects experiments (many files, or a label column),
#                                        drops index/duplicate columns, reads units from headers
#                                        like "g [m/s^2]", picks the noise tolerance by itself,
#                                        discovers laws, asks for the measurements it would need
#                                        next, and WRITES: phy_memory.json, phy_report.md,
#                                        phy_report.json, phy_next_experiments_*.csv
#   --watch                           -> keeps running and re-analyses when files appear/change
#   (no data found)                   -> self-directed practice: invents its own random worlds,
#                                        tries to discover them, records which kinds of law it
#                                        can/cannot find in phy_practice.json (self-assessment)
# #############################################################################
import contextlib as _ctx
import io as _io
import re as _re

DATA_EXT = (".csv", ".tsv", ".txt", ".json", ".npz")
OWN_PREFIX = "phy_"
LABEL_HINTS = ("run", "trial", "exp", "session", "batch", "group", "series", "sample", "condition",
               "setup", "campaign", "case", "dataset", "planet", "material", "id")
BASE_UNITS = ["M", "L", "T", "K", "A", "mol"]
_Z = (0, 0, 0, 0, 0, 0)


def _dim(M=0, L=0, T=0, K=0, A=0, N=0):
    return (M, L, T, K, A, N)


_UNITS = {
    "m": _dim(L=1), "s": _dim(T=1), "g": _dim(M=1), "kg": _dim(M=1), "K": _dim(K=1), "A": _dim(A=1),
    "mol": _dim(N=1), "Hz": _dim(T=-1), "N": _dim(1, 1, -2), "Pa": _dim(1, -1, -2),
    "J": _dim(1, 2, -2), "W": _dim(1, 2, -3), "V": _dim(1, 2, -3, 0, -1), "C": _dim(T=1, A=1),
    "min": _dim(T=1), "h": _dim(T=1), "hr": _dim(T=1), "day": _dim(T=1), "d": _dim(T=1),
    "L": _dim(L=3), "bar": _dim(1, -1, -2), "atm": _dim(1, -1, -2), "psi": _dim(1, -1, -2),
    "eV": _dim(1, 2, -2), "ohm": _dim(1, 2, -3, 0, -2), "Ω": _dim(1, 2, -3, 0, -2),
    "degC": _dim(K=1), "°C": _dim(K=1), "rad": _Z, "deg": _Z, "°": _Z, "%": _Z,
}
_PREFIX = set("pnumcdkMGT")


def _unit_dim(sym):
    if sym in _UNITS:
        return _UNITS[sym]
    if len(sym) > 1 and sym[0] in _PREFIX and sym[1:] in _UNITS and sym not in ("min", "mol"):
        return _UNITS[sym[1:]]
    return None


def parse_units(text):
    """'m/s^2' -> dimension vector over (M, L, T, K, A, mol); None if any symbol is unknown."""
    t = text.strip().replace("·", "*").replace("²", "^2").replace("³", "^3").replace("µ", "u")
    t = t.replace("⁻¹", "^-1").replace("**", "^")
    if t in ("", "1", "-", "dimensionless", "unitless", "none"):
        return _Z
    t = _re.sub(r"/\(([^)]*)\)", lambda m: "/" + "/".join(_re.split(r"[*\s]+", m.group(1).strip())), t)
    vec = np.zeros(6)
    used = 0
    for m in _re.finditer(r"([*/\s]*)([A-Za-z°ΩΩ%]+)(?:\^(-?\d+(?:\.\d+)?)|(-?\d+(?:\.\d+)?))?", t):
        ops, sym = m.group(1), m.group(2)
        ex = m.group(3) or m.group(4)
        d = _unit_dim(sym)
        if d is None:
            return None
        vec += (-1 if "/" in ops else 1) * (float(ex) if ex else 1.0) * np.array(d, float)
        used += 1
    return tuple(float(v) for v in vec) if used else None


def split_header(h):
    """'g [m/s^2]' -> ('g', 'm/s^2');  'T_K' -> ('T_K', None)."""
    m = _re.match(r"^(.*?)\s*[\[\(]([^\]\)]*)[\]\)]\s*$", str(h).strip())
    name, unit = (m.group(1), m.group(2)) if m else (str(h).strip(), None)
    name = _re.sub(r"\W+", "_", name).strip("_") or "col"
    if name[0].isdigit():
        name = "v_" + name
    return name, unit


def _coerce(values):
    """list of cells -> float array (stray 'n/a' = nan) or an object array of strings."""
    vals = [("" if v is None else str(v)).strip() for v in values]
    parsed, bad, filled = [], 0, 0
    for v in vals:
        if v == "":
            parsed.append(np.nan)
            continue
        filled += 1
        try:
            parsed.append(float(v.replace(",", ".") if _re.fullmatch(r"-?\d+,\d+", v) else v))
        except ValueError:
            parsed.append(np.nan)
            bad += 1
    if filled and bad <= 0.2 * filled:
        return np.array(parsed)
    return np.array(vals, dtype=object)


def load_any(path):
    """csv / tsv / txt (delimiter sniffed) / json (records or columns) / npz  ->  {column: array}."""
    low = path.lower()
    if low.endswith(".json"):
        with open(path) as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return {k: _coerce(v) for k, v in obj.items() if isinstance(v, list)}
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return {k: _coerce([r.get(k) for r in obj]) for k in obj[0]}
        raise ValueError("json must be a list of records or a dict of columns")
    if low.endswith(".npz"):
        d = np.load(path, allow_pickle=False)
        return {k: np.asarray(d[k], float) for k in d.files if d[k].ndim == 1}
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        text = f.read()
    sample = text[:4096]
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",;\t| ").delimiter
    except csv.Error:
        delim = "\t" if low.endswith(".tsv") else ","
    rows = [r for r in csv.reader(text.splitlines(), delimiter=delim) if any(c.strip() for c in r)]
    if len(rows) < 3:
        raise ValueError("fewer than 3 rows")
    head = rows[0]
    numeric_head = sum(1 for c in head if _coerce([c]).dtype != object)
    if numeric_head >= max(2, len(head) - 0):            # first row is data: no header
        head, body = [f"c{i + 1}" for i in range(len(head))], rows
    else:
        body = rows[1:]
    cols = {}
    for j, h in enumerate(head):
        cols[h.strip() or f"c{j + 1}"] = _coerce([r[j] if j < len(r) else "" for r in body])
    return cols


def scan_data_files(directory, recursive=False):
    out = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules")]
        for f in sorted(files):
            if f.startswith(OWN_PREFIX) or f.startswith(".") or not f.lower().endswith(DATA_EXT):
                continue
            out.append(os.path.join(root, f))
        if not recursive:
            break
    return out


def _is_index_like(name, col):
    c = np.asarray(col, float)
    if not np.all(np.isfinite(c)) or len(c) < 5:
        return False
    if name.lower() in ("index", "idx", "row", "rownum", "unnamed", "id", "n", "no"):
        if np.all(c == np.round(c)) and len(np.unique(c)) == len(c):
            return True
    d = np.diff(c)
    return bool(np.all(c == np.round(c)) and np.all(d == 1))


def _find_label_column(table):
    best = None
    for name, col in table.items():
        arr = np.asarray(col)
        hint = any(h in name.lower() for h in LABEL_HINTS)
        if arr.dtype == object:
            vals = [str(v) for v in arr.tolist() if str(v) != ""]
            score = 2
        else:
            a = arr.astype(float)
            if not np.all(np.isfinite(a)) or not np.all(a == np.round(a)) or not hint:
                continue
            vals = a.tolist()
            score = 1
        keys = list(dict.fromkeys(vals))
        if not (2 <= len(keys) <= 40):
            continue
        sizes = [sum(1 for v in vals if v == k) for k in keys]
        if min(sizes) < 30:
            continue
        cand = (score + (1 if hint else 0), -len(keys), name)
        if best is None or cand > best:
            best = cand
    return best[2] if best else None


def auto_prepare(group):
    """group: [(source, table)] with a shared schema.
    -> (experiments [dict], names, dims or None, decisions [str])"""
    dec, dims = [], {}
    tabs = []
    for src, table in group:
        ren, seen = {}, set()
        for h in table:
            name, unit = split_header(h)
            base, k = name, 2
            while name in seen:
                name, k = f"{base}_{k}", k + 1
            seen.add(name)
            ren[h] = name
            if unit is not None:
                d = parse_units(unit)
                if d is not None:
                    dims.setdefault(name, d)
                else:
                    dec.append(f"header '{h}': unit '{unit}' not recognised - dimensions skipped for it")
        tabs.append((src, {ren[h]: v for h, v in table.items()}))
    experiments, names = [], []
    if len(tabs) >= 2:
        for src, t in tabs:
            experiments.append(t)
            names.append(os.path.basename(src))
        dec.append(f"{len(tabs)} files share a schema -> treated as {len(tabs)} separate experiments")
    else:
        src, t = tabs[0]
        lab = _find_label_column(t)
        if lab:
            tags = np.asarray(t[lab])
            keys = list(dict.fromkeys(tags.tolist()))
            for k in keys:
                m = tags == k
                experiments.append({c: np.asarray(v)[m] for c, v in t.items() if c != lab})
                names.append(f"{lab}={k}")
            dec.append(f"column '{lab}' labels {len(keys)} groups of >=30 rows -> treated as "
                       f"{len(keys)} separate experiments")
        else:
            experiments.append(t)
            names.append(os.path.basename(src))
            dec.append("no experiment-label column found -> one experiment (generality cannot be verified)")
    # numeric columns shared by all experiments
    common = None
    for e in experiments:
        num = {c for c, v in e.items() if np.asarray(v).dtype != object}
        common = num if common is None else common & num
    order = [c for c in experiments[0] if c in common]
    dropped = [c for c in experiments[0] if c not in common]
    if dropped:
        dec.append(f"ignored non-numeric/unshared columns: {dropped}")
    idx = [c for c in order if all(_is_index_like(c, e[c]) for e in experiments)]
    if idx:
        dec.append(f"dropped index-like columns {idx}")
        order = [c for c in order if c not in idx]
    # exact scaled copies (b = k*a): trivial definitions, not discoveries
    pooled = {c: np.concatenate([np.asarray(e[c], float) for e in experiments]) for c in order}
    copies = {}
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            if a in copies or b in copies:
                continue
            A, B = pooled[a], pooled[b]
            ok = np.isfinite(A) & np.isfinite(B) & (np.abs(A) > 1e-300)
            if ok.sum() > 10 and np.all(np.abs(A[ok]) > 0):
                r = B[ok] / A[ok]
                if abs(np.mean(r)) > 0 and np.std(r) / abs(np.mean(r)) < 1e-7:
                    copies[b] = (a, float(np.mean(r)))
    for b, (a, k) in copies.items():
        dec.append(f"column '{b}' is an exact copy of '{a}' (x{k:.6g}) -> removed from the search")
    order = [c for c in order if c not in copies]
    experiments = [{c: np.asarray(e[c], float) for c in order} for e in experiments]
    dd = {c: d for c, d in dims.items() if c in order}
    if len(dd) >= 2:
        dec.append(f"units read from headers for {sorted(dd)} -> dimensional analysis on")
    else:
        dd = None
    return experiments, names, dd, dec


def _group_tables(tables):
    groups = []
    for src, t in tables:
        cols = {split_header(h)[0] for h, v in t.items() if np.asarray(v).dtype != object}
        for g in groups:
            if len(cols & g["cols"]) / max(1, len(cols | g["cols"])) >= 0.7:
                g["items"].append((src, t))
                g["cols"] &= cols
                break
        else:
            groups.append(dict(items=[(src, t)], cols=set(cols)))
    return groups


def noise_ladder():
    return (0.02, 0.05, 0.10)


def reorient_power_law(rel, w):
    """lhs = c * prod v^q  ->  the same law solved for variable w (None if not a pure power law)."""
    if len(rel.rhs) != 1 or rel.lhs.kind != "m" or rel.rhs[0][1].kind != "m" or not rel.is_explicit():
        return None
    c, f = rel.rhs[0]
    lv = rel.lhs_var()
    if c <= 0:
        return None
    if w == lv:
        return rel
    r = dict(f.exps)
    r[lv] = -1.0
    if w not in r or abs(r[w]) < 1e-12:
        return None
    new = {j: -r[j] / r[w] for j in r if j != w}
    return Relation(Feature.mono({w: 1}), [(c ** (-1.0 / r[w]), Feature.mono(new))])


def _most_sensitive(rel):
    """the variable of a pure power law with the largest exponent: the one measurement noise hurts most."""
    if len(rel.rhs) != 1 or rel.rhs[0][1].kind != "m" or not rel.is_explicit():
        return None
    r = dict(rel.rhs[0][1].exps)
    r[rel.lhs_var()] = 1.0
    return max(r, key=lambda v: (abs(r[v]), v == rel.lhs_var()))


def make_proposals(res, out_dir, tag, seed, k=8, limit=3):
    """Ask for the measurements that would best tell the surviving hypotheses apart."""
    arch, b = res.engine.archive, res.bundle
    files, done = [], set()
    for e in arch.ranked():
        if e.cls not in ("NOVEL", "KNOWN") or not e.rel.is_explicit() or len(files) >= limit:
            continue
        w = _most_sensitive(e.rel)
        rel2 = reorient_power_law(e.rel, w) if w else None
        ent = e
        if rel2 is not None and rel2 is not e.rel:      # propose: set the others, MEASURE the sensitive one
            ent = Entry(rel2, "reoriented", e.stats, e.spread, [], [reorient_power_law(c, w) or c
                                                                  for c in e.committee])
            ent.cls = "NOVEL"
        tgt = ent.rel.lhs_var()
        ctrl = sorted(ent.rel.variables() - {tgt})
        if tgt in done or not ctrl:
            continue
        done.add(tgt)
        d = ActiveDesigner(b, Config(seed=seed, verbose=False), ctrl)
        with _ctx.redirect_stdout(_io.StringIO()):
            prop = d.propose(arch, k=k, replicates=1, only=ent)
        if not prop:
            continue
        sel, _ = prop
        pred = ent.rel.rhs_value(sel)
        e = ent
        path = os.path.join(out_dir, f"phy_next_experiments_{tag}_{tgt}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(ctrl + [f"predicted_{tgt}", "separation_sigma"])
            for i in range(len(pred)):
                w.writerow([f"{sel[v][i]:.6g}" for v in ctrl] + [f"{pred[i]:.6g}",
                                                                  f"{d.last_sep[i]:.2f}"])
        files.append((path, e.rel.text()))
    return files


def self_assessment(path):
    if not os.path.exists(path):
        return []
    try:
        sk = json.load(open(path)).get("skills", {})
    except Exception:
        return []
    rows = []
    for fam, s in sorted(sk.items()):
        if s["attempts"] >= 3:
            rows.append((fam, s["successes"], s["attempts"]))
    return rows


def analyze_group(label, group, out_dir, mem, seed, tag, log):
    exps, names, dims, dec = auto_prepare(group)
    for d in dec:
        log(f"   [auto] {d}")
    chosen = None
    for tol in noise_ladder():
        res = discover(exps, dims=dims, base_dims=BASE_UNITS if dims else None, memory=mem,
                       name=label, seed=seed, commit=False, accept_err=tol, log_fit_tol=tol)
        found = [e for e in res.entries if e.cls in ("NOVEL", "KNOWN")]
        if found or tol == noise_ladder()[-1]:
            chosen = (tol, res)
            break
        msg = f"no relationship met the {tol:.0%} accuracy bar -> relaxing (noisy data?)"
        dec.append(msg)
        log(f"   [auto] {msg}")
    tol, res = chosen
    if tol > noise_ladder()[0] and any(e.cls in ("NOVEL", "KNOWN") for e in res.entries):
        dec.append(f"accuracy bar set to {tol:.0%} (data noisier than the default)")
        log(f"   [auto] accuracy bar set to {tol:.0%}")
    elif not any(e.cls in ("NOVEL", "KNOWN") for e in res.entries):
        dec.append("nothing held up even at the loosest accuracy bar (10%): treat the data as unrelated, "
                   "too noisy, or outside the search vocabulary")
        log("   [auto] nothing held up even at the loosest accuracy bar (10%)")
    res.engine.commit_memory(context=dict(tolerance=tol), notes=[d for d in dec][:5])
    res.notes = list(res.notes) + [f"autopilot: {d}" for d in dec]
    props = make_proposals(res, out_dir, tag, seed)
    return dict(label=label, decisions=dec, tolerance=tol, result=res, proposals=props, names=names)


def write_artifacts(out_dir, analyses, practice_rows, created):
    md = ["# phy_discover autopilot report", "",
          f"generated {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    js = []
    for a in analyses:
        res = a["result"]
        md += [f"## Dataset: {a['label']}", "", "### What the autopilot decided", ""]
        md += [f"- {d}" for d in a["decisions"]] or ["- nothing special"]
        md += [f"- accuracy bar used: {a['tolerance']:.0%}", ""]
        body = res.to_markdown().splitlines()
        md += ["##" + l if l.startswith("#") else l for l in body]
        if a["proposals"]:
            md += ["", "### Measurements it asks you to take next", ""]
            md += [f"- `{os.path.basename(p)}`  -  tests `{t}`" for p, t in a["proposals"]]
        md.append("")
        js.append(dict(dataset=a["label"], decisions=a["decisions"], tolerance=a["tolerance"],
                       result=res.to_dict(), next_experiments=[p for p, _ in a["proposals"]]))
    if practice_rows:
        md += ["## Self-assessment (from its own practice runs)", "",
               "| kind of law | found / tried |", "|---|---|"]
        md += [f"| {f} | {s}/{n} |" for f, s, n in practice_rows]
        md.append("")
    p1, p2 = os.path.join(out_dir, "phy_report.md"), os.path.join(out_dir, "phy_report.json")
    with open(p1, "w") as f:
        f.write("\n".join(md))
    with open(p2, "w") as f:
        json.dump(_plain(js), f, indent=1)
    created += [p1, p2]


# ---------------------------------------------------------------- practice ----
FAMILIES = ["power", "sum-of-products", "exp-decay", "arrhenius", "gaussian", "sinusoid",
            "damped-oscillation", "composite-index", "log-index"]


def _r2(x):
    return float(f"{x:.2g}")


def random_world(rng, family=None):
    """A random hidden law the program writes for itself (the engine is never shown it)."""
    fam = family or str(rng.choice(FAMILIES))
    U = lambda a, b: float(rng.uniform(a, b))
    out = str(rng.choice(["P", "Q", "R", "S", "W", "Y"]))
    knobs = {"x": (0.5, 20), "y": (0.5, 15), "z": (0.5, 10)}
    M = lambda c, **e: (c, Feature.mono(e))
    O = (1, Feature.mono({out: 1}))
    if fam == "power":
        p1, p2 = [float(rng.choice([-2, -1, -0.5, 0.5, 1, 1.5, 2, 3])) for _ in range(2)]
        if rng.random() < 0.4:
            p1 = round(U(0.4, 2.5), 2)
        C = _r2(10 ** U(-1, 2))
        fn = lambda p: C * p["x"] ** p1 * p["y"] ** p2
        truth = [O, M(-C, x=p1, y=p2)]
    elif fam == "sum-of-products":
        C1 = _r2(10 ** U(0, 1.3))
        C2 = _r2(C1 * U(2, 20))
        fn = lambda p: C1 * p["x"] * p["y"] + C2 * p["z"]
        truth = [O, M(-C1, x=1, y=1), M(-C2, z=1)]
    elif fam == "exp-decay":
        knobs = {"x": (0.2, 8), "z": (0.5, 10)}
        C, pz, k = _r2(10 ** U(0, 1.5)), float(rng.choice([0.5, 1, 1.5, 2])), _r2(U(0.15, 0.6))
        fn = lambda p: C * p["z"] ** pz * np.exp(-k * p["x"])
        truth = [O, (-C, Feature.em({"z": pz}, {("x", 1): -k}))]
    elif fam == "arrhenius":
        knobs = {"T": (250, 600), "c": (0.1, 5)}
        C, pc, E = _r2(10 ** U(3, 5)), float(rng.choice([1, 1.5, 2])), float(round(U(1000, 4000), -2))
        fn = lambda p: C * p["c"] ** pc * np.exp(-E / p["T"])
        truth = [O, (-C, Feature.em({"c": pc}, {("T", -1): -E}))]
    elif fam == "gaussian":
        knobs = {"x": (0.2, 8), "z": (0.5, 10)}
        A, mu, s = _r2(10 ** U(0, 1.5)), round(U(2, 6), 1), round(U(1, 2.2), 1)
        fn = lambda p: A * np.exp(-(p["x"] - mu) ** 2 / (2 * s * s))
        truth = [O, (-A * math.exp(-mu * mu / (2 * s * s)),
                     Feature.em({}, {("x", 1): mu / (s * s), ("x", 2): -1 / (2 * s * s)}))]
    elif fam in ("sinusoid", "damped-oscillation"):
        knobs = {"x": (0.2, 8), "z": (0.5, 10)}
        c0 = _r2(10 ** U(0.6, 1.2))
        a, w, ph = _r2(U(0.2, 0.6) * c0), round(U(1.2, 4), 1), round(U(0, 2), 1)
        k = _r2(U(0.15, 0.4)) if fam == "damped-oscillation" else 0.0
        env = (lambda x: np.exp(-k * x)) if k else (lambda x: 1.0)
        fn = lambda p: c0 + a * env(p["x"]) * np.sin(w * p["x"] + ph)
        truth = [O, M(-c0), (-a * math.cos(ph), Feature.trig("sin", "x", w, k)),
                 (-a * math.sin(ph), Feature.trig("cos", "x", w, k))]
    elif fam == "composite-index":
        C1, C2 = _r2(10 ** U(0, 1)), _r2(10 ** U(0, 1))
        u = lambda p: p["x"] * p["y"] / p["z"] ** 2
        fn = lambda p: C1 * u(p) ** 1.5 + C2 * u(p)
        truth = [O, M(-C1, x=1.5, y=1.5, z=-3), M(-C2, x=1, y=1, z=-2)]
    else:                                                    # log-index
        c1 = _r2(U(1, 4))
        c0 = _r2(7 * c1 + U(5, 20))
        fn = lambda p: c0 + c1 * np.log(p["x"] * p["y"] / p["z"] ** 2)
        truth = [O, M(-c0), (-c1, Feature.log("x")), (-c1, Feature.log("y")), (2 * c1, Feature.log("z"))]
    noise = U(0.001, 0.008)

    def make(rng2):
        def f(p):
            cols = dict(p)
            cols[out] = _noisy(rng2, fn(p), noise)
            return cols
        return Lab(knobs, f)
    return fam, make, truth, noise


def _truth_hit(eng, derived, terms):
    ents = [e for e in eng.archive.all if e.cls != "DEFINITION"]
    if any(terms_match(expand_terms(e.rel, derived), terms) for e in ents):
        return True
    if len(terms) == 2 and all(f.kind == "m" for _, f in terms) and eng.archive.mono:
        tm = Archive._mono_vec(Relation(terms[0][1], [(-terms[1][0], terms[1][1])]))
        if tm is not None and tm[0]:
            return eng.archive._mono_class(*tm)[0] in ("DERIVED", "REDUNDANT")
    return False


def practice(n_worlds, seed, mem, say, time_budget=120.0):
    """Self-directed practice: random worlds -> autonomous agent -> self-scored."""
    t_start, rows = time.time(), []
    rng = np.random.default_rng(seed)
    for k in range(n_worlds):
        if time.time() - t_start > time_budget:
            say(f"   [practice] time budget reached after {k} worlds")
            break
        wseed = int(rng.integers(1, 10 ** 6))
        fam, make, truth, noise = random_world(np.random.default_rng(wseed))
        lab = make(np.random.default_rng(wseed + 1))
        t0 = time.time()
        agent = Agent(lab, Config(seed=wseed), np.random.default_rng(wseed + 2))
        with _ctx.redirect_stdout(_io.StringIO()):
            eng = agent.run()
        ok = _truth_hit(eng, agent.derived, truth)
        secs = time.time() - t0
        rows.append((fam, ok, secs, lab.count, noise))
        say(f"   [practice {k + 1}/{n_worlds}] {fam:20s} noise {noise:.2%}  "
            f"{'FOUND' if ok else 'missed'}  ({secs:.1f}s, {lab.count} measurements)")
        if mem is not None:
            mem.record_skill(fam, ok, secs, lab.count, noise)
    if mem is not None:
        mem.note("run", f"practice: {sum(r[1] for r in rows)}/{len(rows)} random worlds solved")
        mem.record_run(dataset="practice", variables=[], experiments=0, rows=0,
                       recalled=0, new=0, search_seconds=round(time.time() - t_start, 1))
        mem.save()
    return rows


def autopilot_once(directory, out_dir, mem_path, seed, recursive, practice_n, log):
    created, analyses = [], []
    os.makedirs(out_dir, exist_ok=True)
    mem = Memory(mem_path) if mem_path else None
    pmem_path = os.path.join(out_dir, "phy_practice.json")
    files = scan_data_files(directory, recursive)
    log(f"[AUTOPILOT] scanning '{os.path.abspath(directory)}' -> {len(files)} candidate data file(s)")
    tables = []
    for p in files:
        try:
            t = load_any(p)
            nnum = sum(1 for v in t.values() if np.asarray(v).dtype != object)
            if nnum < 2 or len(next(iter(t.values()))) < 30:
                log(f"   skipped {os.path.basename(p)}: needs >=2 numeric columns and >=30 rows")
                continue
            tables.append((p, t))
            log(f"   read {os.path.basename(p)}: {len(next(iter(t.values())))} rows, "
                f"{nnum} numeric column(s)")
        except Exception as ex:
            log(f"   skipped {os.path.basename(p)}: {type(ex).__name__}: {ex}")
    if tables:
        for gi, g in enumerate(_group_tables(tables), 1):
            label = ",".join(os.path.basename(s) for s, _ in g["items"])[:70]
            log(f"\n[DATASET {gi}] {label}")
            analyses.append(analyze_group(label, g["items"], out_dir, mem, seed, f"d{gi}", log))
        for a in analyses:
            log("\n" + a["result"].summary())
            for p, t in a["proposals"]:
                log(f"   -> {os.path.basename(p)}: measurements that would best test `{t}`")
                created.append(p)
    else:
        n = practice_n if practice_n is not None else 8
        log(f"[AUTOPILOT] no usable data files found -> self-directed practice on {n} random worlds "
            f"it invents for itself")
        pmem = Memory(pmem_path)
        practice(n, seed, pmem, log)
        created.append(pmem_path)
    rows = self_assessment(pmem_path)
    if rows:
        log("\n[SELF-ASSESSMENT] from its own practice: " +
            ", ".join(f"{f} {s}/{n}" for f, s, n in rows))
    write_artifacts(out_dir, analyses, rows, created)
    if mem is not None:
        created.append(mem_path)
    log("\n[FILES CREATED]")
    for p in dict.fromkeys(created):
        log(f"   {os.path.abspath(p)}")
    return analyses


def autopilot(args, mem_path):
    log = lambda *a: print(*a, flush=True)
    PRIORS["named"] = PRIORS["grammar"] = bool(args.priors)
    out_dir = args.out_dir or args.dir
    if args.practice is not None and not args.watch:
        os.makedirs(out_dir, exist_ok=True)
        pm = Memory(os.path.join(out_dir, "phy_practice.json"))
        rows = practice(args.practice, args.seed, pm, log)
        write_artifacts(out_dir, [], self_assessment(pm.path), [])
        log(f"[FILES CREATED] {os.path.abspath(pm.path)}, {os.path.abspath(os.path.join(out_dir, 'phy_report.md'))}")
        return
    if not args.watch:
        autopilot_once(args.dir, out_dir, mem_path, args.seed, args.recursive, args.practice, log)
        return
    log(f"[WATCH] monitoring '{os.path.abspath(args.dir)}' every {args.poll:g}s - Ctrl-C to stop")
    last, cycles = None, 0
    try:
        while True:
            sig = tuple((p, os.path.getmtime(p), os.path.getsize(p))
                        for p in scan_data_files(args.dir, args.recursive))
            if sig != last:
                last = sig
                if not sig:
                    log(f"[WATCH] no data files yet ({time.strftime('%H:%M:%S')}) - waiting")
                    time.sleep(args.poll)
                    continue
                log(f"\n[WATCH] change detected ({time.strftime('%H:%M:%S')}) -> analysing")
                autopilot_once(args.dir, out_dir, mem_path, args.seed, args.recursive, None, log)
                cycles += 1
                if args.max_cycles and cycles >= args.max_cycles:
                    return
            time.sleep(args.poll)
    except KeyboardInterrupt:
        log("\n[WATCH] stopped")


def main():
    ap = argparse.ArgumentParser(
        description="phy_discover: find hidden relationships in data, or run an autonomous agent "
                    "that designs its own experiments.")
    ap.add_argument("--csv", nargs="+", help="analyse CSV data (one file = one experiment)")
    ap.add_argument("--experiment-col", help="column naming the experiment of each row")
    ap.add_argument("--outputs", nargs="+", help="readings to put on the left-hand side")
    ap.add_argument("--dims", help="JSON file: {\"L\": [0,1,0], ..., \"_base\": [\"M\",\"L\",\"T\"]}")
    ap.add_argument("--lab", help="python file defining KNOBS and measure(points): run the agent on it")
    ap.add_argument("--lab-template", action="store_true", help="print a template lab file")
    ap.add_argument("--out", help="save report to .md or .json")
    ap.add_argument("--mode", choices=["agent", "data"], default=None,
                    help="built-in benchmarks: agent (V5) or data (V4 engine)")
    ap.add_argument("--dir", default=".", help="autopilot: folder to scan for data files (default: here)")
    ap.add_argument("--out-dir", default=None, help="autopilot: where to write the files (default: --dir)")
    ap.add_argument("--recursive", action="store_true", help="autopilot: also scan sub-folders")
    ap.add_argument("--watch", action="store_true", help="autopilot: keep running, re-analyse on new/changed files")
    ap.add_argument("--poll", type=float, default=5.0, help="--watch: seconds between checks")
    ap.add_argument("--max-cycles", type=int, default=0, help="--watch: stop after N analyses (0 = never)")
    ap.add_argument("--practice", nargs="?", const=8, type=int, default=None, metavar="N",
                    help="self-directed practice on N random worlds it invents (default 8)")
    ap.add_argument("--challenge", default=None,
                    help="benchmarks. agent: law1 law3 suite pendulum composite logindex decay "
                         "arrhenius latent | all.  data: law1 law2 law3 law4 suite pendulum | all")
    ap.add_argument("--priors", action="store_true", help="allow pi/e/... constant priors")
    ap.add_argument("--no-active", action="store_true")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--memory", metavar="FILE",
                    help="JSON memory file. Default for --csv/--lab: phy_memory.json. "
                         "Benchmarks use none unless you pass this.")
    ap.add_argument("--domain", default=None,
                    help="memory partition: laws are only recalled inside the same domain "
                         "(use when different projects reuse the same variable names)")
    ap.add_argument("--no-memory", action="store_true", help="do not read or write any memory file")
    ap.add_argument("--memory-show", action="store_true", help="print what the memory contains and exit")
    ap.add_argument("--memory-notes", type=int, default=15, help="how many notes --memory-show prints")
    ap.add_argument("--memory-forget", metavar="TEXT", help="delete remembered laws containing TEXT")
    ap.add_argument("--memory-reset", action="store_true", help="delete the memory file")
    ap.add_argument("--memory-demo", metavar="CHALLENGE",
                    help="run a benchmark twice on a fresh memory and show the speed-up")
    a = ap.parse_args()
    default_mem = os.path.join(a.out_dir or a.dir, "phy_memory.json") \
        if (a.out_dir or a.dir != ".") else "phy_memory.json"
    mpath = None if a.no_memory else (a.memory or default_mem)

    if a.memory_reset:
        if os.path.exists(mpath or "phy_memory.json"):
            os.remove(mpath or "phy_memory.json")
        print("memory reset:", mpath or "phy_memory.json")
        return
    if a.memory_show or a.memory_forget:
        m = Memory(mpath or "phy_memory.json")
        if a.memory_forget:
            n = m.forget(a.memory_forget)
            m.save()
            print(f"forgot {n} law(s) containing '{a.memory_forget}'")
        print(m.describe(a.memory_notes))
        return
    if a.memory_demo:
        import tempfile
        PRIORS["named"] = PRIORS["grammar"] = a.priors
        path = os.path.join(tempfile.mkdtemp(), "demo_memory.json")
        rows = []
        for k in (1, 2):
            m = Memory(path, domain=a.memory_demo)
            t0 = time.time()
            ok = run_challenge(a.memory_demo, Config(seed=a.seed), a.seed, memory=m)
            rows.append((k, time.time() - t0, ok))
        print("\n" + "=" * 78 + "\nMEMORY DEMO (same challenge, same seed, same memory file)")
        for k, sec, ok in rows:
            print(f"  run {k}: {sec:5.1f}s  {'recovered' if ok else 'MISSED'}")
        print(f"  speed-up: {rows[0][1] / max(rows[1][1], 1e-9):.1f}x\n")
        print(Memory(path).describe(8))
        return

    if a.lab_template:
        print(LAB_TEMPLATE)
        return
    if a.csv:
        dims = base = None
        if a.dims:
            raw = json.load(open(a.dims))
            base = raw.pop("_base", None)
            dims = {k: tuple(v) for k, v in raw.items()}
        res = discover(a.csv, experiment=a.experiment_col, outputs=a.outputs, dims=dims,
                       base_dims=base, seed=a.seed, verbose=a.verbose, priors=a.priors,
                       memory=mpath, name=",".join(a.csv)[:60], domain=a.domain)
        print(res.summary())
        if a.verbose:
            res.report()
        if a.out:
            print("saved", res.save(a.out))
        return
    if a.lab:
        knobs, measure = _load_lab(a.lab)
        res = explore(knobs, measure, seed=a.seed, max_rounds=a.rounds, priors=a.priors,
                      memory=mpath, name=a.lab, domain=a.domain)
        print("\n" + "=" * 78 + "\n" + res.summary())
        if a.out:
            print("saved", res.save(a.out))
        return

    # ---- AUTOPILOT: no instructions given -> find the data and do everything ----
    if a.challenge is None and a.mode is None:
        autopilot(a, mpath)
        return
    a.challenge = a.challenge or "all"

    # ---- built-in benchmarks (simulated worlds written by the developer, for self-testing) ----
    mode = a.mode or "agent"
    if mode == "data":
        cfg = Config(seed=a.seed)
        names = list(LAWS) if a.challenge == "all" else [a.challenge]
        for n in names:
            known = [Relation(Feature.mono({"A": 1}), [(2.75, Feature.mono({"x": 1, "y": 1}))])] \
                if n == "suite" else ()
            run_hidden_law_challenge(LAWS[n], cfg, active=not a.no_active, known=known)
        return
    PRIORS["named"] = PRIORS["grammar"] = a.priors
    cfg = Config(seed=a.seed)
    names = CHALLENGES + ["latent"] if a.challenge == "all" else [a.challenge]
    res = {}
    for n in names:
        if n == "latent":
            run_latent(cfg, a.seed)
        else:
            res[n] = run_challenge(n, cfg, a.seed,
                                   memory=Memory(a.memory, domain=n) if (a.memory and not a.no_memory)
                                   else None)
    print("\n" + "=" * 78 + "\nAUTONOMOUS BENCHMARK SUMMARY")
    for n, ok in res.items():
        print(f"  {n:10s} {'RECOVERED' if ok else 'MISSED'}")
    bp = os.path.join(a.out_dir or ".", "phy_benchmark.json")
    with open(bp, "w") as f:
        json.dump(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), seed=a.seed,
                       results={k: bool(v) for k, v in res.items()}), f, indent=1)
    print(f"[FILE CREATED] {os.path.abspath(bp)}")


if __name__ == "__main__":
    main()
