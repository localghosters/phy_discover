#!/usr/bin/env python3
"""
PHYDISCOVER V5 - autonomous discovery agent
============================================

V4 analysed data you handed it. V5 *acts*:

    Lab (knobs + ranges only)  -->  Agent
        1. EXPLORE      designs its own first experiments (full range, high
                        regime, low regime, one-factor-at-a-time sweeps)
        2. DISCOVER     V4 engine: power laws, invariants, sparse implicit laws
        3. INVENT       if a measured output is still unexplained, search for a
                        NEW VARIABLE u = prod(knob^p) such that the output is a
                        1-D function of u (single-index search), add u as a
                        column, and rediscover. Invented variables can chain.
        4. LEARN        exponents found in laws are fed back into the search
                        vocabulary (grammar growth)
        5. FALSIFY      measures fresh points far outside the explored range and
                        tests every hypothesis on them
        6. DESIGN       picks the measurements where surviving hypotheses disagree
                        most, runs them, re-discovers
        7. STOP         when nothing changes and every hypothesis survives

No formula, no target variable and (by default) no constant is built in.
The hidden laws exist ONLY inside the simulated lab, which stands in for a real
instrument: replace `Lab.fn` with a function that talks to hardware.

Latent variables: if a law holds inside every experiment but its constant
differs between experiments, the agent invents a per-experiment latent factor.

Requires numpy and phydiscover_v4.py (same folder).
"""
from __future__ import annotations

import argparse
import itertools
import math
from functools import reduce

import numpy as np

import phydiscover_v4 as v4
from phydiscover_v4 import (Config, DataBundle, Discoverer, Experiment, Feature,
                            HiddenLaw, Relation, ActiveDesigner, fmt, median_err,
                            mono_text, stack_from_cols, terms_match)


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
    def __init__(self, lab, cfg, rng, n0=200, max_rounds=4, quiet_cfg=True):
        self.lab, self.cfg, self.rng = lab, cfg, rng
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
        eng = Discoverer(self.bundle, self.qcfg, name="agent",
                         defs=[d.relation() for d in self.derived],
                         extra_exps=sorted(self.extra_exps))
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
            name = f"u{len(self.derived) + 1}"
            d = DerivedVar(name, res["exps"])
            self.derived.append(d)
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
        return bad

    def design(self, eng, tag):
        d = ActiveDesigner(self.bundle, self.qcfg, list(self.lab.knobs), self.augment,
                           [x.name for x in self.derived])
        prop = d.propose(eng.archive, k=6, replicates=10)
        if prop:
            sel, _ = prop
            self.measure(f"active_{tag}", sel)
            return True
        return False

    # ---- main loop -------------------------------------------------------------
    def run(self):
        self.explore()
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
            if not bad and sig == prev:
                self.say("[STOP] hypotheses unchanged and none falsified")
                break
            prev = sig
            if not bad:
                self.design(eng, r)
        self.eng = self.discover()
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


CHALLENGES = ["law1", "law3", "suite", "pendulum", "composite", "logindex"]


def run_challenge(name, cfg, seed):
    rng = np.random.default_rng(seed)
    lab, truths = make_lab(name, rng)
    print("\n" + "#" * 78)
    print(f"# AUTONOMOUS CHALLENGE: {name}")
    print("# The agent gets knob names + ranges only. No data, no formula, no target.")
    print("#" * 78)
    agent = Agent(lab, cfg, rng)
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


def main():
    ap = argparse.ArgumentParser(description="PhyDiscover V5 autonomous agent")
    ap.add_argument("--challenge", default="all",
                    help="law1 law3 suite pendulum composite logindex latent | all")
    ap.add_argument("--priors", action="store_true",
                    help="allow built-in constant priors (pi, e, ...). Default: learn from data only")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    v4.PRIORS["named"] = a.priors
    v4.PRIORS["grammar"] = a.priors
    cfg = Config(seed=a.seed)
    names = CHALLENGES + ["latent"] if a.challenge == "all" else [a.challenge]
    res = {}
    for n in names:
        if n == "latent":
            run_latent(cfg, a.seed)
        else:
            res[n] = run_challenge(n, cfg, a.seed)
    print("\n" + "=" * 78 + "\nAUTONOMOUS BENCHMARK SUMMARY")
    for n, ok in res.items():
        print(f"  {n:10s} {'RECOVERED' if ok else 'MISSED'}")


if __name__ == "__main__":
    main()
