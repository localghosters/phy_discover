# phy_discover

Finds simple, accurate, non-redundant mathematical relationships hidden in measurements,
or runs an agent that designs its own experiments to find them. One file, NumPy only.

## Use it on your data

```bash
python phy_discover.py --csv run1.csv run2.csv run3.csv --out report.md   # one file = one experiment
python phy_discover.py --csv all.csv --experiment-col run_id               # or one file, a run column
python phy_discover.py --csv all.csv --outputs T E --dims dims.json        # optional: units / which columns are readings
```

```python
from phy_discover import discover
res = discover("all.csv", experiment="run_id")
print(res.summary())          # plain-English list of relationships
res.save("report.md")         # or report.json
```

Messy CSVs are fine: text columns are ignored, stray `n/a` cells and blank cells are dropped
(and reported), a column counts as numeric if >= 80% of its cells parse.
Give it **several experiments** if you can: a relation must hold with the same constants in every
one, which is the main protection against overfitting. A constant that differs between
experiments is reported as a possible hidden variable.

## Let the agent run your instrument

```bash
python phy_discover.py --lab-template > my_lab.py     # describe knobs + a measure() function
python phy_discover.py --lab my_lab.py --out report.md
```

The agent picks every setting itself, invents new variables when a reading is unexplained,
tests its own hypotheses on measurements 3x outside the explored range, designs the most
informative next measurements, and stops when nothing changes. `explore(knobs, measure)` is the
Python equivalent.

## Memory: it learns between runs

Pass a JSON file and the engine reads it at the start and writes to it at the end.
Nothing it has already worked out is rediscovered.

```python
res = discover("runs.csv", experiment="run_id", memory="phy_memory.json")
res = explore(knobs, measure, memory="phy_memory.json")
```
```
python phy_discover.py --csv runs.csv --experiment-col run_id      # default file: phy_memory.json
python phy_discover.py --memory-show                               # laws, inventions, constants, notes
python phy_discover.py --memory-demo composite                     # same task twice: ~300x faster
```

What it stores and what it does with it:

| Memory | Used for |
|---|---|
| **laws** (with confirmed / failed counts) | re-validated on new data in milliseconds. If every reading is explained by a remembered law that still holds, the whole search is skipped; the agent also stops early once fresh out-of-range measurements confirm them |
| **falsified** hypotheses | never trusted blindly again: a candidate matching one gets a warning and half the interestingness. Cleared automatically if the law later holds on new data |
| **constants** seen in 2+ runs | reused when snapping, so the same constant is reported identically across datasets. Related constants are noted (`b = a^2`, ...) |
| **exponents** seen in 2+ runs | added to the search vocabulary |
| **inventions** (invented variables) | re-adopted after a one-line check instead of searched for again |
| **stage statistics** | a search stage that has produced nothing in 8+ runs is only sampled every 5th run |
| **notes** and **history** | the engine's own log: what it learned, what failed, lessons, hints about hidden variables |

Safety: a remembered law is never trusted, only re-validated against the new data. If the world
changes it fails validation, is counted against, and is retired after 3 failures (and fixed laws are
re-learned). The file is written atomically; a corrupt file is moved aside and the run continues.
Use `--domain NAME` (or `domain=`) to keep projects that reuse variable names apart, and
`--memory-forget TEXT` / `--memory-reset` to edit or clear it. Memory is matched by variable
**name**, so keep names consistent between runs.

## What it can find

| Family | Example |
|---|---|
| power laws, invariants, dimensionless groups | `E = 19.75*amp^2*m/T^2` |
| sums / ratios / logs (sparse implicit laws) | `Y*(z+2) = x*y`, `E = 1.7*x*y + 2*z` |
| exponential, Arrhenius, Gaussian | `R = 30000*c^2*exp(-2500/T)` |
| chained laws | `heat = 2 + 0.3*rate*stirring^0.5` |
| laws of an invented variable | `Y = 2.2*u^1.5 + 4*u`, `u := x*y/z^2` |
| per-experiment hidden factors | different gravity on each planet |

Constants are learned from data (`2.7498 +- 0.0003 -> 2.75`) and tagged learned / grammar / fitted.
Equivalent forms (`A=BC`, `A/B=C`, `A-BC=0`) collapse into one entry; relations implied by
others are marked DERIVED. Each result gets accuracy, simplicity, generality, robustness and
novelty scores.

## Benchmarks

```bash
python phy_discover.py                          # 9 autonomous hidden-law challenges + latent variable (~10 s)
python phy_discover.py --mode data --challenge all
```

## Limits (read before trusting a result)

* Search vocabulary: powers, products, ratios, logs, exp of a power. Not `sin`/`cos`, saturation
  curves, or arbitrary functions.
* Variable invention finds single-index structure only (an output that depends on one
  combination of knobs).
* Noise is assumed multiplicative and roughly Gaussian with occasional outliers.
* A hypothesis is only as good as the range it was tested on; it checks 3x beyond the data it
  has seen, not beyond that.
