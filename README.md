# phy_discover

Finds simple, accurate, non-redundant mathematical relationships hidden in measurements,
or runs an agent that designs its own experiments to find them. One file, NumPy only.

## Autopilot: just run it

```bash
python phy_discover.py            # look in this folder, work everything out, write the files
python phy_discover.py --dir data/ --recursive
python phy_discover.py --watch    # keep running; re-analyse whenever a file appears or changes
python phy_discover.py --practice 12   # self-directed practice (also what it does if there is no data)
```

With no instructions it:

1. **finds the data**: `.csv .tsv .txt .json .npz` in the folder (its own `phy_*` files are ignored);
   unreadable or too-small files are skipped and say why
2. **works out the structure**: several files with the same columns are separate experiments; one file
   with a run/trial/batch/... label column (or a text column with >=2 groups of >=30 rows) is split into
   experiments; index columns, text columns and exact scaled copies of another column are dropped
3. **reads units from headers** such as `g [m/s^2]` or `T (K)` and switches dimensional analysis on
   (relations whose terms have different dimensions are rejected)
4. **picks the accuracy bar itself**: tries 2%, then 5%, then 10%, so noisy data is not silently ignored,
   and on pure noise it still reports nothing
5. **handles variables set once per experiment** (gravity per planet, temperature per run)
6. **discovers, merges equivalent forms, and recalls what it already knew** (memory)
7. **asks for what it needs next**: `phy_next_experiments_*.csv` lists the settings that would best
   separate the surviving hypotheses (set the inputs, measure the sensitive quantity)
8. **practises when there is nothing to analyse**: it invents random hidden laws for itself, tries to
   find them, and records which kinds it can and cannot find in `phy_practice.json`; that self-assessment
   is appended to every report

Files it creates in the folder: `phy_memory.json`, `phy_report.md`, `phy_report.json`,
`phy_next_experiments_*.csv`, `phy_practice.json` (practice only), `phy_benchmark.json` (benchmarks).
`--out-dir` puts them elsewhere; `--no-memory` skips the memory file.

## Use it on specific data (explicit control)

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
| **stage statistics** | an *expensive* stage (>= 1 s/run) that has produced nothing in 8+ runs is only sampled every 5th run. Cheap stages are never skipped (a stage that only fires on certain data looks useless until that data arrives) |
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
| periodic and damped oscillation (sin/cos, any phase, optional `exp(-k*x)` envelope, plus other terms) | `Q = 10 + 2*z + 3.06*sin(1.3*x) + 2.58*cos(1.3*x)` and it reports amplitude 4.002, period 4.833, phase 0.7005 |
| chained laws | `heat = 2 + 0.3*rate*stirring^0.5` |
| laws of an invented variable | `Y = 2.2*u^1.5 + 4*u`, `u := x*y/z^2` |
| per-experiment hidden factors | different gravity on each planet |

Constants are learned from data (`2.7498 +- 0.0003 -> 2.75`) and tagged learned / grammar / fitted.
Equivalent forms (`A=BC`, `A/B=C`, `A-BC=0`) collapse into one entry; relations implied by
others are marked DERIVED. Each result gets accuracy, simplicity, generality, robustness and
novelty scores.

## Benchmarks

```bash
python phy_discover.py --challenge all          # 12 hidden-law challenges + latent variable (~16 s)
python phy_discover.py --mode data --challenge all
```

## Limits (read before trusting a result)

* **Autopilot heuristics can guess wrong.** Experiment-label detection needs a name hint (run, trial,
  batch, ...) for numeric columns, or a text column; unit parsing knows common SI symbols only; proposals
  are only made for explicit laws. Everything it decided is printed (`[auto]`) and written to the report, so
  check it.
* **Practice measures itself, not nature.** The random worlds are written by this program from the law
  families it already searches, so the self-assessment is an honest check on those families and says
  nothing about laws outside them.

* Search vocabulary: powers, products, ratios, logs, exp of a power, (damped) sin/cos of a single
  input (additively combined with other terms). Not an amplitude that depends on another variable
  (`z*sin(w*x)`), saturation curves, or arbitrary functions.
* An approximate law that only just passes the accuracy bar is re-examined: if the engine can find a
  relation for the same variable that is at least 2x more accurate, it replaces the approximation
  (e.g. a power law fitted to a log curve). Relations with the predicted quantity on both sides must
  be 3x better than an explicit one, because noise on both sides makes them look better than they are.
* The oscillation scan needs several periods inside the measured range and looks for it against
  inputs (if you declare `outputs`, it does not scan reading-against-reading).
* Variable invention finds single-index structure only (an output that depends on one
  combination of knobs).
* Noise is assumed multiplicative and roughly Gaussian with occasional outliers.
* A hypothesis is only as good as the range it was tested on; it checks 3x beyond the data it
  has seen, not beyond that.
