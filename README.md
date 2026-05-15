# numpy_grover

**Quantum-inspired global optimisation in pure NumPy.**

Implements the [Grover / Durr-Hoyer](https://arxiv.org/abs/quant-ph/9607014) minimum-finding algorithm with a hierarchical zoom-and-refine strategy and a hybrid adaptive minimiser for medium-to-high dimensional problems.

**Author:** Faluke Aikebaier  
**License:** MIT

---

## What it does

Classical exhaustive grid search over N points costs **O(N)**.  
Grover's algorithm finds the minimum in **O(√N)** — a quadratic speedup on the search step.

```
N = 65,536 points  →  classical: 65,536 checks  →  Grover: ~200 oracle calls
```

> **Honest note:** On classical hardware the grid must still be *evaluated* at all N points (unavoidable). The O(√N) speedup applies to the *search* over the pre-evaluated array. On a real quantum computer the oracle would evaluate all points simultaneously, eliminating the O(N) evaluation cost entirely.

---

## Three functions

| Function | Best for | Dimensions |
|---|---|---|
| `grover_minimize` | Quick single-layer search | D=1–4 |
| `grover_minimize_hierarchical` | High accuracy via zoom-and-refine | D=2–4 |
| `hybrid_adaptive_minimize` | Mixed landscapes, higher dimensions | D=2–50+ |

---

## Installation

```bash
# From GitHub (recommended)
pip install git+https://github.com/Faluke-Aikebaier/numpy_grover.git

# With CMA-ES support for D>10 in hybrid_adaptive_minimize
pip install "git+https://github.com/Faluke-Aikebaier/numpy_grover.git#egg=numpy_grover[cmaes]"
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.24, SciPy ≥ 1.10

---

## Quick start

```python
import numpy as np
from numpy_grover import (
    grover_minimize,
    grover_minimize_hierarchical,
    hybrid_adaptive_minimize,
)

# Your function must accept X of shape (N, D) and return shape (N,)
def wavy_bowl(X):
    x, y = X[:, 0], X[:, 1]
    return x**2 + y**2 + 0.4 * np.sin(5*x) * np.sin(5*y)

# ── 1. Quick search ───────────────────────────────────────────────────
res = grover_minimize(wavy_bowl, bounds=[(-2, 2), (-2, 2)], n_bits=8)
print(res.x)    # [0.2596, -0.2596]
print(res.fun)  # -0.2362

# ── 2. High accuracy via zoom ─────────────────────────────────────────
res = grover_minimize_hierarchical(
    wavy_bowl, bounds=[(-2, 2), (-2, 2)],
    n_bits_schedule=(6, 8, 10), max_layers=3)
print(res.fun)           # -0.23617967
print(res.stop_reason)   # 'tol_f=1.0e-06 met'

# ── 3. Adaptive for any dimension ─────────────────────────────────────
res = hybrid_adaptive_minimize(
    wavy_bowl, bounds=[(-2, 2), (-2, 2)],
    n_bits_schedule=(6, 8), max_cycles=4)
print(res.dim_characters)  # ['multimodal', 'multimodal']
print(res.dim_methods)     # ['grover', 'grover']
```

---

## Function 1 — `grover_minimize`

Single-layer search over a discrete grid.

```python
res = grover_minimize(
    func      = my_function,      # f(X) where X shape (N, D) → (N,)
    bounds    = [(-5, 5)] * 2,    # one (lo, hi) per dimension
    n_bits    = 8,                # grid resolution: 2^n_bits per dim
    n_trials  = 3,                # independent Durr-Hoyer runs
    n_repeats = 1,                # averaging for noisy functions
    seed      = 42,
)

res.x          # coordinates of minimum, shape (D,)
res.fun        # function value at minimum
res.n_calls    # oracle calls used
res.elapsed    # wall-clock seconds
```

**Grid resolution guide:**

| `n_bits` | pts/dim | 2D grid | oracle calls |
|---|---|---|---|
| 4 | 16 | 256 | ~12 |
| 6 | 64 | 4,096 | ~50 |
| 8 | 256 | 65,536 | ~200 |
| 10 | 1,024 | 1,048,576 | ~800 |

---

## Function 2 — `grover_minimize_hierarchical`

Zoom-and-refine: repeatedly shrinks the search window for high accuracy.

```
Layer 0:  search full domain at n_bits=6    → locate basin
Layer 1:  zoom to ±6 grid-spacings → n_bits=8  → refine
Layer 2:  zoom again → n_bits=10               → high precision
Stop when: |Δf| < tol_f  or  |Δx| < tol_x  or  max_layers reached
```

```python
res = grover_minimize_hierarchical(
    func            = my_function,
    bounds          = [(-512, 512)] * 2,
    n_bits_schedule = (6, 8, 10),   # bits per run within each layer
    max_layers      = 3,
    zoom_factor     = 6.0,          # window = ±zoom × grid_spacing
    tol_f           = 1e-6,
    tol_x           = 1e-6,
    n_repeats       = 1,
    seed            = 42,
    verbose         = True,
)

res.layers          # list of LayerRecord — trajectory per layer
res.converged       # True if tol_f or tol_x was met
res.stop_reason     # human-readable explanation
res.summary()       # pretty-print the per-layer table
```

**Example output (Eggholder function, true min at (512, 404.23)):**

```
Hierarchical Grover search — 3 layer(s)
==================================================================================
 Layer     window   n_bits sched                x*              f*    calls  conv?
----------------------------------------------------------------------------------
     0 1024.00000     [6, 8, 10] (512.0, 403.89)    -959.511560   13,879      ·
     1   12.01173     [6, 8, 10] (512.0, 404.23)    -959.640654   13,485      ·
     2    0.14090     [6, 8, 10] (512.0, 404.23)    -959.640663    6,113      ·
==================================================================================
  Final:  x = [512.000000, 404.231752]
          f = -959.64066272
  Stop reason: max_layers reached
  Total calls: 33,477   Total time: 50.4s
```

---

## Function 3 — `hybrid_adaptive_minimize`

Coordinate-wise adaptive search. Each dimension gets the cheapest method
that is sufficient for its shape.

```
Stage 1 — Global basin:
    D ≤ 4   → grover_minimize_hierarchical
    D ≤ 10  → scipy.differential_evolution
    D > 10  → CMA-ES (if installed) or differential_evolution

Stage 2/3 — Coordinate-wise refinement:
    Classify each dimension:
        multimodal  → grover_minimize_hierarchical (1D, exhaustive)
        unimodal    → Brent (scipy, ~15 evals)
        monotone    → Brent
        flat        → skip
    Cycle until convergence.

Stage 4 — Joint zoom (if affordable):
    grover_minimize_hierarchical in the full D-dim tiny window
```

```python
res = hybrid_adaptive_minimize(
    func             = my_function,
    bounds           = [(-5.12, 5.12)] * 5,   # 5D problem
    n_bits_schedule  = (6, 8),
    max_cycles       = 5,
    zoom_factor      = 6.0,
    tol_f            = 1e-8,          # absolute convergence
    tol_f_rel        = 1e-6,          # relative convergence (use for large f)
    n_repeats        = 1,             # set >1 for noisy/stochastic functions
    n_trials         = 3,
    dim_names        = ['x0','x1','x2','x3','x4'],   # optional labels
    seed             = 42,
    verbose          = True,
)

res.x                # final coordinates
res.fun              # final function value
res.dim_characters   # ['multimodal', 'unimodal', ...] — one per dim
res.dim_methods      # ['grover', 'brent', ...] — method used per dim
res.coupling_warning # True if dimensions appear strongly coupled
res.cycle_history    # list of CycleRecord — convergence trajectory
res.summary()        # pretty-print
```

**Example output (5D Rastrigin, true min = 0 at origin):**

```
Hybrid Adaptive Minimiser
  D=5  max_cycles=5  schedule=[6, 8]
  tol_x=1.0e-06  tol_f=1.0e-08  tol_f_rel=1.0e-06
==========================================================================
  Stage 1: Global basin finding
    method : Differential Evolution
    x*     : [0.0, 0.0, 0.0, 0.0, 0.0]
    f*     : 0.00000008
    calls  : 6,999

  Stage 3: Coordinate-wise refinement
  ── Cycle 0 ──
   dim     name    character     method    x_before     x_after         Δx   calls
     0       x0   multimodal     grover     0.00000    -0.00001   1.39e-05     361
     1       x1   multimodal     grover     0.00000    -0.00001   1.39e-05     363
     ...
  → f=0.00000008  max_Δx=9.72e-02  Δf=9.95e-01  calls=1,839

  Stop: tol_f_rel=1.0e-06 met (Δf/|f|=1.04e-09)
  Total calls: 21,310   Time: 22.1s
```

---

## Key parameters explained

### `n_bits` / `n_bits_schedule`
Controls grid resolution. Higher = more accurate, more evaluations.

```python
# Fine result — use (6, 8, 10) schedule
grover_minimize_hierarchical(f, bounds, n_bits_schedule=(6, 8, 10))

# Fast result — use (6,) only
grover_minimize_hierarchical(f, bounds, n_bits_schedule=(6,))
```

### `n_repeats` — for noisy functions
Averages multiple evaluations per grid point to suppress stochastic noise.

```python
# Stochastic function (Monte Carlo, simulation output)
hybrid_adaptive_minimize(f, bounds, n_repeats=5)  # average 5 evals per point
```

### `tol_f_rel` — for large function values
Use relative tolerance when `f` is large (e.g. Rastrigin at D=8 has f~80).

```python
# Absolute tol_f=1e-6 on f~1000 fires immediately — meaningless
# Use relative instead:
hybrid_adaptive_minimize(f, bounds, tol_f=1e-9, tol_f_rel=1e-6)
```

### `zoom_factor` — for narrow basins
Larger zoom_factor = wider safety net around the found minimum.

```python
# Schwefel has very narrow basin (~0.4 unit wide in 1000-unit domain)
hybrid_adaptive_minimize(schwefel, bounds, zoom_factor=10.0)
```

---

## Benchmark results

Tested on four standard functions across D=2, 5, 8:

| Function | Character | D=2 dist | D=5 dist | D=8 dist |
|---|---|---|---|---|
| Wavy Bowl | Smooth multimodal | 0.00022 | 0.00003 | 0.00004 |
| Rastrigin | Regular traps | 0.00000 | 0.00003 | 0.00004 |
| Schwefel | Deceptive | 0.00065 | 0.00139 | 0.944* |
| Eggholder | 800 local minima | 0.00015 | — | — |

`dist` = Euclidean distance from found minimum to known true minimum.  
`*` Schwefel D=8: correct f value (≈0.0001), coordinate drift from narrow basin.  
Use `n_bits=(8,10)` or tighter bounds to improve D=8 Schwefel accuracy.

**Comparison with classical methods (Rastrigin D=8):**

| Method | D=8 dist | verdict |
|---|---|---|
| **hybrid_adaptive_minimize** | **0.00004** | ✓ |
| Dual Annealing | 0.00000 | ✓ |
| Differential Evolution | 1.407 | ✗ |
| CMA-ES | 1.723 | ✗ |

`hybrid_adaptive_minimize` finds the true minimum at D=8 where DE and CMA-ES fail, because Grover's exhaustive 1D search finds the *global* minimum of each coordinate slice — not just the nearest local one.

---

## Accuracy vs speed

```
grover_minimize:              fast, moderate accuracy
grover_minimize_hierarchical: moderate speed, high accuracy
hybrid_adaptive_minimize:     best for D>2, adaptive, high accuracy
```

For D=2 problems where you need the highest possible accuracy:

```python
# Recommended: hierarchical with fine schedule
res = grover_minimize_hierarchical(
    f, bounds, n_bits_schedule=(6, 8, 10), max_layers=4, tol_f=1e-8)
```

For D=5+ problems:

```python
# Recommended: adaptive with moderate bits (zoom handles accuracy)
res = hybrid_adaptive_minimize(
    f, bounds, n_bits_schedule=(6, 8), max_cycles=5, tol_f_rel=1e-6)
```

---

## How it works

### Durr-Hoyer algorithm
1. Start with uniform superposition over all N grid points
2. Pick a random threshold index t
3. Run Grover iterations: **oracle** (phase-flip cheaper states) + **diffusion** (inversion about mean)
4. Measure: sample from |amplitude|² — finds cheaper state with high probability
5. Update threshold if cheaper state found; grow iteration count by λ=6/5
6. Terminate after 22.5√N + 1.4·log₂(N)² oracle calls

### Hierarchical zoom
After Durr-Hoyer finds x* on a coarse grid, zoom the search window to
`[x* ± zoom_factor × grid_spacing]` and repeat. Each layer gives ~10-100×
finer resolution with the same computational budget.

### Hybrid adaptive
Routes each 1D coordinate search to the cheapest sufficient method by
profiling the function along that dimension (24 probe points → classify
as flat/monotone/unimodal/multimodal → assign method).

---

## Running the examples

```bash
git clone https://github.com/Faluke-Aikebaier/numpy_grover.git
cd numpy_grover
pip install -e .
python examples.py
```

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Citing

If you use this library in research, please cite:

```
Faluke Aikebaier, numpy_grover: Quantum-inspired global optimisation in NumPy,
https://github.com/Faluke-Aikebaier/numpy_grover, 2025
```

---

## Further reading

- Durr & Hoyer (1996) — [A Quantum Algorithm for Finding the Minimum](https://arxiv.org/abs/quant-ph/9607014)
- Grover (1996) — [A fast quantum mechanical algorithm for database search](https://arxiv.org/abs/quant-ph/9605043)
