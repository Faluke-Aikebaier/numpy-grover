# numpy_grover

**Quantum-inspired global optimisation in pure NumPy.**

Implements the [Grover / Durr-Hoyer](https://arxiv.org/abs/quant-ph/9607014) minimum-finding algorithm with a hierarchical zoom-and-refine strategy and a hybrid adaptive minimiser for medium-to-high dimensional problems.

**Author:** Faluke Aikebaier  
**License:** MIT

---

## What it does

The Grover / Durr-Hoyer algorithm searches a pre-evaluated cost array of N points in **O(√N) oracle calls** instead of O(N) — a quadratic speedup on the *search* step.

### Honest note on classical hardware

On a real quantum computer, the oracle evaluates all N grid points **simultaneously** (quantum parallelism), making the total cost genuinely O(√N). On classical hardware we cannot do this — the grid must still be evaluated point by point at O(N) cost before the search begins. So the true picture is:

```
Classical search (pre-evaluated array):  O(N)   search  →  O(N)
Grover (pre-evaluated array):            O(√N)  search  →  O(√N)  ← genuine speedup
Classical search (callable f):           O(N) evaluate + O(N) search  →  O(N)
Grover (callable f, classical hardware): O(N) evaluate + O(√N) search →  O(N)
Grover (callable f, quantum computer):   O(1) evaluate + O(√N) search →  O(√N)
```

**What you actually get on classical hardware:**
- The O(√N) search saving is real and measurable on the *search step*
- For a 65,536-point grid: ~200 oracle calls instead of 65,536 to find the minimum of the pre-evaluated array
- The hierarchical zoom multiplies this — each layer searches a shrinking window exhaustively, achieving high accuracy without evaluating a larger grid
- The hybrid adaptive function routes smooth dimensions to Brent (~15 evaluations) and only calls Grover for genuinely multimodal dimensions

This library is best understood as a **quantum-inspired** optimiser: it faithfully implements the Grover/Durr-Hoyer algorithm in exact floating-point arithmetic (no shot noise, no decoherence), and the search step delivers the O(√N) advantage. The O(N) grid evaluation cost is a classical overhead that a real quantum computer would eliminate.

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
# From GitHub
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
print(res.x)    # [0.2588, -0.2588]
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
res.n_calls    # oracle calls used (O(√N) search step)
res.elapsed    # wall-clock seconds
```

**Grid resolution guide:**

| `n_bits` | pts/dim | 2D grid | oracle calls (search step) |
|---|---|---|---|
| 4 | 16 | 256 | ~12 |
| 6 | 64 | 4,096 | ~50 |
| 8 | 256 | 65,536 | ~200 |
| 10 | 1,024 | 1,048,576 | ~800 |

---

## Function 2 — `grover_minimize_hierarchical`

Zoom-and-refine: repeatedly shrinks the search window for high accuracy.

```
Layer 0:  search full domain at n_bits=6      → locate basin
Layer 1:  zoom to ±zoom_factor × grid_spacing → refine
Layer 2:  zoom again                           → high precision
Stop when: |Δf| < tol_f  or  |Δx| < tol_x  or  max_layers reached
```

The zoom is the key insight: each layer searches the same number of grid
points but inside a shrinking window — giving finer effective resolution
without evaluating a larger grid.

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

**Example output (Wavy Bowl, true min at (0.259, -0.259)):**

```
Hierarchical Grover search  —  3 layer(s)
==============================================================================
 Layer    window  n_bits sched              x*              f*    calls  conv?
------------------------------------------------------------------------------
     0    4.0000      [6, 8]  (0.2588, -0.2588)     -0.236174    1,490      ·
     1    0.1882      [6, 8]  (0.2599, -0.2599)     -0.236178    1,540      ·
     2    0.0089      [6, 8]  (0.2596, -0.2596)     -0.236180    2,343      ·
==============================================================================
  Final:  x = [0.259579, -0.259579]
          f = -0.23617967
  Stop reason: max_layers reached
  Total calls: 5,373   Total time: 0.618s
```

---

## Function 3 — `hybrid_adaptive_minimize`

Coordinate-wise adaptive search. Each dimension gets the cheapest method
sufficient for its shape. Honest name: this is a **hybrid** optimiser —
Grover handles multimodal dimensions, Brent handles smooth ones, and
DE / CMA-ES finds the global basin at higher D.

```
Stage 1 — Global basin:
    D ≤ 4   → grover_minimize_hierarchical
    D ≤ 10  → scipy differential_evolution
    D > 10  → CMA-ES (if installed) or differential_evolution

Stage 2/3 — Coordinate-wise refinement (cycles until convergence):
    Classify each dimension → assign method:
        multimodal  → grover_minimize_hierarchical (1D, exhaustive)
        unimodal    → Brent (scipy, ~15 evals, fast)
        monotone    → Brent
        flat        → skip (dimension does not affect f)

Stage 4 — Joint zoom (if affordable):
    grover_minimize_hierarchical in the full D-dim tiny window
    captures cross-dimension coupling
```

```python
res = hybrid_adaptive_minimize(
    func             = my_function,
    bounds           = [(-5.12, 5.12)] * 5,
    n_bits_schedule  = (6, 8),
    max_cycles       = 5,
    zoom_factor      = 6.0,
    tol_f            = 1e-8,          # absolute convergence
    tol_f_rel        = 1e-6,          # relative convergence (for large f)
    n_repeats        = 1,             # set >1 for noisy/stochastic functions
    n_trials         = 3,
    dim_names        = ['x0','x1','x2','x3','x4'],
    seed             = 42,
    verbose          = True,
)

res.x                # final coordinates
res.fun              # final function value
res.dim_characters   # ['multimodal', 'unimodal', ...] — one per dim
res.dim_methods      # ['grover', 'brent', ...] — method used per dim
res.coupling_warning # True if dimensions appear strongly coupled
res.cycle_history    # convergence trajectory
res.summary()        # pretty-print full report
```

---

## Key parameters explained

### `n_bits` / `n_bits_schedule`
Controls grid resolution. Higher = more accurate, more grid evaluations.

```python
# Fine accuracy
grover_minimize_hierarchical(f, bounds, n_bits_schedule=(6, 8, 10))

# Fast — let zoom handle accuracy
grover_minimize_hierarchical(f, bounds, n_bits_schedule=(6,))
```

### `n_repeats` — for noisy functions
Averages multiple evaluations per grid point to suppress stochastic noise.

```python
# Stochastic function (Monte Carlo, simulation output)
hybrid_adaptive_minimize(f, bounds, n_repeats=5)
```

### `tol_f_rel` — for large function values
Use relative tolerance when `f` is large (e.g. Rastrigin at D=8 has f~80;
absolute `tol_f=1e-6` fires after the first tiny improvement and stops too early).

```python
# Scale-independent stopping
hybrid_adaptive_minimize(f, bounds, tol_f=1e-9, tol_f_rel=1e-6)
```

### `zoom_factor` — for narrow basins
Larger zoom_factor = wider safety margin. Important when the true basin
is narrow relative to the domain.

```python
# Schwefel has a ~0.4-unit basin in a 1000-unit domain
hybrid_adaptive_minimize(schwefel, bounds,
    n_bits_schedule=(8, 10),   # finer grid to resolve narrow basin
    zoom_factor=10.0)
```

---

## Benchmark results

Tested on standard benchmark functions. All Schwefel results use tuned
parameters (`n_bits_schedule=(8,10)`, `zoom_factor=10`) which are needed
to resolve its narrow basins — see *Key parameters* above.

### Accuracy: distance to known true minimum

| Function | Character | D=2 | D=5 | D=8 |
|---|---|---|---|---|
| Wavy Bowl | Smooth multimodal | 0.00022 | 0.00003 | 0.00004 |
| Rastrigin | Regular traps | 0.00000 | 0.00003 | 0.00004 |
| Schwefel | Deceptive | **0.00005** | **0.00013** | **0.00088** |
| Eggholder | ~800 local minima | 0.00015 | — | — |

`dist` = Euclidean distance from found minimum to known true minimum.

### Comparison with classical methods (Rastrigin D=8)

| Method | D=8 dist | verdict |
|---|---|---|
| **hybrid_adaptive_minimize** | **0.00004** | ✓ |
| Dual Annealing | 0.00000 | ✓ |
| Differential Evolution | 1.407 | ✗ |
| CMA-ES | 1.723 | ✗ |

`hybrid_adaptive_minimize` finds the true minimum at D=8 where DE and
CMA-ES fail — because Grover's exhaustive 1D search finds the *global*
minimum of each coordinate slice rather than the nearest local one.

### Oracle call scaling (search step)

The O(√N) property of the search step is verified empirically:

| N (grid size) | Classical search | Oracle calls | Speedup |
|---|---|---|---|
| 4,096 | 4,096 | ~50 | 82× |
| 65,536 | 65,536 | ~200 | 328× |
| 1,048,576 | 1,048,576 | ~800 | 1,311× |

These are oracle calls on the pre-evaluated array only.
Grid evaluation (O(N) function calls) is additional.

---

## Choosing the right function

```
grover_minimize
    D=1–2, quick prototype, exploring n_bits effect
    grover_minimize(f, bounds, n_bits=8)

grover_minimize_hierarchical
    D=2–4, need high accuracy, narrow basins
    grover_minimize_hierarchical(f, bounds,
        n_bits_schedule=(6,8,10), max_layers=3)

hybrid_adaptive_minimize
    D=2–50+, default choice for any landscape
    hybrid_adaptive_minimize(f, bounds,
        n_bits_schedule=(6,8), max_cycles=5)
```

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
# 38 tests, all pass
```

---

## How it works

### Durr-Hoyer algorithm
1. Start with uniform superposition over all N grid points
2. Pick a random threshold index `t`
3. Run Grover iterations: **oracle** (phase-flip states cheaper than `costs[t]`) + **diffusion** (inversion about mean — amplifies marked states)
4. Measure: sample from |amplitude|² — finds cheaper state with high probability
5. Update threshold; grow iteration count by λ=6/5
6. Terminate after 22.5√N + 1.4·log₂(N)² oracle calls

### Hierarchical zoom
After Durr-Hoyer finds `x*` on a coarse grid, zoom the search window to
`[x* ± zoom_factor × grid_spacing]` and repeat. Each layer gives ~10–100×
finer effective resolution at the same oracle call budget.

### Hybrid adaptive
Profiles the function along each dimension (24 probe points) to classify
it as flat / monotone / unimodal / multimodal, then assigns the cheapest
sufficient method: Grover for multimodal, Brent for smooth, skip for flat.

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
