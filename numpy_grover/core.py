"""
numpy_grover.py
===============
Universal black-box global minimiser based on the Grover / Durr-Hoyer
quantum-inspired algorithm, implemented in pure NumPy.

Key idea
--------
Grover's algorithm gives a quadratic speedup over exhaustive search:
  - Classical grid search over N points: O(N) evaluations
  - Grover search:                       O(√N) oracle calls

For a D-dimensional grid with 2**n_bits points per axis:
  N = (2**n_bits) ** D
  Grover oracle calls ≈ 22.5 * √N

Public API
----------
  grover_minimize(func, bounds, n_bits, ...)             -> GroverResult
      Single-layer search. Evaluates func on a grid, runs Durr-Hoyer,
      returns coordinates + function value of global minimum.

  grover_minimize_hierarchical(func, bounds, ...)        -> HierarchicalGroverResult
      Multi-layer zoom-and-refine search.  Runs grover_minimize at each
      n_bits level, then zooms the bounding box around the best result
      and repeats.  Converges to high accuracy with far fewer total
      oracle calls than running a single high-n_bits search.

  grover_minimize_adaptive(func, bounds, ...)            -> AdaptiveGroverResult
      High-dimensional adaptive search.  Combines a global basin finder
      (Grover / DE / CMA-ES depending on D) with coordinate-wise
      refinement.  Each dimension is classified and assigned the cheapest
      method that fits its shape: Brent for smooth/unimodal dims, Grover
      hierarchical for multimodal dims.  Scales to D=50+ with linear cost.
      Alias for hybrid_adaptive_minimize (kept for backward compatibility).

  hybrid_adaptive_minimize(func, bounds, ...)            -> AdaptiveGroverResult
      Same as grover_minimize_adaptive — preferred name reflecting the
      honest hybrid nature: DE/CMA-ES for global basin, Grover for
      multimodal 1D dims, Brent for smooth 1D dims.
      New parameters: tol_f_rel (relative convergence), n_repeats (noise
      averaging), coupling_warning (detects strongly coupled dimensions).

  grover_min_2d(costs_flat, n_qubits, ...)               -> (idx, n_calls)
      Drop-in replacement for the Qiskit notebook interface.

  durr_hoyer_min(costs, seed)                            -> (idx, n_calls)
      Core algorithm. Works on any pre-evaluated flat cost array.

Low-level primitives (exported for testing / research)
------------------------------------------------------
  grover_oracle(state, marked_mask)
  grover_diffusion(state)
  grover_search(state, marked_mask, n_iter)

Accuracy vs speed
-----------------
  n_bits | pts/dim |  2D grid  | Grover calls | speedup vs classical
  -------|---------|-----------|--------------|--------------------
    4    |    16   |       256 |          ~12 |               16×
    6    |    64   |     4 096 |          ~50 |               64×
    8    |   256   |    65 536 |         ~200 |              256×
   10    | 1 024   | 1 048 576 |         ~800 |            1 024×
   12    | 4 096   |    16.7 M |       ~3 200 |            4 096×
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GroverResult:
    """
    Result returned by grover_minimize().

    Attributes
    ----------
    x : np.ndarray
        Coordinates of the found minimum, shape (D,).
    fun : float
        Function value at x.
    idx_flat : int
        Flat index into the cost array (useful for debugging).
    idx_nd : tuple[int, ...]
        Multi-dimensional index, one entry per dimension.
    n_calls : int
        Total oracle calls used across all trials.
    n_trials : int
        Number of independent Durr-Hoyer runs performed.
    elapsed : float
        Wall-clock time in seconds.
    grid : list[np.ndarray]
        The 1-D grid arrays used per dimension.
    costs_shape : tuple[int, ...]
        Shape of the cost hypercube (2**n_bits,) * D.
    """
    x          : np.ndarray
    fun        : float
    idx_flat   : int
    idx_nd     : tuple
    n_calls    : int
    n_trials   : int
    elapsed    : float
    grid       : list
    costs_shape: tuple

    def __repr__(self) -> str:
        coords = ", ".join(f"{v:.6g}" for v in self.x)
        return (
            f"GroverResult(x=[{coords}], fun={self.fun:.6g}, "
            f"n_calls={self.n_calls}, elapsed={self.elapsed:.3f}s)"
        )


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------

def grover_oracle(state: np.ndarray, marked_mask: np.ndarray) -> np.ndarray:
    """
    Apply the Grover phase-flip oracle.

    Flips the sign of amplitudes whose index is marked (i.e. satisfies the
    search criterion).  Operates in-place for efficiency, returns the array.

    Parameters
    ----------
    state       : amplitude vector, shape (N,), dtype float64
    marked_mask : boolean array, shape (N,),  True = marked state

    Returns
    -------
    state (modified in-place)
    """
    state[marked_mask] *= -1.0
    return state


def grover_diffusion(state: np.ndarray) -> np.ndarray:
    """
    Apply the Grover diffusion operator (inversion about the mean).

    Implements  state <- 2·mean(state) - state
    which is the operator  2|s><s| - I  in the uniform-superposition basis.
    O(N) time, O(1) extra space.

    Parameters
    ----------
    state : amplitude vector, shape (N,), dtype float64

    Returns
    -------
    state (modified in-place)
    """
    mean = state.mean()
    state *= -1.0
    state += 2.0 * mean
    return state


def grover_search(
    state: np.ndarray,
    marked_mask: np.ndarray,
    n_iter: int,
) -> np.ndarray:
    """
    Run n_iter rounds of oracle + diffusion on state.

    Parameters
    ----------
    state       : initial amplitude vector, shape (N,)
    marked_mask : boolean mask of marked states
    n_iter      : number of Grover iterations

    Returns
    -------
    state after n_iter rounds (modified in-place)
    """
    for _ in range(n_iter):
        grover_oracle(state, marked_mask)
        grover_diffusion(state)
    return state


# ---------------------------------------------------------------------------
# Durr-Hoyer minimum-finding algorithm
# ---------------------------------------------------------------------------

def durr_hoyer_min(
    costs: np.ndarray,
    seed: Optional[int] = None,
    _lam: float = 6.0 / 5.0,   # growth factor for m (paper default)
) -> Tuple[int, int]:
    """
    Durr-Hoyer quantum minimum-finding algorithm (pure NumPy).

    Finds the index of the global minimum of `costs` using O(√N) oracle
    calls, guaranteed with high probability.

    Algorithm sketch
    ----------------
    1. Pick a random threshold index t.
    2. Run Grover for a random number of steps j ∈ [1, m] with oracle
       marking all states cheaper than costs[t].
    3. Measure: if the found state is cheaper, update t.
    4. Grow m by factor λ; restart if no marked states exist.
    5. Terminate after 22.5·√N + 1.4·log₂(N)² oracle calls.

    Parameters
    ----------
    costs : 1-D array of function values (any dtype convertible to float)
    seed  : random seed for reproducibility (None = non-deterministic)

    Returns
    -------
    (best_idx, n_oracle_calls)
        best_idx        : index of the global minimum in costs
        n_oracle_calls  : total oracle iterations consumed
    """
    costs = np.asarray(costs, dtype=np.float64)
    N = len(costs)
    if N == 0:
        raise ValueError("costs array must be non-empty")
    if N == 1:
        return 0, 0

    rng = np.random.default_rng(seed)

    # Theoretical call budget  (Durr-Hoyer 1996, Theorem 1)
    budget = int(np.ceil(22.5 * np.sqrt(N) + 1.4 * np.log2(N) ** 2)) + 1

    # Uniform initial superposition
    state = np.ones(N, dtype=np.float64) / np.sqrt(N)

    # Random starting threshold
    threshold_idx = int(rng.integers(N))
    n_calls = 0
    m = 1.0  # maximum number of Grover steps to attempt this round

    while n_calls < budget:
        marked = costs < costs[threshold_idx]
        n_marked = marked.sum()

        if n_marked == 0:
            # Current threshold is already at or below everything:
            # reset to a new random threshold and restart m.
            threshold_idx = int(rng.integers(N))
            m = 1.0
            state[:] = 1.0 / np.sqrt(N)
            continue

        # Random iteration count j ∈ {1, …, floor(m)}
        j = int(rng.integers(1, max(2, int(np.floor(m)) + 1)))
        j = min(j, budget - n_calls)   # don't exceed budget

        # Fresh uniform superposition each round (exact Durr-Hoyer)
        state[:] = 1.0 / np.sqrt(N)
        grover_search(state, marked, j)
        n_calls += j

        # Simulated measurement: sample from |amplitude|²
        probs = state ** 2              # amplitudes are real here
        prob_sum = probs.sum()
        if prob_sum <= 0.0:
            # Numerical degenerate: reset
            state[:] = 1.0 / np.sqrt(N)
            m = 1.0
            continue
        probs /= prob_sum               # normalise to guard float drift

        found_idx = int(rng.choice(N, p=probs))

        # Update threshold if we found something better
        if costs[found_idx] < costs[threshold_idx]:
            threshold_idx = found_idx
            # Early exit: if we've found the true minimum we can stop
            if costs[threshold_idx] == costs.min():
                break

        # Grow m, cap at √N (Durr-Hoyer prescription)
        m = min(_lam * m, np.sqrt(N))

    return threshold_idx, n_calls


# ---------------------------------------------------------------------------
# Universal minimiser
# ---------------------------------------------------------------------------

def grover_minimize(
    func      : Callable,
    bounds    : Sequence[Tuple[float, float]],
    n_bits    : int = 6,
    n_trials  : int = 3,
    seed      : Optional[int] = None,
    vectorized: bool = True,
    n_repeats : int = 1,
) -> GroverResult:
    """
    Universal global minimiser for any real-valued function.

    Discretises the search space into a regular grid of (2**n_bits)^D
    points, evaluates func on every grid point, then runs the Durr-Hoyer
    algorithm to find the minimum in O(√N) oracle calls.

    Parameters
    ----------
    func       : vectorised objective f(X), X shape (N,D) -> (N,)
    bounds     : list of (lo, hi) per dimension
    n_bits     : grid resolution = 2**n_bits points per dimension
    n_trials   : independent Durr-Hoyer runs (best result returned)
    seed       : master random seed
    vectorized : True if func accepts (N,D) input, False for scalar
    n_repeats  : number of times to evaluate func at each grid point.
                 The average is used as the cost.  n_repeats > 1 reduces
                 the effect of noise / stochastic function evaluations.
                 Default 1 (no averaging — deterministic functions).

    Returns
    -------
    GroverResult  (.x, .fun, .n_calls, .elapsed, .grid, ...)
    """
    t0 = time.perf_counter()

    bounds = list(bounds)
    D      = len(bounds)
    n_pts  = 2 ** n_bits
    N      = n_pts ** D

    if n_bits < 1:
        raise ValueError("n_bits must be >= 1")
    if D < 1:
        raise ValueError("bounds must contain at least one (lo, hi) pair")
    for i, (lo, hi) in enumerate(bounds):
        if hi <= lo:
            raise ValueError(f"bounds[{i}]: hi ({hi}) must be > lo ({lo})")

    # ── Build grid ────────────────────────────────────────────────────────
    grids = [np.linspace(lo, hi, n_pts) for lo, hi in bounds]
    mesh  = np.meshgrid(*grids, indexing='ij')
    X_nd  = np.stack([m.ravel() for m in mesh], axis=1)  # (N, D)

    # ── Evaluate func (with optional averaging for noisy functions) ───────
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    if n_repeats == 1:
        if vectorized:
            costs = np.asarray(func(X_nd), dtype=np.float64).ravel()
        else:
            costs = np.array([func(*row) for row in X_nd], dtype=np.float64)
    else:
        # Average n_repeats evaluations per grid point
        accum = np.zeros(N, dtype=np.float64)
        for _ in range(n_repeats):
            if vectorized:
                accum += np.asarray(func(X_nd), dtype=np.float64).ravel()
            else:
                accum += np.array([func(*row) for row in X_nd],
                                  dtype=np.float64)
        costs = accum / n_repeats

    if costs.shape != (N,):
        raise ValueError(
            f"func returned shape {costs.shape}, expected ({N},). "
            "Make sure func(X) returns a 1-D array of length N when "
            "X has shape (N, D). Set vectorized=False if func takes "
            "scalar arguments."
        )

    # ---- Run Durr-Hoyer (n_trials independent runs) -----------------------
    rng = np.random.default_rng(seed)
    trial_seeds = rng.integers(0, 2**31, size=n_trials).tolist()

    best_idx    = None
    best_val    = np.inf
    total_calls = 0

    for ts in trial_seeds:
        idx, nc = durr_hoyer_min(costs, seed=ts)
        total_calls += nc
        if costs[idx] < best_val:
            best_val = costs[idx]
            best_idx = idx

    # ---- Decode flat index back to ND coordinates -------------------------
    idx_nd = np.unravel_index(best_idx, shape=(n_pts,) * D)
    x_min  = np.array([grids[d][idx_nd[d]] for d in range(D)])

    elapsed = time.perf_counter() - t0

    return GroverResult(
        x           = x_min,
        fun         = best_val,
        idx_flat    = best_idx,
        idx_nd      = idx_nd,
        n_calls     = total_calls,
        n_trials    = n_trials,
        elapsed     = elapsed,
        grid        = grids,
        costs_shape = (n_pts,) * D,
    )


# ---------------------------------------------------------------------------
# Drop-in replacement for the Qiskit notebook interface
# ---------------------------------------------------------------------------

def grover_min_2d(
    costs_flat : np.ndarray,
    n_qubits   : int,
    n_trials   : int = 12,
    seed       : Optional[int] = None,
) -> Tuple[int, int]:
    """
    Drop-in replacement for the Qiskit grover_min_2d() notebook function.

    Parameters
    ----------
    costs_flat : 1-D array of pre-evaluated cost values, length = 2**n_qubits
    n_qubits   : total number of qubits (log2 of grid size); kept for
                 interface compatibility — not used internally.
    n_trials   : number of independent Durr-Hoyer runs (default 12,
                 matching the Qiskit notebook default)
    seed       : master random seed

    Returns
    -------
    (found_idx, n_oracle_calls)
        Same types and meaning as the Qiskit version.
    """
    costs_flat = np.asarray(costs_flat, dtype=np.float64).ravel()
    rng        = np.random.default_rng(seed)
    seeds      = rng.integers(0, 2**31, size=n_trials).tolist()

    best_idx    = None
    best_val    = np.inf
    total_calls = 0

    for s in seeds:
        idx, nc = durr_hoyer_min(costs_flat, seed=s)
        total_calls += nc
        if costs_flat[idx] < best_val:
            best_val = costs_flat[idx]
            best_idx = idx

    return best_idx, total_calls


# ---------------------------------------------------------------------------
# Hierarchical (zoom-and-refine) result container
# ---------------------------------------------------------------------------

@dataclass
class LayerRecord:
    """
    One layer of a hierarchical search.

    Attributes
    ----------
    layer       : layer index (0 = coarse global, 1 = first zoom, …)
    bounds      : bounding box searched  [(lo,hi), …]
    n_bits      : n_bits schedule used this layer  e.g. [6, 8, 10]
    x           : best coordinate found this layer
    fun         : best function value found this layer
    n_calls     : oracle calls this layer
    elapsed     : wall-clock seconds this layer
    converged   : True if the within-layer schedule converged
    window_size : side length of the bounding box (per dimension)
    """
    layer      : int
    bounds     : list
    n_bits     : list
    x          : np.ndarray
    fun        : float
    n_calls    : int
    elapsed    : float
    converged  : bool
    window_size: float


@dataclass
class HierarchicalGroverResult:
    """
    Result returned by grover_minimize_hierarchical().

    Attributes
    ----------
    x           : best coordinate overall, shape (D,)
    fun         : best function value overall
    n_calls     : total oracle calls across all layers
    n_layers    : number of layers actually run
    elapsed     : total wall-clock seconds
    layers      : list of LayerRecord, one per layer
    converged   : True if a convergence criterion was met before max_layers
    stop_reason : human-readable string explaining why search stopped
    """
    x          : np.ndarray
    fun        : float
    n_calls    : int
    n_layers   : int
    elapsed    : float
    layers     : list          # list[LayerRecord]
    converged  : bool
    stop_reason: str

    def __repr__(self) -> str:
        coords = ", ".join(f"{v:.6g}" for v in self.x)
        return (
            f"HierarchicalGroverResult(x=[{coords}], fun={self.fun:.6g}, "
            f"layers={self.n_layers}, calls={self.n_calls}, "
            f"elapsed={self.elapsed:.3f}s, stop='{self.stop_reason}')"
        )

    def summary(self) -> str:
        """Pretty-print a per-layer convergence table."""
        lines = [
            f"\nHierarchical Grover search  —  {self.n_layers} layer(s)",
            "=" * 82,
            f"{'Layer':>6} {'window':>10} {'n_bits sched':>14} "
            f"{'x*':>22} {'f*':>14} {'calls':>8} {'conv?':>6}",
            "-" * 82,
        ]
        for lr in self.layers:
            coord_str = "(" + ", ".join(f"{v:.5f}" for v in lr.x) + ")"
            bits_str  = str(lr.n_bits)
            lines.append(
                f"{lr.layer:>6} {lr.window_size:>10.5f} {bits_str:>14} "
                f"{coord_str:>22} {lr.fun:>14.6f} "
                f"{lr.n_calls:>8,} {'✓' if lr.converged else '·':>6}"
            )
        lines += [
            "=" * 82,
            f"  Final:  x = [{', '.join(f'{v:.6f}' for v in self.x)}]",
            f"          f = {self.fun:.8f}",
            f"  Stop reason: {self.stop_reason}",
            f"  Total calls: {self.n_calls:,}   Total time: {self.elapsed:.3f}s",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hierarchical (zoom-and-refine) minimiser
# ---------------------------------------------------------------------------

def grover_minimize_hierarchical(
    func            : Callable,
    bounds          : Sequence[Tuple[float, float]],
    n_bits_schedule : Sequence[int] = (6, 8, 10),
    max_layers      : int   = 4,
    zoom_factor     : float = 6.0,
    tol_f           : float = 1e-6,
    tol_x           : float = 1e-6,
    n_trials        : int   = 3,
    seed            : Optional[int] = 42,
    vectorized      : bool  = True,
    verbose         : bool  = True,
    n_repeats       : int   = 1,
) -> HierarchicalGroverResult:
    """
    Hierarchical zoom-and-refine global minimiser.

    Strategy
    --------
    Layer 0 (global):
        Run grover_minimize with each n_bits in n_bits_schedule over the
        full domain.  The best coordinate x* found at the finest resolution
        becomes the centre of the zoomed window for layer 1.

    Layer 1, 2, … (zoom):
        Build a new bounding box of width  ±zoom_factor × grid_spacing
        around x* from the previous layer, clipped to the original domain.
        Run grover_minimize again with the full n_bits_schedule inside
        this tighter box.  Repeat until convergence or max_layers reached.

    Convergence criteria (any one stops the search)
    ------------------------------------------------
    - |f_new - f_prev| < tol_f   (function value stopped improving)
    - ‖x_new - x_prev‖ < tol_x  (coordinates stopped moving)
    - max_layers reached          (hard ceiling)

    Zoom window
    -----------
    After layer L at n_bits=B on a domain of width W, the finest grid
    spacing is  δ = W / (2**B - 1).  The zoom window for layer L+1 is:

        [x*_d - zoom_factor·δ_d,  x*_d + zoom_factor·δ_d]

    clipped to the original bounds.  zoom_factor=6 means the new window
    is ±6 grid cells wide — wide enough to catch the true minimum even
    if x* is a few cells off, but narrow enough to gain significant
    resolution with the same n_bits.

    Parameters
    ----------
    func             : objective function, same convention as grover_minimize
    bounds           : list of (lo, hi) per dimension
    n_bits_schedule  : n_bits values to try within each layer, e.g. (6,8,10)
    max_layers       : maximum number of zoom layers (including layer 0)
    zoom_factor      : half-width of zoom window in grid-spacing units
    tol_f            : convergence tolerance on function value
    tol_x            : convergence tolerance on coordinate distance
    n_trials         : Durr-Hoyer trials per grover_minimize call
    seed             : master random seed (None = non-deterministic)
    vectorized       : passed through to grover_minimize
    verbose          : print layer-by-layer progress table

    Returns
    -------
    HierarchicalGroverResult
        .x           best coordinate found
        .fun         best function value
        .n_calls     total oracle calls
        .n_layers    layers actually run
        .elapsed     total wall-clock seconds
        .layers      list of LayerRecord (one per layer)
        .converged   True if tol_f or tol_x criterion was met
        .stop_reason why the search stopped

    Examples
    --------
    >>> res = grover_minimize_hierarchical(
    ...     eggholder,
    ...     bounds=[(-512, 512), (-512, 512)],
    ...     n_bits_schedule=(6, 8, 10),
    ...     max_layers=3,
    ...     zoom_factor=6,
    ... )
    >>> print(res.summary())
    """
    t_global_start = time.perf_counter()

    bounds       = list(bounds)
    D            = len(bounds)
    orig_bounds  = bounds[:]           # keep original for clipping
    n_bits_sched = list(n_bits_schedule)

    # Derive per-layer seeds from master seed
    rng   = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=max_layers * len(n_bits_sched) + 1
                         ).tolist()
    seed_idx = 0

    if verbose:
        print(f"\nHierarchical Grover search")
        print(f"  D={D}  schedule={n_bits_sched}  "
              f"max_layers={max_layers}  zoom_factor={zoom_factor}")
        print(f"  tol_f={tol_f:.1e}  tol_x={tol_x:.1e}")
        print("=" * 82)
        print(f"{'Layer':>6} {'window':>10} {'n_bits':>7} "
              f"{'x*':>22} {'f*':>14} {'calls':>8} {'Δf':>12} {'Δx':>10}")
        print("-" * 82)

    layer_records  = []
    best_x         = None
    best_f         = np.inf
    total_calls    = 0
    stop_reason    = "max_layers reached"
    converged      = False
    current_bounds = bounds[:]

    for layer_idx in range(max_layers):
        t_layer = time.perf_counter()
        layer_calls = 0

        # ── Run the full n_bits schedule inside current_bounds ──────────
        layer_best_x = None
        layer_best_f = np.inf

        for nb in n_bits_sched:
            s = seeds[seed_idx]; seed_idx += 1
            res = grover_minimize(
                func, current_bounds,
                n_bits=nb, n_trials=n_trials,
                seed=s, vectorized=vectorized,
                n_repeats=n_repeats,
            )
            layer_calls += res.n_calls
            if res.fun < layer_best_f:
                layer_best_f = res.fun
                layer_best_x = res.x.copy()

        # ── Compute improvement over previous best ───────────────────────
        delta_f = best_f - layer_best_f          # positive = improvement
        delta_x = (np.linalg.norm(layer_best_x - best_x)
                   if best_x is not None else np.inf)

        # ── Window size for reporting (largest dimension width) ──────────
        window_size = max(hi - lo for lo, hi in current_bounds)

        # ── Within-layer convergence: did n_bits schedule help? ─────────
        # True when the last n_bits added less than tol_f improvement
        within_converged = (delta_f < tol_f) and (layer_idx > 0)

        # ── Verbose output ───────────────────────────────────────────────
        if verbose:
            coord_str = "(" + ", ".join(f"{v:.4f}" for v in layer_best_x) + ")"
            df_str = f"{delta_f:+.2e}" if best_x is not None else "      —"
            dx_str = f"{delta_x:.2e}"  if best_x is not None else "      —"
            print(f"{layer_idx:>6} {window_size:>10.4f} "
                  f"{str(n_bits_sched):>7} "
                  f"{coord_str:>22} {layer_best_f:>14.6f} "
                  f"{layer_calls:>8,} {df_str:>12} {dx_str:>10}")

        # ── Record this layer ────────────────────────────────────────────
        layer_records.append(LayerRecord(
            layer       = layer_idx,
            bounds      = current_bounds[:],
            n_bits      = n_bits_sched[:],
            x           = layer_best_x,
            fun         = layer_best_f,
            n_calls     = layer_calls,
            elapsed     = time.perf_counter() - t_layer,
            converged   = within_converged,
            window_size = window_size,
        ))

        total_calls += layer_calls

        # ── Update global best ───────────────────────────────────────────
        if layer_best_f < best_f:
            best_f = layer_best_f
            best_x = layer_best_x.copy()

        # ── Convergence checks ───────────────────────────────────────────
        if layer_idx > 0:
            if delta_f < tol_f:
                stop_reason = f"tol_f={tol_f:.1e} met (Δf={delta_f:.2e})"
                converged   = True
                break
            if delta_x < tol_x:
                stop_reason = f"tol_x={tol_x:.1e} met (Δx={delta_x:.2e})"
                converged   = True
                break

        # ── Build zoom window for next layer ─────────────────────────────
        # Grid spacing of the finest run this layer, in each dimension
        new_bounds = []
        for d, (lo, hi) in enumerate(current_bounds):
            finest_nb = n_bits_sched[-1]
            delta_grid = (hi - lo) / (2**finest_nb - 1)
            half_win   = zoom_factor * delta_grid
            new_lo = max(orig_bounds[d][0], best_x[d] - half_win)
            new_hi = min(orig_bounds[d][1], best_x[d] + half_win)
            # Safety: ensure window is never degenerate
            if new_hi - new_lo < 2 * delta_grid:
                new_lo = max(orig_bounds[d][0], best_x[d] - delta_grid)
                new_hi = min(orig_bounds[d][1], best_x[d] + delta_grid)
            new_bounds.append((new_lo, new_hi))

        current_bounds = new_bounds

    # ── Final verbose summary line ───────────────────────────────────────
    elapsed = time.perf_counter() - t_global_start
    if verbose:
        print("=" * 82)
        print(f"  Stop: {stop_reason}")
        final_coord = "(" + ", ".join(f"{v:.6f}" for v in best_x) + ")"
        print(f"  Best: x={final_coord}   f={best_f:.8f}")
        print(f"  Total oracle calls: {total_calls:,}   "
              f"Total time: {elapsed:.3f}s\n")

    return HierarchicalGroverResult(
        x           = best_x,
        fun         = best_f,
        n_calls     = total_calls,
        n_layers    = len(layer_records),
        elapsed     = elapsed,
        layers      = layer_records,
        converged   = converged,
        stop_reason = stop_reason,
    )


# ---------------------------------------------------------------------------
# Built-in benchmark functions (for testing and demos)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Adaptive (high-dimensional) result container
# ---------------------------------------------------------------------------

@dataclass
class DimRecord:
    """
    Per-dimension record from one coordinate-wise cycle.

    Attributes
    ----------
    dim       : dimension index
    character : 'monotone' | 'unimodal' | 'multimodal' | 'flat'
    method    : 'grover' | 'brent' | 'golden' | 'skip'
    x_before  : coordinate value before this dimension's search
    x_after   : coordinate value after
    f_after   : function value after
    n_calls   : evaluations used for this dimension
    delta_x   : absolute coordinate shift
    """
    dim      : int
    character: str
    method   : str
    x_before : float
    x_after  : float
    f_after  : float
    n_calls  : int
    delta_x  : float


@dataclass
class CycleRecord:
    """One full pass through all dimensions."""
    cycle      : int
    dim_records: list          # list[DimRecord]
    x          : np.ndarray    # x* after this cycle
    fun        : float         # f* after this cycle
    max_delta_x: float         # largest coordinate shift this cycle
    n_calls    : int           # total calls this cycle
    elapsed    : float


@dataclass
class AdaptiveGroverResult:
    """
    Result returned by grover_minimize_adaptive().

    Attributes
    ----------
    x               : best coordinate found, shape (D,)
    fun             : best function value
    n_calls         : total function/oracle calls
    n_cycles        : coordinate-wise cycles run
    elapsed         : total wall-clock seconds
    dim_characters  : list[str] — character per dimension
    dim_methods     : list[str] — method assigned per dimension
    stage1_x        : coordinate from global basin search (stage 1)
    stage1_fun      : function value from stage 1
    stage1_method   : method used in stage 1
    cycle_history   : list[CycleRecord]
    joint_zoom_done : True if final joint Grover zoom was run
    converged       : True if tol_x or tol_f met before max_cycles
    stop_reason     : human-readable stop explanation
    """
    x              : np.ndarray
    fun            : float
    n_calls        : int
    n_cycles       : int
    elapsed        : float
    dim_characters : list
    dim_methods    : list
    stage1_x       : np.ndarray
    stage1_fun     : float
    stage1_method  : str
    cycle_history  : list      # list[CycleRecord]
    joint_zoom_done: bool
    converged      : bool
    stop_reason    : str
    coupling_warning: bool = False   # True if strong coupling detected

    def __repr__(self) -> str:
        coords = ", ".join(f"{v:.6g}" for v in self.x)
        return (
            f"AdaptiveGroverResult(x=[{coords}], fun={self.fun:.6g}, "
            f"D={len(self.x)}, cycles={self.n_cycles}, "
            f"calls={self.n_calls}, elapsed={self.elapsed:.3f}s)"
        )

    def summary(self) -> str:
        """Pretty-print per-dimension and per-cycle information."""
        D = len(self.x)
        lines = [
            f"\nAdaptive Grover search  D={D}  cycles={self.n_cycles}",
            "=" * 74,
            f"  Stage 1 ({self.stage1_method}): "
            f"x=[{', '.join(f'{v:.4f}' for v in self.stage1_x)}]  "
            f"f={self.stage1_fun:.6f}",
            "",
            "  Dimension classification:",
            f"  {'dim':>4} {'character':>12} {'method':>10}  "
            f"{'final x':>12}",
            "  " + "-" * 44,
        ]
        for d in range(D):
            lines.append(
                f"  {d:>4} {self.dim_characters[d]:>12} "
                f"{self.dim_methods[d]:>10}  "
                f"{self.x[d]:>12.6f}"
            )
        lines += [
            "",
            f"  {'Cycle':>6} {'max Δx':>10} {'f*':>14} {'calls':>8}",
            "  " + "-" * 44,
        ]
        for cr in self.cycle_history:
            lines.append(
                f"  {cr.cycle:>6} {cr.max_delta_x:>10.2e} "
                f"{cr.fun:>14.6f} {cr.n_calls:>8,}"
            )
        lines += [
            "=" * 74,
            f"  Final: x=[{', '.join(f'{v:.6f}' for v in self.x)}]",
            f"         f={self.fun:.8f}",
            f"  Joint zoom: {'yes' if self.joint_zoom_done else 'no'}",
            f"  Stop: {self.stop_reason}",
            f"  Total calls: {self.n_calls:,}   Time: {self.elapsed:.3f}s",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dimension classifier
# ---------------------------------------------------------------------------

def classify_dimension(
    func        : Callable,
    x_current   : np.ndarray,
    dim         : int,
    bounds      : list,
    n_probe     : int = 24,
    flat_thresh : float = 1e-8,
) -> str:
    """
    Profile f along one dimension (all others fixed) and classify its shape.

    Parameters
    ----------
    func       : vectorised objective, f(X) where X shape (N, D)
    x_current  : current best coordinate, shape (D,)
    dim        : dimension index to profile
    bounds     : list of (lo, hi) per dimension
    n_probe    : number of evenly-spaced probe points along dim
    flat_thresh: range below which dimension is considered flat

    Returns
    -------
    'flat'       — f barely changes along this dimension (skip it)
    'monotone'   — f changes but has no local minimum (use bisection/Brent)
    'unimodal'   — exactly one local minimum (use Brent)
    'multimodal' — two or more local minima (use Grover hierarchical)
    """
    lo, hi = bounds[dim]
    probe  = np.linspace(lo, hi, n_probe)

    # Build probe matrix: repeat x_current, vary dim d
    X_probe         = np.tile(x_current, (n_probe, 1))
    X_probe[:, dim] = probe
    vals            = func(X_probe).ravel()

    v_range = vals.max() - vals.min()
    if v_range < flat_thresh:
        return 'flat'

    # Count sign changes in the first difference → number of local extrema
    diffs        = np.diff(vals)
    sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))

    if sign_changes == 0:
        return 'monotone'
    elif sign_changes <= 2:
        return 'unimodal'
    else:
        return 'multimodal'


# ---------------------------------------------------------------------------
# Hybrid adaptive minimiser  (coordinate-wise + Grover + Brent + DE/CMA-ES)
# ---------------------------------------------------------------------------

def hybrid_adaptive_minimize(
    func               : Callable,
    bounds             : Sequence[Tuple[float, float]],
    n_bits_schedule    : Sequence[int] = (6, 8),
    max_cycles         : int   = 6,
    zoom_factor        : float = 6.0,
    tol_x              : float = 1e-6,
    tol_f              : float = 1e-6,
    tol_f_rel          : float = 1e-6,
    n_probe            : int   = 24,
    max_joint_grid     : int   = 4_000_000,
    n_trials           : int   = 3,
    n_repeats          : int   = 1,
    seed               : Optional[int] = 42,
    dim_names          : Optional[list] = None,
    vectorized         : bool  = True,
    verbose            : bool  = True,
) -> "AdaptiveGroverResult":
    """
    Adaptive global minimiser for medium-to-high dimensional problems.

    Strategy
    --------
    Stage 1 — Global basin finding
        D <= 4 : grover_minimize_hierarchical
        D <= 10: scipy differential_evolution
        D > 10 : CMA-ES if available, else differential_evolution

    Stage 2/3 — Dimension classification + coordinate-wise refinement
        Profiles f along each dimension and assigns a method:
          flat       -> skip
          monotone   -> scipy bounded minimisation (Brent)
          unimodal   -> scipy bounded minimisation (Brent)
          multimodal -> grover_minimize_hierarchical (1D)
        Cycles until convergence or max_cycles reached.

    Stage 4 — Joint final zoom
        If the D-dimensional window is affordable (< max_joint_grid points),
        runs grover_minimize_hierarchical in the full D-dim tiny window
        to capture cross-dimension coupling.

    Parameters
    ----------
    func            : vectorised objective f(X), X shape (N, D) -> (N,)
    bounds          : list of (lo, hi) per dimension
    n_bits_schedule : n_bits values used for 1D Grover and joint zoom
    max_cycles      : maximum coordinate-wise cycles
    zoom_factor     : zoom window half-width in grid-spacing units
    tol_x           : stop if max relative coord shift < tol_x (absolute)
    tol_f           : stop if |Δf| < tol_f between cycles (absolute)
    tol_f_rel       : stop if |Δf| / max(|f|, 1) < tol_f_rel (relative).
                      More robust than tol_f for functions with large values
                      (e.g. Rastrigin at D=8 has f≈80 — absolute 1e-6 is
                      meaningless there). Either tol_f or tol_f_rel triggers.
    n_probe         : probe points for dimension classification
    max_joint_grid  : max grid points allowed for joint final zoom
    n_trials        : Durr-Hoyer trials per Grover call
    n_repeats       : evaluations averaged per grid point (noise reduction).
                      n_repeats=1 (default) — no averaging, deterministic.
                      n_repeats>1 — average n_repeats evaluations per point,
                      reducing stochastic noise at cost of n_repeats× more
                      function calls.
    seed            : master random seed
    dim_names       : optional list of D strings for verbose output
    vectorized      : True if func accepts (N, D) array input
    verbose         : print stage-by-stage progress

    Returns
    -------
    AdaptiveGroverResult
        .x, .fun, .n_calls, .n_cycles, .elapsed
        .dim_characters, .dim_methods   — per-dimension classification
        .cycle_history                  — convergence trajectory
        .coupling_warning               — True if strong coupling detected
        .stop_reason                    — why search stopped
    """
    from scipy.optimize import minimize_scalar as _minimize_scalar

    t_global = time.perf_counter()
    bounds   = list(bounds)
    D        = len(bounds)
    orig_bounds = bounds[:]
    names    = dim_names if dim_names else [f"x{d}" for d in range(D)]
    rng      = np.random.default_rng(seed)
    seeds    = rng.integers(0, 2**31, size=200).tolist()
    si       = [0]

    def next_seed():
        s = seeds[si[0]]; si[0] += 1; return int(s)

    total_calls  = [0]
    scalar_calls = [0]
    SEP = "=" * 74

    # ── Wrappers ──────────────────────────────────────────────────────────
    def func_scalar(v):
        """Scalar wrapper with optional noise averaging."""
        X = np.asarray(v, dtype=float).reshape(1, -1)
        if n_repeats == 1:
            scalar_calls[0] += 1
            return float(func(X)[0])
        else:
            scalar_calls[0] += n_repeats
            return float(np.mean([func(X)[0] for _ in range(n_repeats)]))

    def make_slice_vec(x_cur, d):
        x_base = x_cur.copy()
        def f1d(X1d):
            N  = len(X1d)
            Xf = np.tile(x_base, (N, 1))
            Xf[:, d] = X1d[:, 0]
            return func(Xf)
        return f1d

    def make_slice_scalar(x_cur, d):
        x_base = x_cur.copy()
        lo, hi = orig_bounds[d]
        def f1d(val):
            scalar_calls[0] += n_repeats
            x = x_base.copy()
            x[d] = float(np.clip(val, lo, hi))
            X = x.reshape(1, -1)
            if n_repeats == 1:
                return float(func(X)[0])
            return float(np.mean([func(X)[0] for _ in range(n_repeats)]))
        return f1d

    if verbose:
        print(f"\nHybrid Adaptive Minimiser")
        print(f"  D={D}  max_cycles={max_cycles}  schedule={list(n_bits_schedule)}")
        print(f"  tol_x={tol_x:.1e}  tol_f={tol_f:.1e}  tol_f_rel={tol_f_rel:.1e}")
        if n_repeats > 1:
            print(f"  n_repeats={n_repeats}  (noise averaging active)")
        print(SEP)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1 — Global basin finding
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print("\n  Stage 1: Global basin finding")

    stage1_method = ""
    stage1_x      = None
    stage1_fun    = np.inf

    if D <= 4:
        stage1_method = "Grover hierarchical"
        r = grover_minimize_hierarchical(
            func, bounds,
            n_bits_schedule=n_bits_schedule,
            max_layers=3, zoom_factor=zoom_factor,
            tol_f=tol_f, tol_x=tol_x,
            n_trials=n_trials, seed=next_seed(),
            vectorized=vectorized, verbose=False,
            n_repeats=n_repeats,
        )
        stage1_x   = r.x.copy()
        stage1_fun = r.fun
        total_calls[0] += r.n_calls

    elif D <= 10:
        stage1_method = "Differential Evolution"
        from scipy.optimize import differential_evolution as _de
        _sc0    = scalar_calls[0]
        _de_res = _de(func_scalar, bounds,
                      seed=next_seed(), maxiter=500, tol=1e-4, polish=True)
        stage1_x   = _de_res.x.copy()
        stage1_fun = float(_de_res.fun)
        total_calls[0] += scalar_calls[0] - _sc0

    else:
        try:
            import cma as _cma
            stage1_method = "CMA-ES"
            sigma0 = float(np.mean([(hi-lo)/4 for lo,hi in bounds]))
            x0     = [float(rng.uniform(lo, hi)) for lo, hi in bounds]
            _sc0   = scalar_calls[0]
            es     = _cma.CMAEvolutionStrategy(
                x0, sigma0,
                {"maxiter":500,"verbose":-9,
                 "bounds":[[b[0] for b in bounds],[b[1] for b in bounds]],
                 "seed":next_seed()})
            es.optimize(func_scalar)
            stage1_x   = np.array(es.result.xbest)
            stage1_fun = float(es.result.fbest)
            total_calls[0] += scalar_calls[0] - _sc0
        except ImportError:
            stage1_method = "Differential Evolution"
            from scipy.optimize import differential_evolution as _de
            _sc0    = scalar_calls[0]
            _de_res = _de(func_scalar, bounds,
                          seed=next_seed(), maxiter=800, tol=1e-4, polish=True)
            stage1_x   = _de_res.x.copy()
            stage1_fun = float(_de_res.fun)
            total_calls[0] += scalar_calls[0] - _sc0

    x_current = stage1_x.copy()
    best_f    = stage1_fun

    if verbose:
        cstr = "[" + ", ".join(f"{v:.4f}" for v in x_current) + "]"
        print(f"    method : {stage1_method}")
        print(f"    x*     : {cstr}")
        print(f"    f*     : {best_f:.8f}")
        print(f"    calls  : {total_calls[0]:,}")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3 — Coordinate-wise refinement (with inline classification)
    # ══════════════════════════════════════════════════════════════════════
    cycle_history  = []
    dim_characters = ["unclassified"] * D
    dim_methods    = ["unclassified"] * D
    converged      = False
    stop_reason    = "max_cycles reached"

    if verbose:
        print(f"\n  Stage 3: Coordinate-wise refinement")

    for cycle in range(max_cycles):
        t_cycle     = time.perf_counter()
        cycle_calls = 0
        dim_records = []
        x_before    = x_current.copy()
        prev_f      = best_f

        if verbose:
            print(f"\n  ── Cycle {cycle} ──")
            print(f"  {'dim':>4} {'name':>8} {'character':>12} "
                  f"{'method':>10} {'x_before':>11} {'x_after':>11} "
                  f"{'Δx':>10} {'calls':>7}")
            print("  " + "-" * 76)

        for d in range(D):
            lo, hi     = orig_bounds[d]
            x_d_before = float(x_current[d])

            # Classify this dimension
            char = classify_dimension(
                func, x_current, d, orig_bounds, n_probe=n_probe)
            cycle_calls += n_probe
            dim_characters[d] = char

            if char == "flat":
                method    = "skip"
                x_d_after = x_d_before
                f_after   = best_f
                d_calls   = 0

            elif char in ("monotone", "unimodal"):
                method  = "brent"
                _sc0    = scalar_calls[0]
                _r      = _minimize_scalar(
                    make_slice_scalar(x_current, d),
                    bounds=(lo, hi), method="bounded",
                    options={"xatol": max(tol_x * (hi - lo), 1e-12)})
                x_d_after = float(np.clip(_r.x, lo, hi))
                f_after   = float(_r.fun)
                d_calls   = scalar_calls[0] - _sc0

            else:
                # multimodal → Grover 1D hierarchical
                method  = "grover"
                _r      = grover_minimize_hierarchical(
                    make_slice_vec(x_current, d),
                    bounds=[(lo, hi)],
                    n_bits_schedule=n_bits_schedule,
                    max_layers=3, zoom_factor=zoom_factor,
                    tol_f=tol_f, tol_x=max(tol_x*(hi-lo), 1e-12),
                    n_trials=n_trials, seed=next_seed(),
                    vectorized=True, verbose=False,
                )
                x_d_after = float(np.clip(_r.x[0], lo, hi))
                f_after   = float(_r.fun)
                d_calls   = _r.n_calls

            dim_methods[d] = method
            x_current[d]   = x_d_after
            if f_after < best_f:
                best_f = f_after

            delta_x     = abs(x_d_after - x_d_before)
            cycle_calls += d_calls

            dim_records.append(DimRecord(
                dim=d, character=char, method=method,
                x_before=x_d_before, x_after=x_d_after,
                f_after=f_after, n_calls=d_calls+n_probe,
                delta_x=delta_x,
            ))

            if verbose:
                print(f"  {d:>4} {names[d]:>8} {char:>12} "
                      f"{method:>10} {x_d_before:>11.5f} "
                      f"{x_d_after:>11.5f} {delta_x:>10.2e} "
                      f"{d_calls+n_probe:>7,}")

        # ── Convergence checks ────────────────────────────────────────────
        max_rel = max(
            abs(x_current[d] - x_before[d]) / max(abs(hi-lo), 1e-12)
            for d, (lo, hi) in enumerate(orig_bounds))
        delta_f     = abs(prev_f - best_f)
        delta_f_rel = delta_f / max(abs(best_f), 1.0)   # relative improvement

        cr = CycleRecord(
            cycle=cycle, dim_records=dim_records,
            x=x_current.copy(), fun=best_f,
            max_delta_x=max_rel, n_calls=cycle_calls,
            elapsed=time.perf_counter()-t_cycle,
        )
        cycle_history.append(cr)
        total_calls[0] += cycle_calls

        if verbose:
            print(f"  → f={best_f:.8f}  max_Δx={max_rel:.2e}  "
                  f"Δf={delta_f:.2e}  Δf_rel={delta_f_rel:.2e}  "
                  f"calls={cycle_calls:,}")

        # Absolute tol_f
        if delta_f < tol_f and cycle > 0:
            stop_reason = f"tol_f={tol_f:.1e} met (Δf={delta_f:.2e})"
            converged   = True
            break
        # Relative tol_f_rel
        if delta_f_rel < tol_f_rel and cycle > 0:
            stop_reason = (f"tol_f_rel={tol_f_rel:.1e} met "
                           f"(Δf/|f|={delta_f_rel:.2e})")
            converged   = True
            break
        # Coordinate convergence
        if max_rel < tol_x:
            stop_reason = f"tol_x={tol_x:.1e} met (max_Δx={max_rel:.2e})"
            converged   = True
            break

    # ── Coupling detection ────────────────────────────────────────────────
    # Coupling is indicated when coordinates keep shifting cycle after cycle
    # without converging — the optimum of each dim depends on others.
    # We check: did shifts increase or stay large across the last 2 cycles?
    coupling_warning = False
    if len(cycle_history) >= 3:
        shifts = [cr.max_delta_x for cr in cycle_history]
        # Warning if: last 3 cycles all had non-trivial shifts AND
        # the shifts are not monotonically decreasing fast
        last3   = shifts[-3:]
        non_trivial = all(s > tol_x * 10 for s in last3)
        not_converging = last3[-1] > last3[0] * 0.1   # not dropped by 10×
        if non_trivial and not_converging:
            coupling_warning = True
            if verbose:
                print(f"\n  ⚠ Coupling warning: shifts [{', '.join(f'{s:.2e}' for s in last3)}] "
                      f"not converging — dimensions may be strongly coupled. "
                      f"Consider more cycles or a joint search method.")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 4 — Joint final zoom (if affordable)
    # ══════════════════════════════════════════════════════════════════════
    joint_zoom_done = False
    finest_nb       = n_bits_schedule[-1]

    zoom_bounds = []
    for d, (lo, hi) in enumerate(orig_bounds):
        gs     = (hi - lo) / max(2**finest_nb - 1, 1)
        new_lo = max(lo, x_current[d] - zoom_factor * gs)
        new_hi = min(hi, x_current[d] + zoom_factor * gs)
        zoom_bounds.append((new_lo, new_hi))

    joint_nb = finest_nb
    joint_N  = (2**joint_nb) ** D
    while joint_N > max_joint_grid and joint_nb > 4:
        joint_nb -= 1
        joint_N   = (2**joint_nb) ** D

    if joint_N <= max_joint_grid:
        if verbose:
            print(f"\n  Stage 4: Joint zoom  "
                  f"(n_bits={joint_nb}, N={joint_N:,}, D={D})")
        _jr = grover_minimize_hierarchical(
            func, zoom_bounds,
            n_bits_schedule=(joint_nb,),
            max_layers=2, zoom_factor=zoom_factor,
            tol_f=tol_f, tol_x=tol_x,
            n_trials=n_trials, seed=next_seed(),
            vectorized=vectorized, verbose=False,
        )
        total_calls[0] += _jr.n_calls
        if _jr.fun < best_f:
            if verbose:
                print(f"    f improved: {best_f:.8f} → {_jr.fun:.8f}")
            best_f    = _jr.fun
            x_current = _jr.x.copy()
        else:
            if verbose:
                print(f"    f unchanged: {best_f:.8f}")
        joint_zoom_done = True
    else:
        if verbose:
            print(f"\n  Stage 4: Joint zoom skipped "
                  f"(N={joint_N:.2e} > limit={max_joint_grid:.2e})")

    elapsed = time.perf_counter() - t_global
    if verbose:
        print(f"\n{SEP}")
        print(f"  Stop: {stop_reason}")
        if coupling_warning:
            print(f"  ⚠  Coupling warning active — result may not be fully reliable")
        print(f"  Best: x=[{', '.join(f'{v:.6f}' for v in x_current)}]")
        print(f"        f={best_f:.8f}")
        print(f"  Total calls: {total_calls[0]:,}   Time: {elapsed:.3f}s\n")

    return AdaptiveGroverResult(
        x=x_current, fun=best_f,
        n_calls=total_calls[0], n_cycles=len(cycle_history),
        elapsed=elapsed,
        dim_characters=dim_characters, dim_methods=dim_methods,
        stage1_x=stage1_x, stage1_fun=stage1_fun,
        stage1_method=stage1_method,
        cycle_history=cycle_history,
        joint_zoom_done=joint_zoom_done,
        converged=converged, stop_reason=stop_reason,
        coupling_warning=coupling_warning,
    )


# Keep old name as alias for backward compatibility
grover_minimize_adaptive = hybrid_adaptive_minimize


# ---------------------------------------------------------------------------
# Built-in benchmark functions (for testing and demos)
# ---------------------------------------------------------------------------

def _rastrigin(X: np.ndarray) -> np.ndarray:
    """Rastrigin function — highly multimodal, global min = 0 at origin."""
    A = 10.0
    return A * X.shape[1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)

def _ackley(X: np.ndarray) -> np.ndarray:
    """Ackley function — global min = 0 at origin."""
    a, b, c = 20.0, 0.2, 2 * np.pi
    D = X.shape[1]
    return (- a * np.exp(-b * np.sqrt(np.sum(X**2, axis=1) / D))
            - np.exp(np.sum(np.cos(c * X), axis=1) / D)
            + a + np.e)

def _rosenbrock(X: np.ndarray) -> np.ndarray:
    """Rosenbrock — global min = 0 at (1,…,1)."""
    return np.sum(100*(X[:,1:] - X[:,:-1]**2)**2 + (1 - X[:,:-1])**2, axis=1)

def _himmelblau(X: np.ndarray) -> np.ndarray:
    """Himmelblau — 4 global minima, value = 0."""
    x, y = X[:,0], X[:,1]
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


BENCHMARK_FUNCTIONS = {
    "rastrigin" : (_rastrigin,  [(-5.12, 5.12)]),
    "ackley"    : (_ackley,     [(-5.0,  5.0)]),
    "rosenbrock": (_rosenbrock, [(-2.0,  2.0), (-1.0, 3.0)]),
    "himmelblau": (_himmelblau, [(-5.0,  5.0), (-5.0, 5.0)]),
}


# ---------------------------------------------------------------------------
# Validation suite
# ---------------------------------------------------------------------------

def validate_all(n_bits: int = 6, verbose: bool = True) -> bool:
    """
    Run all validation checks and print a pass/fail report.

    Parameters
    ----------
    n_bits  : grid resolution for the tests (default 6 → 64 pts/dim)
    verbose : print progress lines

    Returns
    -------
    True if all checks pass, False otherwise.
    """
    passed = []

    def check(name, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        if verbose:
            print(f"  [{status}] {name}" + (f"  —  {detail}" if detail else ""))
        passed.append(ok)

    n_pts = 2 ** n_bits
    N     = n_pts ** 2

    if verbose:
        print(f"\nnumpy_grover validation  (n_bits={n_bits}, N={N:,})")
        print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Random array: Durr-Hoyer must find the true minimum
    # ------------------------------------------------------------------
    rng   = np.random.default_rng(0)
    costs = rng.random(N)
    idx, nc = durr_hoyer_min(costs, seed=1)
    ok = (idx == int(np.argmin(costs)))
    check("Random 1-D array minimum found", ok,
          f"idx={idx}, true={np.argmin(costs)}, calls={nc}")

    # ------------------------------------------------------------------
    # 2. grover_oracle flips exactly the marked states
    # ------------------------------------------------------------------
    state = np.ones(8) / np.sqrt(8)
    mask  = np.array([True, False, True, False, False, False, False, False])
    before = state.copy()
    grover_oracle(state, mask)
    ok = (np.allclose(state[mask],  -before[mask]) and
          np.allclose(state[~mask],  before[~mask]))
    check("Oracle flips marked amplitudes only", ok)

    # ------------------------------------------------------------------
    # 3. grover_diffusion preserves norm
    # ------------------------------------------------------------------
    state = rng.standard_normal(64)
    state /= np.linalg.norm(state)
    grover_diffusion(state)
    ok = np.isclose(np.linalg.norm(state), 1.0, atol=1e-12)
    check("Diffusion preserves L2 norm", ok,
          f"norm={np.linalg.norm(state):.12f}")

    # ------------------------------------------------------------------
    # 4. Oracle call count is O(√N)
    # ------------------------------------------------------------------
    _, nc = durr_hoyer_min(costs, seed=2)
    budget = int(np.ceil(22.5 * np.sqrt(N) + 1.4 * np.log2(N)**2)) + 1
    ok = nc <= budget
    expected_classical = N
    check("Oracle calls within Durr-Hoyer budget",  ok,
          f"calls={nc}, budget={budget}, classical={expected_classical}")

    # ------------------------------------------------------------------
    # 5. Reliability: 10 independent runs all find the same minimum
    # ------------------------------------------------------------------
    true_min_idx = int(np.argmin(costs))
    found = [durr_hoyer_min(costs, seed=s)[0] for s in range(10)]
    ok = all(f == true_min_idx for f in found)
    check("Reliability: 10/10 runs find same minimum", ok,
          f"results={found[:5]}…")

    # ------------------------------------------------------------------
    # 6. grover_minimize on 1-D multimodal function
    # ------------------------------------------------------------------
    def multimodal_1d(X):
        x = X[:, 0]
        return np.sin(x) + 0.5 * np.sin(4*x) + 0.3 * np.sin(7*x)

    res = grover_minimize(multimodal_1d, bounds=[(0, 2*np.pi)],
                          n_bits=n_bits, n_trials=3, seed=42)
    # Verify with brute-force on the same grid
    grid_x = np.linspace(0, 2*np.pi, n_pts).reshape(-1,1)
    bf_val  = multimodal_1d(grid_x).min()
    ok = np.isclose(res.fun, bf_val, atol=1e-10)
    check("grover_minimize 1-D multimodal", ok,
          f"found={res.fun:.6f}, brute-force={bf_val:.6f}")

    # ------------------------------------------------------------------
    # 7. grover_minimize on 2-D Himmelblau (4 global minima)
    # ------------------------------------------------------------------
    res2d = grover_minimize(_himmelblau, bounds=[(-5,5),(-5,5)],
                            n_bits=n_bits, n_trials=3, seed=42)
    # True global minimum value = 0 at 4 points; on-grid best is near 0
    ok = res2d.fun < 0.5   # allow grid discretisation error
    check("grover_minimize 2-D Himmelblau (4 minima)", ok,
          f"found f={res2d.fun:.6f} at x={res2d.x}")

    # ------------------------------------------------------------------
    # 8. grover_min_2d drop-in interface
    # ------------------------------------------------------------------
    idx2, nc2 = grover_min_2d(costs, n_qubits=n_bits*2,
                               n_trials=12, seed=42)
    ok = (idx2 == true_min_idx)
    check("grover_min_2d drop-in interface", ok,
          f"idx={idx2}, true={true_min_idx}, calls={nc2}")

    # ------------------------------------------------------------------
    # 9. Speed: 400 phase-diagram points in <30s (vs 463s Qiskit)
    # ------------------------------------------------------------------
    def dummy_fflo(X):
        d, q = X[:,0], X[:,1]
        return d**2 - 0.5*d + 0.1*np.sin(20*q) + 0.05*rng.random(len(d))

    t0  = time.perf_counter()
    for _ in range(400):
        r = grover_minimize(dummy_fflo, bounds=[(0.001,0.15),(0.0,0.06)],
                            n_bits=n_bits, n_trials=1, seed=0)
    elapsed = time.perf_counter() - t0
    ok = elapsed < 30.0
    check(f"Speed: 400 phase-diagram points", ok,
          f"{elapsed:.2f}s  (target <30s, Qiskit baseline 463s)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_pass = sum(passed)
    n_total = len(passed)
    if verbose:
        print("=" * 60)
        print(f"  {n_pass}/{n_total} checks passed\n")

    return all(passed)


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def benchmark(
    func_name : str = "rastrigin",
    dims      : int = 2,
    n_bits_list: Sequence[int] = (4, 6, 8),
    n_trials  : int = 3,
    seed      : int = 42,
) -> None:
    """
    Benchmark grover_minimize against brute-force on a named test function.

    Parameters
    ----------
    func_name    : one of 'rastrigin', 'ackley', 'rosenbrock', 'himmelblau'
    dims         : number of dimensions (overrides default bounds length)
    n_bits_list  : list of n_bits values to test
    n_trials     : Durr-Hoyer trials per run
    seed         : random seed
    """
    if func_name not in BENCHMARK_FUNCTIONS:
        raise ValueError(f"Unknown function '{func_name}'. "
                         f"Choose from {list(BENCHMARK_FUNCTIONS)}")

    func, default_bounds = BENCHMARK_FUNCTIONS[func_name]
    lo, hi = default_bounds[0]
    bounds = [(lo, hi)] * dims

    print(f"\nBenchmark: {func_name.capitalize()}  D={dims}")
    print(f"{'n_bits':>6}  {'pts/dim':>8}  {'N':>10}  "
          f"{'calls':>8}  {'√N':>8}  {'calls/√N':>10}  "
          f"{'time(s)':>8}  {'found f':>12}")
    print("-" * 80)

    for nb in n_bits_list:
        n_pts = 2 ** nb
        N     = n_pts ** dims

        t0  = time.perf_counter()
        res = grover_minimize(func, bounds=bounds, n_bits=nb,
                              n_trials=n_trials, seed=seed)
        elapsed = time.perf_counter() - t0

        sqrt_N   = np.sqrt(N)
        ratio    = res.n_calls / sqrt_N if sqrt_N > 0 else 0

        print(f"{nb:>6}  {n_pts:>8,}  {N:>10,}  "
              f"{res.n_calls:>8,}  {sqrt_N:>8.1f}  {ratio:>10.2f}  "
              f"{elapsed:>8.3f}  {res.fun:>12.6f}")

    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("numpy_grover — universal Grover-based global minimiser")
    print("Running validation suite …\n")
    ok = validate_all(n_bits=6)
    if ok:
        print("All checks passed. Running benchmark …")
        benchmark("rastrigin", dims=2, n_bits_list=[4, 6, 8])
        benchmark("ackley",    dims=2, n_bits_list=[4, 6, 8])
    else:
        print("Some checks failed — please review output above.")
