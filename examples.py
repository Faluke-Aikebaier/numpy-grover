"""
examples.py
===========
Usage examples for the numpy_grover library.

Demonstrates all four public functions plus BenchmarkSuite:

    1. grover_minimize              — single-layer grid search
    2. grover_minimize_hierarchical — zoom-and-refine for high accuracy
    3. hybrid_adaptive_minimize     — coordinate-wise for D=2 to D=50+
    4. hybrid_adaptive_minimize     — with adaptive_n_bits=True
    5. BenchmarkSuite               — built-in test functions
    6. Custom function template

Run:
    python examples.py

Author: Faluke Aikebaier
"""

import numpy as np

from numpy_grover import (
    grover_minimize,
    grover_minimize_hierarchical,
    hybrid_adaptive_minimize,
    BenchmarkSuite,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared test functions
# ─────────────────────────────────────────────────────────────────────────────

def wavy_bowl(X):
    """
    f(x,y) = x² + y² + 0.4·sin(5x)·sin(5y)

    Smooth quadratic bowl with sinusoidal ripples.
    ~25 local minima; true global minimum near (0.259, -0.259).
    """
    x, y = X[:, 0], X[:, 1]
    return x**2 + y**2 + 0.4 * np.sin(5*x) * np.sin(5*y)


def rastrigin(X):
    """
    f(x) = 10D + Σ [xᵢ² - 10·cos(2π·xᵢ)]

    Highly multimodal; ~10^D local minima.
    True global minimum = 0 at the origin.
    """
    A = 10.0
    return A * X.shape[1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)


def schwefel(X):
    """
    f(x) = 418.9829·D - Σ xᵢ·sin(√|xᵢ|)

    Deceptive: second-best basin is far from the global minimum.
    True global minimum ≈ 0 at (420.969, …).
    """
    D = X.shape[1]
    return 418.9829 * D - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)


def styblinski_tang(X):
    """
    f(x) = 0.5 · Σ [xᵢ⁴ - 16xᵢ² + 5xᵢ]

    Near-symmetric double wells per axis.
    True global minimum = -78.332 at (-2.904, -2.904).
    """
    return 0.5 * np.sum(X**4 - 16*X**2 + 5*X, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Example 1 — grover_minimize
# ─────────────────────────────────────────────────────────────────────────────

def example_grover_minimize():
    """
    Single-layer Grover search.

    Best for:
      - 2D problems where you want a quick result
      - Exploring what resolution (n_bits) you need
      - Understanding the raw Grover / Durr-Hoyer algorithm

    n_bits controls the grid resolution:
      n_bits=6  →  64 pts/dim  →  4,096 total points
      n_bits=8  →  256 pts/dim →  65,536 total points
      n_bits=10 →  1024 pts/dim → 1,048,576 total points
    """
    print("=" * 60)
    print("EXAMPLE 1: grover_minimize")
    print("  Wavy Bowl — 2D multimodal function")
    print("  True minimum ≈ (0.259, -0.259)  f ≈ -0.2362")
    print("=" * 60)

    for n_bits in [6, 8, 10]:
        res = grover_minimize(
            func      = wavy_bowl,
            bounds    = [(-2.0, 2.0), (-2.0, 2.0)],
            n_bits    = n_bits,
            n_trials  = 3,
            seed      = 42,
        )
        true = np.array([0.25942, -0.25942])
        dist = np.linalg.norm(res.x - true)
        print(f"  n_bits={n_bits:>2}  grid={2**n_bits}×{2**n_bits}"
              f"  x=({res.x[0]:.4f}, {res.x[1]:.4f})"
              f"  f={res.fun:.6f}  dist={dist:.5f}"
              f"  calls={res.n_calls:,}")

    print()
    print("  Key fields on the result:")
    res = grover_minimize(wavy_bowl, [(-2,2),(-2,2)], n_bits=8, seed=42)
    print(f"    res.x          = {res.x}          (coordinates)")
    print(f"    res.fun        = {res.fun:.6f}       (function value)")
    print(f"    res.n_calls    = {res.n_calls}              (oracle calls)")
    print(f"    res.elapsed    = {res.elapsed:.3f}s            (wall clock)")
    print(f"    res.grid       = [array of {len(res.grid[0])} pts, ...]  (grid per dim)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Example 2 — grover_minimize_hierarchical
# ─────────────────────────────────────────────────────────────────────────────

def example_grover_minimize_hierarchical():
    """
    Hierarchical zoom-and-refine search.

    Best for:
      - When you need high accuracy (sub-millimetre coordinates)
      - 2D-4D problems with narrow basins

    Strategy:
      Layer 0: search full domain → find the basin
      Layer 1: zoom into a small window around x* → refine
      Layer 2: zoom again → refine further
      Stop when: |Δf| < tol_f  or  |Δx| < tol_x  or  max_layers reached
    """
    print("=" * 60)
    print("EXAMPLE 2: grover_minimize_hierarchical")
    print("  Wavy Bowl — comparing single layer vs hierarchical")
    print("  True minimum ≈ (0.259, -0.259)  f ≈ -0.23618")
    print("=" * 60)

    true_xy  = np.array([0.25942, -0.25942])
    true_val = -0.236179

    # Single layer for comparison
    r_single = grover_minimize(
        wavy_bowl, [(-2, 2), (-2, 2)], n_bits=6, seed=42)
    print(f"  Single layer (n_bits=6):")
    print(f"    f={r_single.fun:.6f}  dist={np.linalg.norm(r_single.x-true_xy):.5f}")

    # Hierarchical — much better
    r_hier = grover_minimize_hierarchical(
        func            = wavy_bowl,
        bounds          = [(-2.0, 2.0), (-2.0, 2.0)],
        n_bits_schedule = (6, 8),
        max_layers      = 3,
        zoom_factor     = 6.0,
        tol_f           = 1e-7,
        tol_x           = 1e-7,
        n_trials        = 3,
        seed            = 42,
        verbose         = False,
    )

    print(f"\n  Hierarchical (3 layers, schedule=[6,8]):")
    for lr in r_hier.layers:
        dist = np.linalg.norm(lr.x - true_xy)
        print(f"    Layer {lr.layer}: window={lr.window_size:.4f}"
              f"  f={lr.fun:.6f}  dist={dist:.5f}")

    print(f"\n  Final: f={r_hier.fun:.8f}"
          f"  dist={np.linalg.norm(r_hier.x-true_xy):.6f}")
    print(f"  Stop reason: {r_hier.stop_reason}")
    print(f"  Total oracle calls: {r_hier.n_calls:,}")
    print()
    print(r_hier.summary())
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Example 3 — hybrid_adaptive_minimize
# ─────────────────────────────────────────────────────────────────────────────

def example_hybrid_adaptive():
    """
    Hybrid adaptive coordinate-wise minimiser.

    Best for:
      - D=2 to D=50+ problems
      - Functions where different dimensions have different characters
        (some multimodal, some smooth, some flat)
      - When you want the library to figure out the best method per dim

    The function:
      1. Finds the global basin (Grover for D≤4, DE for D≤10, CMA-ES for D>10)
      2. Classifies each dimension: multimodal / unimodal / monotone / flat
      3. Optimises each dimension with the cheapest sufficient method:
            multimodal → Grover hierarchical (exhaustive, finds global 1D min)
            unimodal   → Brent (fast, ~15 evals)
            monotone   → Brent
            flat       → skip
      4. Repeats until convergence
      5. Optional joint zoom in a small D-dim window around the result
    """
    print("=" * 60)
    print("EXAMPLE 3a: hybrid_adaptive_minimize — 2D Wavy Bowl")
    print("=" * 60)

    res = hybrid_adaptive_minimize(
        func             = wavy_bowl,
        bounds           = [(-2.0, 2.0), (-2.0, 2.0)],
        n_bits_schedule  = (6, 8),
        max_cycles       = 4,
        zoom_factor      = 6.0,
        tol_f            = 1e-8,
        tol_f_rel        = 1e-6,     # relative tolerance — scale-independent
        n_repeats        = 1,        # 1 = no averaging (deterministic f)
        n_trials         = 3,
        seed             = 42,
        verbose          = False,
    )

    true = np.array([0.25942, -0.25942])
    print(f"  x           = {res.x}")
    print(f"  f           = {res.fun:.8f}")
    print(f"  dist        = {np.linalg.norm(res.x - true):.6f}")
    print(f"  n_cycles    = {res.n_cycles}")
    print(f"  n_calls     = {res.n_calls:,}")
    print(f"  stage1      = {res.stage1_method}")
    print(f"  dim chars   = {res.dim_characters}")
    print(f"  dim methods = {res.dim_methods}")
    print(f"  coupling    = {res.coupling_warning}")
    print(f"  stop        = {res.stop_reason}")
    print()

    # Full summary table
    print(res.summary())

    # ── 2D Rastrigin ─────────────────────────────────────────────────────
    print("=" * 60)
    print("EXAMPLE 3b: hybrid_adaptive_minimize — 2D Rastrigin")
    print("  All dims multimodal → Grover used for all")
    print("  True minimum = 0 at origin")
    print("=" * 60)

    res2 = hybrid_adaptive_minimize(
        func            = rastrigin,
        bounds          = [(-5.12, 5.12)] * 2,
        n_bits_schedule = (6, 8),
        max_cycles      = 3,
        seed            = 42,
        verbose         = False,
    )
    true2 = np.zeros(2)
    print(f"  x        = {np.round(res2.x, 5)}")
    print(f"  f        = {res2.fun:.8f}  (true = 0.0)")
    print(f"  dist     = {np.linalg.norm(res2.x - true2):.6f}")
    print(f"  chars    = {res2.dim_characters}")
    print()

    # ── Noisy function with n_repeats ────────────────────────────────────
    print("=" * 60)
    print("EXAMPLE 3c: n_repeats — noise-aware oracle")
    print("  Rastrigin + Gaussian noise (std=1.0)")
    print("=" * 60)

    _call = [0]
    def rastrigin_noisy(X):
        _call[0] += 1
        return rastrigin(X) + np.random.RandomState(_call[0]).normal(0, 1.0, len(X))

    for nr in [1, 5]:
        _call[0] = 0
        r = hybrid_adaptive_minimize(
            rastrigin_noisy, [(-5.12, 5.12)] * 2,
            n_bits_schedule=(6, 8), max_cycles=2,
            n_repeats=nr, seed=42, verbose=False)
        dist = np.linalg.norm(r.x - np.zeros(2))
        print(f"  n_repeats={nr:>2}: dist={dist:.5f}  f={r.fun:.4f}"
              f"  calls={r.n_calls:,}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Example 4 — adaptive_n_bits: auto-tune resolution per dimension
# ─────────────────────────────────────────────────────────────────────────────

def example_adaptive_n_bits():
    """
    adaptive_n_bits=True: automatically estimate the required grid resolution
    per dimension from local curvature (basin width).

    Best for:
      - Functions with narrow basins (Schwefel: ~0.4 unit in 1000-unit domain)
      - When you don't know in advance how fine the grid needs to be
      - Eliminates manual tuning of n_bits_schedule

    How it works:
      1. Probes f''(x*) via finite difference at current best coordinate
      2. Estimates basin width = 2√(2·tol_f / |f''|)
      3. Sets n_bits so ≥ 4 grid points span the basin
      4. Clips result to [min_bits, max_bits]
    """
    print("=" * 60)
    print("EXAMPLE 4: adaptive_n_bits — auto-tune resolution")
    print("  Schwefel 2D: narrow ~0.4-unit basin in 1000-unit domain")
    print("  True min ≈ (420.969, 420.969)  f ≈ 0")
    print("=" * 60)

    def schwefel(X):
        D = X.shape[1]
        return 418.9829*D - np.sum(X*np.sin(np.sqrt(np.abs(X))), axis=1)

    true_xy = np.array([420.9687, 420.9687])

    # Without adaptive — default schedule may miss narrow basin
    r_default = hybrid_adaptive_minimize(
        schwefel, [(-500., 500.)] * 2,
        n_bits_schedule = (6, 8),
        adaptive_n_bits = False,
        max_cycles=3, seed=42, verbose=False,
    )
    d_default = np.linalg.norm(r_default.x - true_xy)
    print(f"\n  Without adaptive_n_bits (schedule=(6,8)):")
    print(f"    n_bits used = (6, 8) for all dims")
    print(f"    dist        = {d_default:.5f}")
    print(f"    f           = {r_default.fun:.8f}")

    # With adaptive — curvature estimator chooses n_bits=12 automatically
    r_adaptive = hybrid_adaptive_minimize(
        schwefel, [(-500., 500.)] * 2,
        n_bits_schedule = (6, 8),    # fallback schedule
        adaptive_n_bits = True,      # enable per-dim resolution
        min_bits        = 4,
        max_bits        = 14,
        max_cycles=3, seed=42, verbose=False,
    )
    d_adaptive = np.linalg.norm(r_adaptive.x - true_xy)
    print(f"\n  With adaptive_n_bits=True:")
    print(f"    n_bits used = {r_adaptive.dim_n_bits}  (auto-detected narrow basin)")
    print(f"    dist        = {d_adaptive:.5f}  ({d_default/d_adaptive:.1f}× more accurate)")
    print(f"    f           = {r_adaptive.fun:.8f}")
    print()

    # Show the improvement across all five benchmark functions
    print("  Comparison across all benchmark functions:")
    suite = BenchmarkSuite()
    print(f"  {'Function':<22} {'without adapt':>14} {'with adapt':>12}  improvement")
    print("  " + "-"*62)
    for fname in suite.names:
        entry = suite[fname]
        r1 = suite.run(fname, hybrid_adaptive_minimize,
                       n_bits_schedule=(6,8), max_cycles=3, seed=42,
                       adaptive_n_bits=False, verbose=False)
        r2 = suite.run(fname, hybrid_adaptive_minimize,
                       n_bits_schedule=(6,8), max_cycles=3, seed=42,
                       adaptive_n_bits=True, min_bits=4, max_bits=14,
                       verbose=False)
        imp = f"{r1.dist/r2.dist:.1f}×" if r2.dist > 1e-9 else "max"
        print(f"  {fname:<22} {r1.dist:>14.5f} {r2.dist:>12.5f}  {imp}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Example 5 — BenchmarkSuite: built-in test functions
# ─────────────────────────────────────────────────────────────────────────────

def example_benchmark_suite():
    """
    BenchmarkSuite: all five canonical test functions with known true minima.

    Use to:
      - Verify the library is working after changes
      - Compare methods systematically
      - Understand which functions are hard/easy for each method
    """
    print("=" * 60)
    print("EXAMPLE 5: BenchmarkSuite")
    print("=" * 60)

    suite = BenchmarkSuite()
    print(f"\n{suite}\n")

    # Show metadata for each function
    print("  Function details:")
    for name in suite.names:
        e = suite[name]
        print(f"  {name:<22}  [{e.difficulty}]")
        print(f"    bounds:  {e.bounds}")
        print(f"    true_x:  {np.round(e.true_x, 4)}")
        print(f"    true_f:  {e.true_f:.6f}")

    print()

    # Run all methods on all functions
    print("  run_all — grover_minimize (n_bits=8):")
    results_gm = suite.run_all(
        grover_minimize, n_bits=8, n_trials=3, seed=42, verbose=True)

    print()
    print("  run_all — hybrid_adaptive_minimize:")
    results_ha = suite.run_all(
        hybrid_adaptive_minimize,
        n_bits_schedule=(6,8), max_cycles=3, seed=42, verbose=True)

    # Compare
    print()
    print("  Improvement: hybrid vs grover_minimize (dist ratio):")
    for fname in suite.names:
        d_gm = results_gm[fname].dist
        d_ha = results_ha[fname].dist
        ratio = d_gm / d_ha if d_ha > 1e-10 else float('inf')
        print(f"    {fname:<22}  {d_gm:.5f} → {d_ha:.5f}"
              f"  ({ratio:.1f}× better)" if ratio < 1000 else
              f"    {fname:<22}  {d_gm:.5f} → {d_ha:.5f}  (max improvement)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Example 6 — custom function template
# ─────────────────────────────────────────────────────────────────────────────

def example_custom_function():
    """
    Template: plug in your own function.

    Your function must accept X of shape (N, D) and return shape (N,).
    This is the vectorised convention — all grid points evaluated at once.
    """
    print("=" * 60)
    print("EXAMPLE 6: Custom function template")
    print("=" * 60)

    def my_function(X):
        """
        Replace this with your own function.
        X shape: (N, D)  →  return shape: (N,)
        """
        x, y = X[:, 0], X[:, 1]
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2   # Himmelblau

    # ── Quick search ──────────────────────────────────────────────────────
    r = grover_minimize(my_function, [(-5, 5), (-5, 5)], n_bits=7, seed=0)
    print(f"  Quick search (n_bits=7):       x={np.round(r.x,4)}  f={r.fun:.4f}")

    # ── High accuracy via zoom ────────────────────────────────────────────
    r = grover_minimize_hierarchical(
        my_function, [(-5, 5), (-5, 5)],
        n_bits_schedule=(6, 8), max_layers=2,
        verbose=False, seed=0)
    print(f"  High accuracy (hierarchical):  x={np.round(r.x,5)}  f={r.fun:.6f}")

    # ── Adaptive (good default) ───────────────────────────────────────────
    r = hybrid_adaptive_minimize(
        my_function, [(-5, 5), (-5, 5)],
        n_bits_schedule=(6, 8), max_cycles=4,
        verbose=False, seed=0)
    print(f"  Adaptive:                      x={np.round(r.x,5)}  f={r.fun:.6f}")
    print(f"  Dim characters: {r.dim_characters}")

    # ── Adaptive with auto n_bits ─────────────────────────────────────────
    r = hybrid_adaptive_minimize(
        my_function, [(-5, 5), (-5, 5)],
        n_bits_schedule=(6, 8), max_cycles=4,
        adaptive_n_bits=True, min_bits=4, max_bits=12,
        verbose=False, seed=0)
    print(f"  Adaptive + auto n_bits:        x={np.round(r.x,5)}  f={r.fun:.6f}")
    print(f"  Auto n_bits per dim: {r.dim_n_bits}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Example 7 — decision guide
# ─────────────────────────────────────────────────────────────────────────────

def example_decision_guide():
    print("=" * 60)
    print("EXAMPLE 7: Which function to use?")
    print("=" * 60)
    print("""
  grover_minimize
    Use when:  D=1-2, quick exploration, understanding n_bits effect
    Accuracy:  grid-spacing limited (set n_bits=8-10 for high accuracy)
    Example:   grover_minimize(f, bounds, n_bits=8)

  grover_minimize_hierarchical
    Use when:  D=2-4, need high accuracy, function is 2D/3D
    Accuracy:  near machine precision after 3-4 layers
    Example:   grover_minimize_hierarchical(f, bounds,
                   n_bits_schedule=(6,8,10), max_layers=3)

  hybrid_adaptive_minimize
    Use when:  D=2 to D=50+, mixed function landscapes, default choice
    Accuracy:  high (coordinate-wise + joint zoom)
    Example:   hybrid_adaptive_minimize(f, bounds,
                   n_bits_schedule=(6,8), max_cycles=5)

  hybrid_adaptive_minimize with adaptive_n_bits=True
    Use when:  narrow basins, unknown resolution requirements
    Accuracy:  best overall — auto-tunes n_bits per dimension
    Example:   hybrid_adaptive_minimize(f, bounds,
                   n_bits_schedule=(6,8), adaptive_n_bits=True,
                   min_bits=4, max_bits=14)

  Key parameters:
    n_bits / n_bits_schedule  — grid resolution (higher = more accurate, slower)
    adaptive_n_bits           — auto-tune n_bits from basin curvature
    zoom_factor               — zoom window size (larger = safer for narrow basins)
    tol_f / tol_f_rel         — convergence tolerance (use tol_f_rel for large f)
    n_repeats                 — noise averaging (>1 for stochastic functions)
    max_layers / max_cycles   — hard ceiling on iterations
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Run all examples
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  numpy_grover — usage examples")
    print("  Author: Faluke Aikebaier")
    print("  https://github.com/Faluke-Aikebaier/numpy_grover")
    print("=" * 60 + "\n")

    example_grover_minimize()
    example_grover_minimize_hierarchical()
    example_hybrid_adaptive()
    example_adaptive_n_bits()
    example_benchmark_suite()
    example_custom_function()
    example_decision_guide()

    print("All examples complete.")
