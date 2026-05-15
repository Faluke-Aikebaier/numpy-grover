"""
numpy_grover
============
Quantum-inspired global optimisation library.

Implements the Grover / Durr-Hoyer search algorithm in pure NumPy,
extended with a hierarchical zoom-and-refine strategy and a hybrid
adaptive minimiser for medium-to-high dimensional problems.

Public API
----------
Core functions::

    grover_minimize(func, bounds, n_bits, ...)
        Single-layer Grover search over a discrete grid.

    grover_minimize_hierarchical(func, bounds, n_bits_schedule, ...)
        Zoom-and-refine: repeatedly shrinks the search window for
        high accuracy without paying for a finer global grid.

    hybrid_adaptive_minimize(func, bounds, n_bits_schedule, ...)
        Coordinate-wise adaptive search for D=2 to D=50+.
        Combines Grover (multimodal dims), Brent (smooth dims),
        and DE / CMA-ES (global basin finding).

Result containers::

    GroverResult
    HierarchicalGroverResult
    AdaptiveGroverResult

Low-level primitives::

    durr_hoyer_min(costs, seed)
    grover_oracle(state, marked_mask)
    grover_diffusion(state)
    grover_search(state, marked_mask, n_iter)
    classify_dimension(func, x_current, dim, bounds)

Compatibility::

    grover_minimize_adaptive   — alias for hybrid_adaptive_minimize
    grover_min_2d              — drop-in for Qiskit notebook interface

Author
------
Faluke Aikebaier
https://github.com/Faluke-Aikebaier/numpy_grover
"""

from .core import (
    # ── Main functions ────────────────────────────────────────────────
    grover_minimize,
    grover_minimize_hierarchical,
    hybrid_adaptive_minimize,

    # ── Backward-compatible alias ─────────────────────────────────────
    grover_minimize_adaptive,

    # ── Result containers ─────────────────────────────────────────────
    GroverResult,
    HierarchicalGroverResult,
    LayerRecord,
    AdaptiveGroverResult,
    DimRecord,
    CycleRecord,

    # ── Low-level primitives ──────────────────────────────────────────
    durr_hoyer_min,
    grover_oracle,
    grover_diffusion,
    grover_search,
    classify_dimension,

    # ── Qiskit drop-in ────────────────────────────────────────────────
    grover_min_2d,

    # ── Built-in benchmark functions ─────────────────────────────────
    BENCHMARK_FUNCTIONS,
    BenchmarkSuite,
    validate_all,
    benchmark,
)

__version__ = "0.1.0"
__author__  = "Faluke Aikebaier"
__email__   = ""
__license__ = "MIT"
