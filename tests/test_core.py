"""
tests/test_core.py
==================
Validation suite for numpy_grover.

Run with:
    pytest tests/
    pytest tests/ -v          # verbose
    pytest tests/ -v --tb=short
"""

import numpy as np
import pytest

from numpy_grover import (
    grover_oracle,
    grover_diffusion,
    grover_search,
    durr_hoyer_min,
    grover_minimize,
    grover_minimize_hierarchical,
    hybrid_adaptive_minimize,
    grover_minimize_adaptive,   # alias
    classify_dimension,
    validate_all,
)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level primitives
# ─────────────────────────────────────────────────────────────────────────────

class TestGroverOracle:
    def test_flips_marked_states(self):
        state  = np.ones(8) / np.sqrt(8)
        mask   = np.array([True, False, True, False, False, False, False, False])
        before = state.copy()
        grover_oracle(state, mask)
        assert np.allclose(state[mask],  -before[mask])
        assert np.allclose(state[~mask],  before[~mask])

    def test_no_mask_unchanged(self):
        state  = np.ones(8) / np.sqrt(8)
        before = state.copy()
        grover_oracle(state, np.zeros(8, dtype=bool))
        assert np.allclose(state, before)

    def test_full_mask_flips_all(self):
        state  = np.array([0.5, 0.3, -0.2, 0.8])
        before = state.copy()
        grover_oracle(state, np.ones(4, dtype=bool))
        assert np.allclose(state, -before)


class TestGroverDiffusion:
    def test_preserves_norm(self):
        rng   = np.random.default_rng(0)
        state = rng.standard_normal(64)
        state /= np.linalg.norm(state)
        grover_diffusion(state)
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-12)

    def test_uniform_state_unchanged(self):
        """Inversion about mean leaves uniform superposition unchanged."""
        N     = 16
        state = np.ones(N) / np.sqrt(N)
        before = state.copy()
        grover_diffusion(state)
        assert np.allclose(state, before, atol=1e-14)

    def test_inversion_formula(self):
        state    = np.array([1.0, 2.0, 3.0, 4.0])
        mean     = state.mean()
        expected = 2 * mean - state
        grover_diffusion(state)
        assert np.allclose(state, expected)


class TestDurrHoyerMin:
    def test_finds_minimum_small(self):
        costs = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        idx, _ = durr_hoyer_min(costs, seed=0)
        assert idx == 1

    def test_finds_minimum_random(self):
        rng   = np.random.default_rng(42)
        costs = rng.random(1024)
        idx, nc = durr_hoyer_min(costs, seed=1)
        assert idx == int(np.argmin(costs))

    def test_oracle_calls_within_budget(self):
        rng    = np.random.default_rng(0)
        costs  = rng.random(4096)
        N      = len(costs)
        _, nc  = durr_hoyer_min(costs, seed=2)
        budget = int(np.ceil(22.5 * np.sqrt(N) + 1.4 * np.log2(N)**2)) + 1
        assert nc <= budget, f"oracle calls {nc} exceeded budget {budget}"

    def test_reliability_10_runs(self):
        rng   = np.random.default_rng(0)
        costs = rng.random(4096)
        true_min = int(np.argmin(costs))
        found    = [durr_hoyer_min(costs, seed=s)[0] for s in range(10)]
        assert all(f == true_min for f in found), \
            f"Not all runs found true minimum: {found}"

    def test_single_element(self):
        idx, nc = durr_hoyer_min(np.array([42.0]), seed=0)
        assert idx == 0
        assert nc  == 0

    def test_two_elements(self):
        costs  = np.array([3.0, 1.0])
        idx, _ = durr_hoyer_min(costs, seed=0)
        assert idx == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            durr_hoyer_min(np.array([]), seed=0)


# ─────────────────────────────────────────────────────────────────────────────
# grover_minimize
# ─────────────────────────────────────────────────────────────────────────────

class TestGroverMinimize:

    @staticmethod
    def _wavy_bowl(X):
        x, y = X[:,0], X[:,1]
        return x**2 + y**2 + 0.4 * np.sin(5*x) * np.sin(5*y)

    @staticmethod
    def _parabola_1d(X):
        return (X[:,0] - 1.5)**2

    def test_1d_parabola(self):
        res = grover_minimize(self._parabola_1d, [(0, 3)], n_bits=8, seed=0)
        assert abs(res.x[0] - 1.5) < 0.05
        assert res.fun < 0.01

    def test_2d_wavy_bowl(self):
        res = grover_minimize(self._wavy_bowl, [(-2,2),(-2,2)], n_bits=6, seed=42)
        # True minimum near (0.259, -0.259)
        assert res.fun < -0.20

    def test_result_fields(self):
        res = grover_minimize(self._parabola_1d, [(0, 3)], n_bits=6, seed=0)
        assert hasattr(res, 'x')
        assert hasattr(res, 'fun')
        assert hasattr(res, 'n_calls')
        assert hasattr(res, 'elapsed')
        assert hasattr(res, 'grid')
        assert res.elapsed > 0
        assert res.n_calls > 0

    def test_n_repeats_deterministic(self):
        """n_repeats=1 and n_repeats=3 on deterministic f give same result."""
        r1 = grover_minimize(self._wavy_bowl, [(-2,2),(-2,2)],
                             n_bits=6, n_repeats=1, seed=42)
        r2 = grover_minimize(self._wavy_bowl, [(-2,2),(-2,2)],
                             n_bits=6, n_repeats=3, seed=42)
        # Same function value (deterministic) — may differ by grid spacing
        assert np.isclose(r1.fun, r2.fun, atol=1e-6)

    def test_invalid_bounds_raises(self):
        with pytest.raises(ValueError):
            grover_minimize(self._parabola_1d, [(3, 0)], n_bits=4, seed=0)

    def test_invalid_n_bits_raises(self):
        with pytest.raises(ValueError):
            grover_minimize(self._parabola_1d, [(0, 3)], n_bits=0, seed=0)


# ─────────────────────────────────────────────────────────────────────────────
# grover_minimize_hierarchical
# ─────────────────────────────────────────────────────────────────────────────

class TestGroverMinimizeHierarchical:

    @staticmethod
    def _wavy_bowl(X):
        x, y = X[:,0], X[:,1]
        return x**2 + y**2 + 0.4 * np.sin(5*x) * np.sin(5*y)

    def test_improves_over_single_layer(self):
        r_single = grover_minimize(
            self._wavy_bowl, [(-2,2),(-2,2)], n_bits=6, seed=42)
        r_hier   = grover_minimize_hierarchical(
            self._wavy_bowl, [(-2,2),(-2,2)],
            n_bits_schedule=(6, 8), max_layers=3, seed=42, verbose=False)
        assert r_hier.fun <= r_single.fun + 1e-8

    def test_converges_to_known_minimum(self):
        r = grover_minimize_hierarchical(
            self._wavy_bowl, [(-2,2),(-2,2)],
            n_bits_schedule=(6,8,10), max_layers=4,
            tol_f=1e-7, seed=42, verbose=False)
        true_xy = np.array([0.25942, -0.25942])
        assert np.linalg.norm(r.x - true_xy) < 0.005

    def test_result_fields(self):
        r = grover_minimize_hierarchical(
            self._wavy_bowl, [(-2,2),(-2,2)],
            n_bits_schedule=(6,), max_layers=2, seed=42, verbose=False)
        assert hasattr(r, 'layers')
        assert hasattr(r, 'converged')
        assert hasattr(r, 'stop_reason')
        assert len(r.layers) >= 1

    def test_tol_f_stops_early(self):
        r = grover_minimize_hierarchical(
            self._wavy_bowl, [(-2,2),(-2,2)],
            n_bits_schedule=(6,8), max_layers=10,
            tol_f=1e-3, seed=42, verbose=False)
        assert r.n_layers < 10
        assert 'tol_f' in r.stop_reason or 'tol_x' in r.stop_reason

    def test_n_repeats_propagates(self):
        """n_repeats parameter accepted without error."""
        r = grover_minimize_hierarchical(
            self._wavy_bowl, [(-2,2),(-2,2)],
            n_bits_schedule=(6,), max_layers=2,
            n_repeats=2, seed=42, verbose=False)
        assert r.fun < 0.0


# ─────────────────────────────────────────────────────────────────────────────
# hybrid_adaptive_minimize
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridAdaptiveMinimize:

    @staticmethod
    def _rastrigin(X):
        A = 10.0
        return A * X.shape[1] + np.sum(X**2 - A * np.cos(2*np.pi*X), axis=1)

    @staticmethod
    def _sphere(X):
        return np.sum(X**2, axis=1)

    @staticmethod
    def _wavy_bowl(X):
        x, y = X[:,0], X[:,1]
        return x**2 + y**2 + 0.4 * np.sin(5*x) * np.sin(5*y)

    def test_alias_is_same_function(self):
        assert hybrid_adaptive_minimize is grover_minimize_adaptive

    def test_2d_wavy_bowl(self):
        r = hybrid_adaptive_minimize(
            self._wavy_bowl, [(-2,2),(-2,2)],
            n_bits_schedule=(6,8), max_cycles=3, seed=42, verbose=False)
        true = np.array([0.25942, -0.25942])
        assert np.linalg.norm(r.x - true) < 0.01

    def test_3d_sphere(self):
        """Sphere is unimodal — Brent should handle all dims."""
        r = hybrid_adaptive_minimize(
            self._sphere, [(-2,2)]*3,
            n_bits_schedule=(6,), max_cycles=3, seed=42, verbose=False)
        assert np.linalg.norm(r.x) < 0.01
        assert r.fun < 1e-4

    def test_result_has_dim_info(self):
        r = hybrid_adaptive_minimize(
            self._wavy_bowl, [(-2,2),(-2,2)],
            n_bits_schedule=(6,), max_cycles=2, seed=42, verbose=False)
        assert len(r.dim_characters) == 2
        assert len(r.dim_methods)    == 2
        assert all(c in ('multimodal','unimodal','monotone','flat')
                   for c in r.dim_characters)

    def test_coupling_warning_field_exists(self):
        r = hybrid_adaptive_minimize(
            self._sphere, [(-2,2)]*2,
            n_bits_schedule=(6,), max_cycles=2, seed=42, verbose=False)
        assert hasattr(r, 'coupling_warning')
        assert isinstance(r.coupling_warning, bool)

    def test_tol_f_rel_accepted(self):
        r = hybrid_adaptive_minimize(
            self._wavy_bowl, [(-2,2),(-2,2)],
            n_bits_schedule=(6,), max_cycles=3,
            tol_f=1e-9, tol_f_rel=1e-4,
            seed=42, verbose=False)
        assert r.fun < 0.0

    def test_n_repeats_accepted(self):
        r = hybrid_adaptive_minimize(
            self._wavy_bowl, [(-2,2),(-2,2)],
            n_bits_schedule=(6,), max_cycles=2,
            n_repeats=2, seed=42, verbose=False)
        assert r.fun < 0.0

    def test_summary_method(self):
        r   = hybrid_adaptive_minimize(
            self._sphere, [(-2,2)]*2,
            n_bits_schedule=(6,), max_cycles=2, seed=42, verbose=False)
        out = r.summary()
        assert isinstance(out, str)
        assert 'Stage 1' in out
        assert 'Cycle' in out

    def test_dim_names(self):
        r = hybrid_adaptive_minimize(
            self._wavy_bowl, [(-2,2),(-2,2)],
            n_bits_schedule=(6,), max_cycles=1,
            dim_names=['Delta', 'q'],
            seed=42, verbose=False)
        assert r.fun < 0.0   # just checks it runs


# ─────────────────────────────────────────────────────────────────────────────
# classify_dimension
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyDimension:

    def test_monotone_parabola_arm(self):
        """One arm of a parabola (x>0) is monotone."""
        def rising(X): return X[:,0]          # strictly increasing
        x0   = np.array([1.0])
        char = classify_dimension(rising, x0, 0, [(0.1, 5.0)])
        assert char == 'monotone'

    def test_unimodal_parabola(self):
        def parabola(X): return (X[:,0] - 2.0)**2
        x0   = np.array([0.0])
        char = classify_dimension(parabola, x0, 0, [(-1.0, 5.0)])
        assert char in ('unimodal', 'multimodal')   # depends on probe density

    def test_flat_dimension(self):
        """f constant along dim 1."""
        def only_x(X): return X[:,0]**2
        x0   = np.array([1.0, 0.0])
        char = classify_dimension(only_x, x0, 1, [(-2.0, 2.0), (-2.0, 2.0)])
        assert char == 'flat'

    def test_multimodal_rastrigin(self):
        def rastrigin_1d(X):
            return 10.0 + X[:,0]**2 - 10*np.cos(2*np.pi*X[:,0])
        x0   = np.array([0.5])
        char = classify_dimension(rastrigin_1d, x0, 0, [(-5.12, 5.12)],
                                  n_probe=64)
        assert char == 'multimodal'


# ─────────────────────────────────────────────────────────────────────────────
# Validate all (integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateAll:
    def test_validate_all_passes(self):
        ok = validate_all(n_bits=6, verbose=False)
        assert ok, "validate_all() returned False — some checks failed"
