"""Small, non-production tests for the XXZ QVC implementation."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import unitary_group


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "experiments" / "virtual_cooling"
sys.path.insert(0, str(MODULE_DIR))

from generate_heisenberg_data import (  # noqa: E402
    diagonal_probabilities,
    initialize_worker,
    simulate_trial,
)
from heisenberg_model import (  # noqa: E402
    exact_ground_fidelity,
    full_spectrum,
    gibbs_eigenvalues,
)


X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)


def two_site(operator_a, site_a, operator_b, site_b, n):
    result = np.eye(1, dtype=complex)
    for site in range(n):
        if site == site_a:
            factor = operator_a
        elif site == site_b:
            factor = operator_b
        else:
            factor = I
        result = np.kron(result, factor)
    return result


def full_hamiltonian(n, j_coupling, delta):
    result = np.zeros((1 << n, 1 << n), dtype=complex)
    for site in range(n - 1):
        result += j_coupling * (
            two_site(X, site, X, site + 1, n)
            + two_site(Y, site, Y, site + 1, n)
            + delta * two_site(Z, site, Z, site + 1, n)
        )
    return result


class HeisenbergModelTests(unittest.TestCase):
    def test_sector_spectrum_matches_full_hamiltonian(self):
        expected = np.linalg.eigvalsh(full_hamiltonian(4, 1.0, 0.7))
        actual = full_spectrum(4, 1.0, 0.7)
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0.0)

    def test_exact_cooling_fidelity_matches_gibbs_eigenvalues(self):
        energies = full_spectrum(4, 1.0, 1.0)
        beta = 0.8
        for order in (1, 2, 3):
            weights = gibbs_eigenvalues(energies, beta)
            expected = weights[0] ** order / np.sum(weights**order)
            self.assertAlmostEqual(
                exact_ground_fidelity(energies, beta, order), expected, places=13
            )

    def test_blocked_diagonal_probabilities_match_dense_expression(self):
        rng = np.random.default_rng(17)
        unitary = unitary_group.rvs(8, random_state=rng)
        weights = rng.random(8)
        weights /= weights.sum()
        expected = np.real(np.diag(unitary @ np.diag(weights) @ unitary.conj().T))
        actual = diagonal_probabilities(unitary, weights, column_block=3)
        np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0.0)

    def test_one_trial_returns_finite_fidelities(self):
        energies = full_spectrum(3, 1.0, 1.0)
        initialize_worker(energies, 0.8, 1, 20_000, 123, 80_100, 4)
        result = simulate_trial(0)
        self.assertEqual(result["idx"], 0)
        for key in (
            "purity",
            "third_moment",
            "pf_rhoO2_fidelity",
            "pf_rhoO3_fidelity",
        ):
            self.assertTrue(np.isfinite(result[key]), key)


if __name__ == "__main__":
    unittest.main()
