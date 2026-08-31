"""Open-boundary antiferromagnetic XXZ model in the Pauli convention."""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh


CONVENTION = (
    "H = J sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Delta Z_i Z_{i+1}), "
    "open boundary, Pauli matrices, J > 0 antiferromagnetic"
)


def states_with_nup(n, nup):
    return np.asarray(
        [state for state in range(1 << n) if state.bit_count() == nup],
        dtype=np.uint32,
    )


def build_xxz_pauli_sector(n, nup, j_coupling=1.0, delta=1.0):
    """Build one fixed-magnetization block of the open XXZ Hamiltonian."""
    basis = states_with_nup(n, nup)
    index = {int(state): position for position, state in enumerate(basis)}
    rows = []
    columns = []
    values = []

    def z_value(bit):
        return 1.0 if bit else -1.0

    for row, state_value in enumerate(basis):
        state = int(state_value)
        diagonal = 0.0
        for site in range(n - 1):
            neighbor = site + 1
            bit_i = (state >> site) & 1
            bit_j = (state >> neighbor) & 1
            diagonal += j_coupling * delta * z_value(bit_i) * z_value(bit_j)
            if bit_i != bit_j:
                flipped = state ^ ((1 << site) | (1 << neighbor))
                rows.append(row)
                columns.append(index[flipped])
                values.append(2.0 * j_coupling)
        rows.append(row)
        columns.append(row)
        values.append(diagonal)

    matrix = sp.csr_matrix(
        (np.asarray(values, dtype=float), (rows, columns)),
        shape=(basis.size, basis.size),
    )
    return matrix, basis


def full_spectrum(n, j_coupling=1.0, delta=1.0):
    """Diagonalize all magnetization sectors and return the sorted full spectrum."""
    sector_energies = []
    for nup in range(n + 1):
        block, _ = build_xxz_pauli_sector(n, nup, j_coupling, delta)
        energies = eigh(
            block.toarray(),
            eigvals_only=True,
            overwrite_a=True,
            check_finite=False,
        )
        sector_energies.append(np.asarray(energies, dtype=float))
    return np.sort(np.concatenate(sector_energies))


def spectrum_filename(n, j_coupling, delta):
    return f"xxz_spectrum_N{n}_J{j_coupling:.2f}_Delta{delta:.2f}.npz"


def save_spectrum(path, energies, n, j_coupling, delta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ground_energy = float(energies[0])
    ground_degeneracy = int(
        np.count_nonzero(np.isclose(energies, ground_energy, rtol=0.0, atol=1e-9))
    )
    np.savez_compressed(
        path,
        energies=np.asarray(energies, dtype=float),
        n=np.int64(n),
        J=np.float64(j_coupling),
        Delta=np.float64(delta),
        ground_degeneracy=np.int64(ground_degeneracy),
        convention=np.asarray(CONVENTION),
    )


def load_spectrum(path, expected_n=None, expected_j=None, expected_delta=None):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        energies = np.asarray(archive["energies"], dtype=float)
        n = int(archive["n"])
        j_coupling = float(archive["J"])
        delta = float(archive["Delta"])
        degeneracy = int(archive["ground_degeneracy"])
    if energies.size != 1 << n:
        raise ValueError(f"Spectrum size mismatch in {path}: {energies.size} != 2^{n}")
    if np.any(np.diff(energies) < 0):
        raise ValueError(f"Spectrum is not sorted: {path}")
    if expected_n is not None and n != int(expected_n):
        raise ValueError(f"Unexpected n in {path}: {n}")
    if expected_j is not None and not np.isclose(j_coupling, expected_j):
        raise ValueError(f"Unexpected J in {path}: {j_coupling}")
    if expected_delta is not None and not np.isclose(delta, expected_delta):
        raise ValueError(f"Unexpected Delta in {path}: {delta}")
    if degeneracy != 1:
        raise ValueError(
            f"QVC ground-state fidelity requires a unique ground state; "
            f"found degeneracy {degeneracy} in {path}."
        )
    return energies


def gibbs_eigenvalues(energies, beta):
    shifted = np.asarray(energies, dtype=float) - float(energies[0])
    weights = np.exp(-float(beta) * shifted)
    return weights / weights.sum()


def exact_ground_fidelity(energies, beta, order=1):
    """Return <gs|rho^order|gs>/tr(rho^order) for a Gibbs state rho."""
    shifted = np.asarray(energies, dtype=float) - float(energies[0])
    return float(1.0 / np.exp(-float(order) * float(beta) * shifted).sum())
