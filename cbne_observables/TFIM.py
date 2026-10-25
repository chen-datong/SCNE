import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
import os
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parents[1] / "data" / "cbne_observables" / "TFIM"


def build_tfim_pauli_sparse(N: int, J: float = 1.0, h: float = 1.0, pbc: bool = False):
    r"""
    Transverse-field Ising model with Pauli matrices (NO 1/2 factors):

        H = -J * Σ_i (σ^z_i σ^z_{i+1}) - h * Σ_i (σ^x_i)

    Computational basis: bit=1 -> |↑z>, bit=0 -> |↓z>.
    Then σ^z eigenvalue is +1 for bit=1 else -1.
    σ^x flips a single bit with matrix element 1, so off-diagonal amplitude is -h.
    """
    d = 1 << N

    bonds = [(i, i + 1) for i in range(N - 1)]
    if pbc and N > 2:
        bonds.append((N - 1, 0))

    rows, cols, data = [], [], []

    def zval(bit):
        return 1 if bit else -1

    for s in range(d):
        # diagonal ZZ part
        diag = 0.0
        for i, j in bonds:
            bi = (s >> i) & 1
            bj = (s >> j) & 1
            diag += -J * (zval(bi) * zval(bj))

        rows.append(s)
        cols.append(s)
        data.append(diag)

        # transverse field X part: -h * σ^x_i flips bit i
        # <s|σ^x_i|s^flip> = 1
        for i in range(N):
            s2 = s ^ (1 << i)
            rows.append(s)
            cols.append(s2)
            data.append(-h)

    H = sp.csr_matrix(
        (np.array(data, dtype=np.float64), (np.array(rows), np.array(cols))),
        shape=(d, d),
    )
    return H

def gibbs_full_rho_tfim_to_npy(
    out_npy: str,
    N: int,
    beta: float,
    J: float = 1.0,
    h: float = 1.0,
    pbc: bool = False,
    dtype=np.complex64,
):
    """
    Construct full Gibbs state rho = exp(-beta H) / Z for TFIM (Pauli definition),
    save as .npy.

    Uses memmap to write rho without huge RAM spike.
    """
    d = 1 << N

    # build sparse H
    Hs = build_tfim_pauli_sparse(N=N, J=J, h=h, pbc=pbc)

    # diagonalize dense H (TFIM does NOT block-diagonalize in Sz sectors)
    Hd = Hs.toarray()  # this is the big memory step
    evals, evecs = eigh(Hd)  # Hd is real symmetric

    # stable weights and partition function
    Emin = float(evals[0])
    w_shift = np.exp(-beta * (evals - Emin))
    Z = float(w_shift.sum()) * float(np.exp(-beta * Emin))

    # prepare disk-backed output
    raw = out_npy + ".raw"
    if os.path.exists(raw):
        os.remove(raw)

    rho_mm = np.memmap(raw, mode="w+", dtype=dtype, shape=(d, d))
    rho_mm[:] = 0

    # rho = V diag(exp(-beta E)) V^T / Z  (V real orthonormal here)
    w = np.exp(-beta * (evals - Emin)) * np.exp(-beta * Emin)  # exp(-beta E)
    Vw = evecs * w[np.newaxis, :]                               # scale columns
    rho = (Vw @ evecs.T) / Z                                    # dense block

    rho_mm[:] = rho.astype(dtype, copy=False)
    rho_mm.flush()

    # Save as true .npy (has header) — later read with np.load(...)
    rho_read = np.memmap(raw, mode="r", dtype=dtype, shape=(d, d))
    np.save(out_npy, rho_read)

    del rho_mm, rho_read
    os.remove(raw)


def ground_state_full(out_npy: str,
    N: int,
    beta: float,
    J: float = 1.0,
    h: float = 1.0,
    pbc: bool = False,
    dtype=np.complex64
):
    d = 1 << N
    Hs = build_tfim_pauli_sparse(N=N, J=J, h=h, pbc=pbc)
    Hd = Hs.toarray()
    evals, evecs = eigh(Hd)
    Emin = float(evals[0])
    psi0 = evecs[:, 0]
    np.save(out_npy, psi0.astype(dtype))
    print(f"Ground state energy: {Emin}")
    return Emin, psi0
