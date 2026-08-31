import numpy as np
import scipy.sparse as sp
import pandas as pd
from scipy.linalg import eigh
import os
from XXZ import ppt_moments_from_rho, depolarize_state, partial_transpose_rho
from utils import D3, D4, ppt3, D5, B2

def states_with_nup(N, nup):
    return np.array(
        [s for s in range(1 << N) if s.bit_count() == nup],
        dtype=np.uint32
    )

def build_heisenberg_pauli_sector(N, nup, J=1.0, pbc=False):
    """
    H = -J Σ (σxσx + σyσy + σzσz)
    Pauli matrices, NO 1/2 factors.
    """
    basis = states_with_nup(N, nup)
    dim = len(basis)
    index = {int(s): i for i, s in enumerate(basis)}

    rows, cols, data = [], [], []

    bonds = [(i, i+1) for i in range(N-1)]
    if pbc and N > 2:
        bonds.append((N-1, 0))

    def sz(bit):
        return +1 if bit else -1

    for a, s in enumerate(basis):
        diag = 0.0

        for i, j in bonds:
            bi = (s >> i) & 1
            bj = (s >> j) & 1

            # ZZ term: -J * (±1)
            diag += -J * (sz(bi) * sz(bj))

            # XX + YY flip term: amplitude = -2J
            if bi != bj:
                s2 = s ^ ((1 << i) | (1 << j))
                b = index[int(s2)]
                rows.append(a)
                cols.append(b)
                data.append(-2.0 * J)

        rows.append(a)
        cols.append(a)
        data.append(diag)

    H = sp.csr_matrix(
        (np.array(data), (np.array(rows), np.array(cols))),
        shape=(dim, dim),
        dtype=np.float64
    )
    return H, basis

def gibbs_full_rho_pauli(
    out_npy, N, beta, J=1.0, pbc=False, dtype=np.complex64
):
    d = 1 << N
    raw = out_npy + ".raw"
    if os.path.exists(raw):
        os.remove(raw)

    rho_mm = np.memmap(raw, mode="w+", dtype=dtype, shape=(d, d))
    rho_mm[:] = 0.0

    blocks = []
    logs = []

    # --- diagonalize each Sz sector ---
    for nup in range(N + 1):
        Hs, basis = build_heisenberg_pauli_sector(N, nup, J, pbc)
        Hd = Hs.toarray()
        evals, evecs = eigh(Hd)
        Emin = evals[0]
        Zs = np.exp(-beta * (evals - Emin)).sum()
        logs.append(-beta * Emin)
        blocks.append((basis, evals, evecs, Emin, Zs))

    # --- global partition function ---
    m = max(logs)
    Z = sum(np.exp(l - m) * Zs for (_, _, _, _, Zs), l in zip(blocks, logs))
    Z *= np.exp(m)

    # --- embed blocks into full rho ---
    for basis, evals, evecs, Emin, Zs in blocks:
        w = np.exp(-beta * evals)
        rho_block = (evecs * w) @ evecs.T.conj() / Z
        idx = basis.astype(np.int64)
        rho_mm[np.ix_(idx, idx)] = rho_block.astype(dtype)

    rho_mm.flush()
    rho = np.memmap(raw, mode="r", dtype=dtype, shape=(d, d))
    np.save(out_npy, rho)

    del rho_mm, rho
    os.remove(raw)
    return Z

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

    return 

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
    return Emin, psi0, Hd


def gibbs_ent_detect(N, nA, nB, beta, J=1.0, h=2.5, pbc=1, sys="B"):
    d = 2 ** N
    dimA = 2 ** nA
    dimB = 2 ** nB
    # out_npy = f"./Gibbs/gibbs_heisenberg_N{N}_beta{beta:.2f}.npy"
    out_npy = f"./Gibbs/TFIM_N{N}_beta{beta}_J{J}_h{h}_pbc{int(pbc)}.npy"
    if not os.path.exists(out_npy):
        # gibbs_full_rho_pauli(out_npy, N, beta, J=1.0, pbc=True, dtype=np.complex64)
        gibbs_full_rho_tfim_to_npy(out_npy, N, beta, J=J, h=h, pbc=bool(pbc), dtype=np.complex64)
    rho_mm = np.load(out_npy)

    moments = ppt_moments_from_rho(rho_mm, dimA, dimB, t_max=5, sys=sys)
    ppt3_val = ppt3(moments[2], moments[3])
    D3_val = D3(moments[2], moments[3])
    D4_val = D4(moments[2], moments[3], moments[4])
    D5_val = D5(moments[2], moments[3], moments[4], moments[5])
    B2_val = B2(moments[2], moments[3], moments[4], moments[5])
    print(f"N={N}, beta={beta:.2f}, ppt3={ppt3_val}, B2={B2_val}, D3={D3_val}, D4={D4_val}, D5={D5_val}, moments={moments}")
    return moments

def load_results_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["nA", "nB", "Nu", "NM"]:
        df[c] = df[c].astype(int)
    return df


def extract_arrays_gibbs(df: pd.DataFrame, nA: int, nB: int, Nu: int, NM: int, beta: float):

    sub = df[(df["nA"] == nA) & (df["nB"] == nB) & (df["NM"] == NM) & (df["Nu"] == Nu) & (df["beta"] == beta)]
    if sub.empty:
        raise ValueError(f"No data for nA={nA}, nB={nB}, NM={NM}, beta={beta}")

    # # 排序只是为了让顺序稳定（不影响平均）
    # sub = sub.sort_values(["nB", "Nu"])

    purity  = sub["purity"].to_numpy(float)
    third   = sub["thirdM"].to_numpy(float)
    fourth  = sub["fourthM"].to_numpy(float)
    fifth   = sub["fifthM"].to_numpy(float)

    return purity, third, fourth, fifth


def negativity(rho, dA, dB, sys="B"):
    rho_pt = partial_transpose_rho(rho, dA, dB, sys=sys)
    evals = np.linalg.eigvalsh(rho_pt)
    neg = np.sum(np.abs(evals[evals < 0]))

    return neg


if __name__ == "__main__":
    N = 11
    beta = 1
    # out_npy = f"./Gibbs/gibbs_heisenberg_N{N}_beta{beta:.2f}.npy"
    # gibbs_full_rho_pauli(out_npy, N, beta, J=1.0, pbc=True, dtype=np.complex64)
    # gibbs_ent_detect(N, nA=10, nB=1, beta=beta, sys="B")
    J = 1.0
    h = 2.5          # transverse field strength
    pbc = True

    out = f"./Gibbs/TFIM_N{N}_beta{beta}_J{J}_h{h}_pbc{int(pbc)}.npy"
    # Z = gibbs_full_rho_tfim_to_npy(out, N, beta, J=J, h=h, pbc=pbc, dtype=np.complex64)
    gibbs_ent_detect(N, nA=10, nB=1, beta=beta, h=h, sys="B")