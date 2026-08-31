import numpy as np
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import eigsh
import os

# Pauli matrices
sx = csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.float64))
sy = csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex128))
sz = csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.float64))
I2 = identity(2, format="csr", dtype=np.complex128)

# Spin-1/2 operators S = sigma/2
Sx = (sx.astype(np.complex128)) * 0.5
Sy = sy * 0.5
Sz = (sz.astype(np.complex128)) * 0.5

def op_on_site(op, site, N):
    out = None
    for i in range(N):
        piece = op if i == site else I2
        out = piece if out is None else kron(out, piece, format="csr")
    return out

def two_site(op1, i, op2, j, N):
    out = None
    for k in range(N):
        if k == i:
            piece = op1
        elif k == j:
            piece = op2
        else:
            piece = I2
        out = piece if out is None else kron(out, piece, format="csr")
    return out

def xxz_full_hamiltonian(N, J=1.0, Delta=1.0, pbc=False):
    dim = 2**N
    H = csr_matrix((dim, dim), dtype=np.complex128)

    bonds = [(i, i+1) for i in range(N-1)]
    if pbc and N > 2:
        bonds.append((N-1, 0))

    for (i, j) in bonds:
        H += J * (two_site(Sx, i, Sx, j, N) +
                  two_site(Sy, i, Sy, j, N) +
                  Delta * two_site(Sz, i, Sz, j, N))
    return H

def ground_state_full(N, J=1.0, Delta=1.0, pbc=False):
    H = xxz_full_hamiltonian(N, J=J, Delta=Delta, pbc=pbc)
    # smallest algebraic eigenvalue
    vals, vecs = eigsh(H, k=1, which="SA")
    E0 = float(np.real(vals[0]))
    psi0 = vecs[:, 0]  # normalized
    return E0, psi0, H

def partial_transpose_rho(rho: np.ndarray, dA: int, dB: int, sys: str = "B") -> np.ndarray:
    """
    Partial transpose of density matrix rho on subsystem A or B.
    rho is (dA*dB, dA*dB).
    """
    if rho.shape != (dA * dB, dA * dB):
        raise ValueError("rho shape mismatch with dA*dB.")

    # reshape indices: (iA, iB, jA, jB)
    R = rho.reshape(dA, dB, dA, dB)

    if sys.upper() == "B":
        # transpose iB <-> jB  => (iA, jB, jA, iB)
        R_pt = R.transpose(0, 3, 2, 1)
    elif sys.upper() == "A":
        # transpose iA <-> jA  => (jA, iB, iA, jB)
        R_pt = R.transpose(2, 1, 0, 3)
    else:
        raise ValueError("sys must be 'A' or 'B'.")

    return R_pt.reshape(dA * dB, dA * dB)

def depolarize_state(rho: np.ndarray, d: int, dprob: float):
    return (1 - dprob) * rho + dprob / d * np.eye(d)

def ppt_moments_from_rho(rho: np.ndarray, dA: int, dB: int, t_max: int = 5, sys: str = "B"):
    """
    Return PPT moments M_t = Tr[(rho^{T_sys})^t] for t=2..t_max.
    """
    rho_pt = partial_transpose_rho(rho, dA, dB, sys=sys)

    # iterative powers to avoid repeated np.linalg.matrix_power overhead
    moments = {}
    X = rho_pt.copy()
    for t in range(1, t_max + 1):
        if t >= 2:
            moments[t] = np.trace(X)
        X = X @ rho_pt

    # keep only 2..t_max
    return {t: float(np.real_if_close(moments[t])) for t in range(2, t_max + 1)}


def ppt_moments_from_psi(psi: np.ndarray, dA: int, dB: int, t_max: int = 5, sys: str = "B"):
    """
    psi is statevector length dA*dB. Pure state rho = |psi><psi|.
    """
    psi = np.asarray(psi)
    if psi.ndim != 1 or psi.shape[0] != dA * dB:
        raise ValueError("psi must be a 1D vector of length dA*dB.")
    # normalize (optional but safe)
    norm = np.vdot(psi, psi).real
    if not np.isclose(norm, 1.0):
        psi = psi / np.sqrt(norm)

    rho = np.outer(psi, psi.conj())
    return ppt_moments_from_rho(rho, dA, dB, t_max=t_max, sys=sys)


def ppt_moment(na, nb, t_max, sys="B"):
    N = na + nb
    J = 1.0
    Delta = 1.0  # Heisenberg
    if not os.path.exists(f"./heisenberg/heisenberg_N{N}_Jz{Delta}.npy"):
        E0, psi0, H = ground_state_full(N, J=J, Delta=Delta, pbc=False)
        print("E0 =", E0)
        np.save(f"./heisenberg/heisenberg_N{N}_Jz{Delta}.npy", psi0)
    psi = np.load(f"./heisenberg/heisenberg_N{N}_Jz{Delta}.npy")
    dA, dB = 2**na, 2**nb
    moments = ppt_moments_from_psi(psi, dA, dB, t_max=t_max, sys=sys)
    return moments

def ppt_moment_depolarize(na, nb, p, t_max, sys="B"):
    N = na + nb
    J = 1.0
    Delta = 1.0  # Heisenberg
    if not os.path.exists(f"./heisenberg/heisenberg_N{N}_Jz{Delta}.npy"):
        E0, psi0, H = ground_state_full(N, J=J, Delta=Delta, pbc=False)
        print("E0 =", E0)
        np.save(f"./heisenberg/heisenberg_N{N}_Jz{Delta}.npy", psi0)
    psi = np.load(f"./heisenberg/heisenberg_N{N}_Jz{Delta}.npy")
    rho = np.outer(psi, psi.conj())
    dA, dB = 2**na, 2**nb
    rhop = depolarize_state(rho, 2 ** (na + nb), p)
    moments = ppt_moments_from_rho(rhop, dA, dB, t_max=t_max, sys=sys)
    return moments


if __name__ == "__main__":
    N = 10
    J = 1.0
    Delta = 1.0  # Heisenberg
    # E0, psi0, H = ground_state_full(N, J=J, Delta=Delta, pbc=False)
    # print("E0 =", E0)
    # np.save(f"./heisenberg/heisenberg_N{N}_Jz{Delta}.npy", psi0)
    # psi0 = np.load(f"./heisenberg/heisenberg_N{N}_Jz{Delta}.npy")
    # dA, dB = 2**9, 2**1
    # mom = ppt_moments_from_psi(psi0, dA, dB, t_max=5, sys="B")
    # print(mom)
