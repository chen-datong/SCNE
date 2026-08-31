import numpy as np
from sympy.combinatorics import Permutation, SymmetricGroup
import math

# Block 1: Weingarten function implementation
def weingarten_n3(partition, d):
    """
    Calculates the Weingarten function for n=3 for a given partition and dimension d.
    Formulas are from the paper: https://arxiv.org/abs/math-ph/0205010
    This will fail for d < 3, as the formula is not defined there.
    """
    d = float(d)
    d2 = d**2
    
    # These are the formulas for n=3 partitions
    formulas = {
        (1, 1, 1): (d2 - 2) / (d * (d2 - 1) * (d2 - 4)),
        (2, 1): -1 / ((d2 - 1) * (d2 - 4)),
        (3,): 2 / (d * (d2 - 1) * (d2 - 4)),
    }
    
    # The partition should be sorted to match keys, e.g. (1, 2) -> (2, 1)
    sorted_partition = tuple(sorted(partition, reverse=True))
    
    return formulas.get(sorted_partition, float('nan'))

def get_GHZ_state(d):
    """
    Creates a maximally entangled Bell state rho = |psi><psi|
    where |psi> = 1/sqrt(d) * sum_{i=0}^{d-1} |i>|i>.
    """
    ghz_vec = np.zeros(d); ghz_vec[0] = 1/np.sqrt(2); ghz_vec[-1] = 1/np.sqrt(2)
    rho = np.outer(ghz_vec, ghz_vec.conj())
    return rho

def calculate_S3(rho, d_A, d_B):
    """
    Calculates sum_{s,t} Wg(s^{-1}t, d_A) * tr( (s_A (x) t_B^{-1}) * rho^{\otimes 3} )
    This optimized version avoids constructing large matrices.
    """
    # print("Starting calculation for Eq. (61) (Optimized)...")
    s3_elements = get_s3_elements()
    total_sum = 0.0

    # The GHZ state is a superposition of 2 states, so its triple tensor product is a superposition of 2^3=8 states.
    # We can work in this 8-dimensional subspace.
    # The basis vectors of the subspace can be indexed by (i,j,k) where i,j,k are 0 or 1.
    # '0' corresponds to the |0> state, '1' corresponds to the |d-1> state.
    basis = [
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
    ]
    basis_map = {b: i for i, b in enumerate(basis)}

    # print("Entering main loop over S3 x S3...")
    for sigma in s3_elements:
        for tau in s3_elements:
            # --- Calculate Weingarten value ---
            perm_prod = (sigma**-1) * tau
            partition = tuple(sorted([len(c) for c in perm_prod.cyclic_form] + [1] * (perm_prod.size - len(perm_prod.support())), reverse=True))
            
            try:
                wg_val = weingarten_n3(partition, d_A)
                if math.isinf(wg_val) or math.isnan(wg_val):
                    raise ZeroDivisionError
            except ZeroDivisionError:
                print(f"ERROR: Division by zero for Wg with partition {partition} and d_A={d_A}.")
                return float('nan')

            # --- Calculate Trace term without building matrices ---
            # The trace is (1/8) * sum_{i,j,k,i',j',k'} <A_i'A_j'A_k'|U_s|A_iA_jA_k> * <B_i'B_j'B_k'|U_t_inv|B_iB_jB_k>
            # This can be computed as (1/8) * Tr(M_A * M_B.T) where M_A and M_B are 8x8 matrices.
            
            s_inv = sigma**-1
            
            trace_val = 0
            for b1_idx, b1 in enumerate(basis):
                # Apply permutation s_inv to basis b1 to get b_perm_A
                b_perm_A = (b1[s_inv(0)], b1[s_inv(1)], b1[s_inv(2)])
                
                # Apply permutation tau to basis b1 to get b_perm_B
                # The operator is U_{tau^{-1}}, which permutes by (tau^{-1})^{-1} = tau.
                b_perm_B = (b1[tau(0)], b1[tau(1)], b1[tau(2)])

                for b2_idx, b2 in enumerate(basis):
                    # Check if the permuted A state overlaps with b2
                    overlap_A = 1 if b_perm_A == b2 else 0
                    # Check if the permuted B state overlaps with b2
                    overlap_B = 1 if b_perm_B == b2 else 0
                    
                    if overlap_A and overlap_B:
                        trace_val += 1
            
            trace_val /= 8.0  # From the (1/sqrt(2))^6 factor of the state
            
            total_sum += wg_val * trace_val

    # print("Calculation finished.")
    return total_sum * (d_A**3)

def get_s3_elements():
    """Returns a list of all permutations in the S3 group."""
    S3 = SymmetricGroup(3)
    return list(S3.elements)

if __name__ == '__main__':
    # --- Main execution for Eq. (61) calculation ---
    print("--- Calculating Equation (61) ---")
    nA = 9
    nB = 1

    # User-defined parameters
    d_A_val = 2 ** nA
    d_B_val = 2 ** nB
    
    print(f"Parameters: n_A = {nA}, n_B = {nB}, rho = GHZ state")
    
    # Get the state
    rho_state = get_GHZ_state(d_A_val* d_B_val) # Assuming d_A=d_B for Bell state
    
    # Run the calculation
    result = calculate_S3(rho_state, d_A_val, d_B_val)
    
    print("\n--- RESULT ---")
    print(f"The final result of the summation is: {result}")
    print("-" * 20)
