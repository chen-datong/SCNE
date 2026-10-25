#include "itensor/all.h"
#include <random>
#include <complex>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>   // Save counts.
#include <cstddef>   // std::size_t

using namespace itensor;

// Haar-random U(4) gates

void
haarRandomU4(std::mt19937 &rng,
             std::complex<double> U[4][4])
{
    using C = std::complex<double>;
    std::normal_distribution<double> ndist(0.0,1.0);

    // 4x4 complex Gaussian matrix.
    C Z[4][4];
    for(int i = 0; i < 4; ++i)
    for(int j = 0; j < 4; ++j)
    {
        double x = ndist(rng);
        double y = ndist(rng);
        Z[i][j] = C(x,y);
    }

    // Column Gram-Schmidt gives a Haar-random U(4).
    C Q[4][4];
    for(int j = 0; j < 4; ++j)
    {
        C v[4];
        for(int i = 0; i < 4; ++i) v[i] = Z[i][j];

        for(int k = 0; k < j; ++k)
        {
            C proj = 0.0;
            for(int i = 0; i < 4; ++i)
            {
                proj += std::conj(Q[i][k]) * v[i];
            }
            for(int i = 0; i < 4; ++i)
            {
                v[i] -= proj * Q[i][k];
            }
        }

        double norm2 = 0.0;
        for(int i = 0; i < 4; ++i) norm2 += std::norm(v[i]);
        double norm = std::sqrt(norm2);

        for(int i = 0; i < 4; ++i)
        {
            Q[i][j] = v[i] / norm;
        }
    }

    for(int i = 0; i < 4; ++i)
    for(int j = 0; j < 4; ++j)
    {
        U[i][j] = Q[i][j];
    }
}

// Two-qubit gates on an MPS

// Apply gate G to sites (i, i+1).
// Gate indices: (s1, s2, s1', s2').
void
applyTwoSiteGateMPS(MPS &psi,
                    ITensor const& G,
                    int i,
                    int maxDim)
{
    psi.position(i);

    auto wf = psi(i) * psi(i+1);

    // Set the gate index directions explicitly.
    // Map (s_i, s_{i+1}) to (s_i', s_{i+1}').
    wf = G * wf;

    wf.noPrime();

    auto s1 = siteIndex(psi,i);
    auto li = leftLinkIndex(psi,i); // May be absent at i=1.

    IndexSet left = (li ? IndexSet(li,s1) : IndexSet(s1));

    auto [U,S,V] = svd(wf, left,
                       Args("Cutoff=",1E-12,"MaxDim=",maxDim));

    psi.setA(i,U);
    psi.setA(i+1,S*V);

}

// Build and apply a Haar-random gate to sites (i, i+1).
void
applyRandomTwoQubitGateMPS(MPS &psi,
                           int i,
                           std::mt19937 &rng,
                           int maxDim)
{
    using C = std::complex<double>;
    C U4[4][4];
    haarRandomU4(rng,U4);

    auto s1  = siteIndex(psi,i);
    auto s2  = siteIndex(psi,i+1);
    auto s1p = prime(s1);
    auto s2p = prime(s2);

    ITensor G(s1,s2,s1p,s2p);

    for(int in1=0; in1<2; ++in1)
    for(int in2=0; in2<2; ++in2)
    for(int out1=0; out1<2; ++out1)
    for(int out2=0; out2<2; ++out2)
    {
        int col = in1*2 + in2;
        int row = out1*2 + out2;

        auto inIdx1  = (in1  == 0 ? s1(1)  : s1(2));
        auto inIdx2  = (in2  == 0 ? s2(1)  : s2(2));
        auto outIdx1 = (out1 == 0 ? s1p(1) : s1p(2));
        auto outIdx2 = (out2 == 0 ? s2p(1) : s2p(2));

        G.set(inIdx1,inIdx2,outIdx1,outIdx2,U4[row][col]);
    }

    applyTwoSiteGateMPS(psi,G,i,maxDim);
}

// TFIM ground state (DMRG)
// H = -J * sum sigma^z_i sigma^z_{i+1} - h * sum sigma^x_i
// SpinHalf uses sigma/2; rescale terms for Pauli operators.
// sigma^z = 2*Sz, sigma^x = 2*Sx
// => sigma^z sigma^z = 4*Sz*Sz, sigma^x = 2*Sx
MPS
makeTFIMGroundState(const SiteSet &sites, double J, double h)
{
    int N = length(sites);

    AutoMPO ampo(sites);

    for(int i = 1; i <= N-1; ++i)
    {
        ampo += -4.0*J,"Sz",i,"Sz",i+1;   // -J * sigma^z sigma^z
    }
    for(int i = 1; i <= N; ++i)
    {
        ampo += -2.0*h,"Sx",i;           // -h * sigma^x
    }

    auto H = toMPO(ampo);

    // Start from a product-state MPS.
    InitState init(sites);
    for(int i = 1; i <= N; ++i) init.set(i,"Dn");
    MPS psi0(init);

    Sweeps sweeps(8);
    sweeps.maxdim() = 20, 40, 80, 120, 200, 200, 200, 200;
    sweeps.cutoff() = 1E-12;

    auto energy = dmrg(psi0, H, sweeps, Args("Quiet=",true));
    std::ostringstream oss;
    oss << "MPS/ising_N" << N
        << "_J" << std::fixed << std::setprecision(2) << J
        << "_h" << std::fixed << std::setprecision(2) << h
        << ".mps";

    std::string fname = oss.str();
    writeToFile(fname, psi0);
    MPS saved;
    readFromFile(fname, saved);
    auto savedNorm = norm(saved);
    auto savedEnergy = inner(saved, H, saved) / (savedNorm * savedNorm);
    if(std::abs(savedNorm - 1.0) > 1E-8 || std::abs(savedEnergy - energy) > 1E-8)
    {
        throw std::runtime_error("Saved ground state failed norm/energy validation");
    }
    if(std::abs(J - 1.0) < 1E-12 && std::abs(h - 1.0) < 1E-12)
    {
        const double exactEnergy = 1.0 - 1.0/std::sin(std::acos(-1.0)/(4.0*N+2.0));
        if(std::abs(savedEnergy - exactEnergy) > 1E-7)
            throw std::runtime_error("Ground-state energy disagrees with exact open TFIM");
    }
    std::cout << std::setprecision(16) << "Validated saved ground state: N=" << N
              << " energy=" << savedEnergy << " norm=" << savedNorm
              << " maxLinkDim=" << maxLinkDim(saved) << "\n";

    // DMRG updates psi0 in place.
    return psi0;
}


// Brickwork circuit (MPS)

void
applyBrickworkCircuit(MPS &psi,
                      int N,
                      int depth,
                      std::mt19937 &rng,
                      int maxDim)
{

    for(int layer = 0; layer < depth; ++layer)
    {
        if(layer % 2 == 0)
        {
            // Even layers: pairs (1,2), (3,4), ...
            for(int i = 1; i <= N-1; i += 2)
            {
                applyRandomTwoQubitGateMPS(psi,i,rng,maxDim);
            }
        }
        else
        {
            // Odd layers: pairs (2,3), (4,5), ...
            for(int i = 2; i <= N-1; i += 2)
            {
                applyRandomTwoQubitGateMPS(psi,i,rng,maxDim);
            }
        }
    }
}

// Full outcome probabilities

std::vector<double>
computeProbDistribution(const MPS &psi)
{
    const int N = length(psi);
    const std::size_t dim = std::size_t(1) << N;

    // Copy the MPS before changing its gauge.
    auto psiC = psi;
    psiC.position(1); // Set the orthogonality center to site 1.

    std::vector<double> probs(dim, 0.0);

    std::vector<ITensor> env_cur, env_next;
    env_cur.reserve(1);
    env_cur.emplace_back(1.0); // The empty prefix is scalar 1.

    for(int i = 1; i <= N; ++i)
    {
        env_next.clear();
        env_next.reserve(env_cur.size() * 2);

        // Use the site index from psiC.
        auto s  = siteIndex(psiC, i);
        auto Ai = psiC.A(i);  // site tensor

        // Project onto basis states 0 and 1.
        ITensor bra0(s), bra1(s);
        bra0.set(s(1), 1.0);
        bra1.set(s(2), 1.0);

        for(const auto &L : env_cur)
        {
            // prefix + 0
            env_next.emplace_back(L * Ai * bra0);

            // prefix + 1
            env_next.emplace_back(L * Ai * bra1);
        }

        env_cur.swap(env_next);
    }

    for(std::size_t x = 0; x < dim; ++x)
    {
        std::complex<double> amp = eltC(env_cur[x]);
        probs[x] = std::norm(amp);
    }

    double Z = 0.0;
    for(double p : probs) Z += p;
    if(Z > 0.0)
    {
        for(auto &p : probs) p /= Z;
    }

    return probs;
}


// ====================== depolarization on probabilities =======================
// p_noisy(x) = (1-p)*p(x) + p/2^N
void applyDepolarizationProb(std::vector<double> &probs, double p)
{
    std::size_t dim = probs.size();
    double uni = 1.0 / double(dim);

    // Update probs in place.
    for (std::size_t s = 0; s < dim; ++s)
    {
        probs[s] = (1.0 - p) * probs[s] + p * uni;
    }

    double Z = 0.0;
    for (std::size_t s = 0; s < dim; ++s) Z += probs[s];
    for (std::size_t s = 0; s < dim; ++s) probs[s] /= Z;
}


long double choose3(std::size_t M)
{
    if(M < 3) return 0.0L;
    long double m = static_cast<long double>(M);
    return m*(m-1)*(m-2)/6;
}

long double choose2(std::size_t M)
{
    if (M < 2) return 0.0L;
    long double m = static_cast<long double>(M);
    return m * (m - 1) / 2;
}

// ====================== main =======================

int
main(int argc, char* argv[])
{
    if(argc > 1 && std::string(argv[1]) == "--prepare-state")
    {
        if(argc < 3)
        {
            std::cerr << "Usage: ./main --prepare-state N [h]\n";
            return 2;
        }
        int stateN = std::stoi(argv[2]);
        double stateH = argc > 3 ? std::stod(argv[3]) : 1.0;
        if(stateN < 2)
        {
            std::cerr << "N must be at least 2\n";
            return 2;
        }
        std::ostringstream stateName;
        stateName << "MPS/ising_N" << stateN << "_J1.00_h"
                  << std::fixed << std::setprecision(2) << stateH << ".mps";
        std::ifstream existingState(stateName.str());
        if(existingState.good())
        {
            std::cout << "State already exists: " << stateName.str() << "\n";
            return 0;
        }
        // The Bash --prepare-states command creates MPS/ first.
        // Reuse the original DMRG Hamiltonian and convergence settings.
        auto sites = SpinHalf(stateN, {"ConserveQNs=",false});
        makeTFIMGroundState(sites, 1.0, stateH);
        std::cout << "Prepared: " << stateName.str() << "\n";
        return 0;
    }
    // Default parameters
    int N        = 8;         // Number of qubits.
    int depth    = 35;        // Circuit depth.
    int nSamples = 100000000; // Fixed shot count for panel (b).
    int maxDim   = 500;       // Maximum bond dimension.
    double dprob = 0.2;
    double h = 1.0;
    int experiment_id = 0;  // Default: 0.
    std::string outputDir = "../../data/cbne_moment/TIM_complete";

    if(argc > 1) N        = std::stoi(argv[1]);
    if(argc > 2) depth    = std::stoi(argv[2]);
    if(argc > 3) nSamples = std::stoi(argv[3]);
    if(argc > 4) maxDim   = std::stoi(argv[4]);
    if(argc > 5) dprob    = std::stod(argv[5]);
    if(argc > 6) h        = std::stod(argv[6]);
    if(argc > 7) experiment_id = std::stoi(argv[7]);
    if(argc > 8) outputDir = argv[8];

    // Initial MPS: |00...0>.
    // for(int i = 1; i <= N; ++i) init.set(i,"Dn");

    // Build a GHZ state.

    double J = 1.0;
    MPS psi;
    std::ostringstream oss;
    oss << "MPS/ising_N" << N
        << "_J" << std::fixed << std::setprecision(2) << J
        << "_h" << std::fixed << std::setprecision(2) << h
        << ".mps";

    std::string fname = oss.str();
    readFromFile(fname, psi);
    if(length(psi) != N)
    {
        std::cerr << "Error: loaded psi length = " << length(psi)
                << " but N = " << N << "\n";
        return 1;
    }


    // Apply the Haar-random brickwork circuit.
    std::random_device rd;
    std::mt19937 rng(rd());
    applyBrickworkCircuit(psi,N,depth,rng,maxDim);

    // Compute all computational-basis probabilities.
    auto probs = computeProbDistribution(psi);
    std::size_t dim = (std::size_t(1) << N);
    applyDepolarizationProb(probs, dprob);
    // Count sampled outcomes
    std::vector<std::size_t> counts(dim,0);

    // Reuse one discrete distribution for all draws.
    std::discrete_distribution<std::size_t> dist(probs.begin(),probs.end());

std::vector<std::size_t> checklist = {static_cast<std::size_t>(nSamples)};
std::size_t check_idx = 0; // Index of the next checkpoint.

auto write_checkpoint = [&](std::size_t curSamples) -> int {
    // Compute sum_C3
    long double sum_C3 = 0.0L;
    long double sum_C2 = 0.0L;
    for (std::size_t x = 0; x < dim; ++x) {
        sum_C3 += choose3(counts[x]);
        sum_C2 += choose2(counts[x]);
    }

    long double S_ld = sum_C3;
    long double S_nd = sum_C2;
    // d_ld = 2^N (long double for calculation)
    long double d_ld = powl(2.0L, (long double)N);

    // numerator = S * (d+2)(d+1)
    long double numerator = S_ld * (d_ld + 2.0L) * (d_ld + 1.0L);
    long double numerator_nd = S_nd * (d_ld + 1.0L);

    // denominator = C(curSamples,3)
    long double sample_ld = (long double)curSamples;
    long double denom = sample_ld * (sample_ld - 1.0L) * (sample_ld - 2.0L) / 6.0L;
    long double denom_nd = sample_ld * (sample_ld - 1.0L) / 2.0L;

    long double F = numerator / denom;
    long double Purity = numerator_nd / denom_nd;


    // Write curSamples (s+1) as nSamples.
    std::ostringstream oss;
    oss << outputDir << "/n" << N
        << "_depth" << depth
        << "_J" << std::fixed << std::setprecision(2) << J
        << "_h" << std::fixed << std::setprecision(2) << h
        << "_p" << std::fixed << std::setprecision(2) << dprob
        << "_Sample" << curSamples
        << ".jsonl";

    std::string filename = oss.str();

    std::ofstream ofs(filename, std::ios::app);
    if (!ofs) {
        std::cerr << "Error: cannot open " << filename << " for writing\n";
        return 1;
    }

    ofs << "{";
    ofs << "\"experiment_id\":" << experiment_id << ",";
    ofs << "\"N\":" << N << ",";
    ofs << "\"depth\":" << depth << ",";
    ofs << "\"nSamples\":" << curSamples << ",";
    ofs << "\"MaxDim\":" << maxDim << ",";
    ofs << "\"p\":" << std::setprecision(2) << dprob << ",";
    ofs << "\"Purity\":" << std::setprecision(20) << Purity << ",";
    ofs << "\"F\":" << std::setprecision(20) << F;
    ofs << "}\n";

    return 0;
};

    for(int s = 0; s < nSamples; ++s)
    {
        std::size_t x = dist(rng); // x encodes a bitstring from 0 to 2^N-1.
        // Update counts.
        counts[x]++;
        std::size_t curSamples = s + 1;
        if (check_idx < checklist.size() && curSamples == checklist[check_idx]) {
            if (write_checkpoint(curSamples) != 0) return 1;
            ++check_idx;
        }
        
    }


    return 0;
}
