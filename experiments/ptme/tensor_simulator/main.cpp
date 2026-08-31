#include "itensor/all.h"
#include <random>
#include <complex>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>   // 用于保存 counts
#include <cstddef>   // std::size_t

using namespace itensor;

// ====================== Haar 随机 U(4) 生成 =======================

void
haarRandomU4(std::mt19937 &rng,
             std::complex<double> U[4][4])
{
    using C = std::complex<double>;
    std::normal_distribution<double> ndist(0.0,1.0);

    // 4x4 复高斯矩阵
    C Z[4][4];
    for(int i = 0; i < 4; ++i)
    for(int j = 0; j < 4; ++j)
    {
        double x = ndist(rng);
        double y = ndist(rng);
        Z[i][j] = C(x,y);
    }

    // 列 Gram-Schmidt -> Haar 随机 U(4)
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

// ====================== MPS 上作用两比特门 =======================

// 作用 gate G 在 (i,i+1) 上
// G 索引结构: (s1,s2,s1',s2')
void
applyTwoSiteGateMPS(MPS &psi,
                    ITensor const& G,
                    int i,
                    int maxDim)
{
    psi.position(i);

    auto wf = psi(i) * psi(i+1);

    // 更推荐：明确门作用方向，避免隐式匹配
    // 让门把 (s_i,s_{i+1}) -> (s_i',s_{i+1}')
    wf = G * wf;

    wf.noPrime();

    auto s1 = siteIndex(psi,i);
    auto li = leftLinkIndex(psi,i); // i==1 时可能不存在

    IndexSet left = (li ? IndexSet(li,s1) : IndexSet(s1));

    auto [U,S,V] = svd(wf, left,
                       Args("Cutoff=",1E-12,"MaxDim=",maxDim));

    psi.setA(i,U);
    psi.setA(i+1,S*V);

}

// 构造 Haar 随机两比特门并作用在 (i,i+1)
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

// ====================== TFIM 基态 (DMRG) =======================
// H = -J * sum sigma^z_i sigma^z_{i+1} - h * sum sigma^x_i
// 注意：SpinHalf 的 Sz/Sx 是自旋算符 (1/2)*sigma，所以要乘因子：
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

    // 初始 MPS（随便给个简单直积态）
    InitState init(sites);
    for(int i = 1; i <= N; ++i) init.set(i,"Dn");
    MPS psi0(init);

    Sweeps sweeps(8);
    sweeps.maxdim() = 20, 40, 80, 120, 200, 200, 200, 200;
    sweeps.cutoff() = 1E-12;

    auto energy = dmrg(H, psi0, sweeps, Args("Quiet=",true));
    std::ostringstream oss;
    oss << "MPS/ising_N" << N
        << "_J" << std::fixed << std::setprecision(2) << J
        << "_h" << std::fixed << std::setprecision(2) << h
        << ".mps";

    std::string fname = oss.str();
    writeToFile(fname, psi0);

    // dmrg 会把 psi0 就地更新成基态
    return psi0;
}


// ====================== GHZ 态构造（MPS） =======================

void
applyHadamardMPS(MPS &psi,
                 const SiteSet &sites,
                 int i)
{
    psi.position(i);
    auto s  = sites(i);
    auto sp = prime(s);

    ITensor H(sp,s);

    auto dn  = s(1);
    auto up  = s(2);
    auto dnp = sp(1);
    auto upp = sp(2);

    double isrt2 = 1.0/std::sqrt(2.0);

    H.set(dnp,dn,isrt2);
    H.set(dnp,up,isrt2);
    H.set(upp,dn,isrt2);
    H.set(upp,up,-isrt2);

    auto Ai = psi.A(i);
    Ai = H * Ai;
    Ai.noPrime();
    psi.setA(i,Ai);
}

// CNOT 控制 i -> 受控 i+1
void
applyCNOTMPS(MPS &psi,
             const SiteSet &sites,
             int i,
             int maxDim)
{
    auto j = i+1;
    auto s1  = sites(i);
    auto s2  = sites(j);
    auto s1p = prime(s1);
    auto s2p = prime(s2);

    ITensor G(s1,s2,s1p,s2p);

    auto dn1  = s1(1);
    auto up1  = s1(2);
    auto dn2  = s2(1);
    auto up2  = s2(2);

    auto dn1p = s1p(1);
    auto up1p = s1p(2);
    auto dn2p = s2p(1);
    auto up2p = s2p(2);

    // |00> -> |00>
    G.set(dn1,dn2,dn1p,dn2p,1.0);
    // |01> -> |01>
    G.set(dn1,up2,dn1p,up2p,1.0);
    // |10> -> |11>
    G.set(up1,dn2,up1p,up2p,1.0);
    // |11> -> |10>
    G.set(up1,up2,up1p,dn2p,1.0);

    applyTwoSiteGateMPS(psi,G,i,maxDim);
}

// |00..0> -> GHZ: (|00..0> + |11..1>)/sqrt(2)
void
makeGHZMPS(MPS &psi,
           const SiteSet &sites,
           int maxDim)
{
    int N = length(sites);

    applyHadamardMPS(psi,sites,1);
    for(int i = 1; i < N; ++i)
    {
        applyCNOTMPS(psi,sites,i,maxDim);
    }
}

// ====================== brickwork 浅层线路（MPS） =======================

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
            // 偶数层：作用 (1,2), (3,4), ...
            for(int i = 1; i <= N-1; i += 2)
            {
                applyRandomTwoQubitGateMPS(psi,i,rng,maxDim);
            }
        }
        else
        {
            // 奇数层：作用 (2,3), (4,5), ...
            for(int i = 2; i <= N-1; i += 2)
            {
                applyRandomTwoQubitGateMPS(psi,i,rng,maxDim);
            }
        }
    }
}

// ====================== 计算完整概率分布 =======================

// 把 MPS 收缩成一个 wavefunction ITensor（只带所有 site index）
ITensor
makeFullWavefunction(const MPS &psi)
{
    int N = length(psi);

    auto wf = psi(1);
    for(int i = 2; i <= N; ++i)
    {
        wf *= psi(i);
    }
    wf.noPrime();
    return wf;
}


// std::vector<double> computeProbDistribution(const MPS &psi, const SiteSet &sites) {
//     int N = length(sites); 
//     auto wf = makeFullWavefunction(psi); 
//     std::size_t dim = (std::size_t(1) << N); 
//     std::vector<double> probs(dim,0.0); 
//     for(std::size_t x = 0; x < dim; ++x) { 
//         ITensor ampT = wf; 
//         // 把 wf 投影到 |x> = |b1 b2 ... bN> 
//         for(int i = 1; i <= N; ++i) { 
//             int bit = ( (x >> (i-1)) & 1 ); 
//             auto s = sites(i); ITensor ket(s); 
//             ket.set( bit==0 ? s(1) : s(2), 1.0 ); ampT *= ket; 
//         } 
//         std::complex<double> amp = eltC(ampT); 
//         probs[x] = std::norm(amp); 
//     } 
//     // 归一化 
//     double Z = 0.0; 
//     for(auto p : probs) Z += p; 
//     if(Z > 0.0) { 
//         for(auto &p : probs) p /= Z; 
//     } 
//     return probs; 
// }


std::vector<double>
computeProbDistribution(const MPS &psi)
{
    const int N = length(psi);
    // Modified: Contract first N-2 sites in Z basis, last 2 in Bell/Orthogonal basis.
    if(N < 2) throw std::runtime_error("N must be >= 2");

    // 复制一份 MPS，方便调整规范
    auto psiC = psi;
    psiC.position(1); // 左规范化

    // New output dimension: 2^(N-2) prefixes * 2 outcomes = 2^(N-1)
    std::size_t mixed_dim = std::size_t(1) << (N - 1);
    std::vector<double> probs(mixed_dim, 0.0);

    std::vector<ITensor> env_cur, env_next;
    env_cur.reserve(1);
    env_cur.emplace_back(1.0); // 空前缀 = 标量 1

    // 1. Contract first N-2 sites (Standard Basis)
    for(int i = 1; i <= N-2; ++i)
    {
        env_next.clear();
        env_next.reserve(env_cur.size() * 2);

        // 关键：从 psiC 取 site index（而不是外部 sites）
        auto s  = siteIndex(psiC, i);
        auto Ai = psiC.A(i);  // site tensor

        // bra |0> 和 |1>
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

    // 2. Contract last 2 sites (Singlet & Orthogonal Complement)
    // env_cur now contains 2^(N-2) tensors
    auto s1 = siteIndex(psiC, N-1);
    auto s2 = siteIndex(psiC, N);
    
    // Construct |psi> = (|01> - |10>)/sqrt(2)
    // braSinglet = <psi|
    ITensor braSinglet(s1, s2);
    double is2 = 1.0/std::sqrt(2.0);
    // <01| term: s1=1 (0), s2=2 (1), coeff +1/sqrt(2)
    braSinglet.set(s1(1), s2(2), is2);
    // <10| term: s1=2 (1), s2=1 (0), coeff -1/sqrt(2)
    braSinglet.set(s1(2), s2(1), -is2);

    auto A1 = psiC.A(N-1);
    auto A2 = psiC.A(N);

    // Loop over all N-2 prefixes
    for(std::size_t k = 0; k < env_cur.size(); ++k)
    {
        // Contraction for the branch
        ITensor T = env_cur[k] * A1 * A2;

        // --- Outcome 1: Singlet Projector |psi><psi| ---
        std::complex<double> ampS = eltC(T * braSinglet);
        double pS = std::norm(ampS);

        // --- Outcome 2: Orthogonal Compl. (I - |psi><psi|) ---
        // Total probability on this branch = <T|T>
        double pTot = itensor::norm(T);
        pTot = pTot * pTot; 

        double pOther = pTot - pS;
        if(pOther < 0.0) pOther = 0.0; // safety

        probs[2*k]     = pS;
        probs[2*k + 1] = pOther;
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
// p_noisy(x) = (1-p)*p(x) + ...
// For singlet outcome (index 2*b):   (1-p)*prob + p/d
// For triplet outcome (index 2*b+1): (1-p)*prob + 3*p/d
// Note: here d refers to the full dimension 2^N.
// The reasoning: Singlet is 1 state, Triplet/Complement subspace has dimension 3 + (2^(N-2)-1)*4 ?
// Actually, the user requirement is specific:
// "For 2*b pos: (1-p)*Prob + p/d"
// "For 2*b+1 pos: (1-p)*Prob + 3*p/d"
// where d = 2^N (total Hilbert space dimension).
void applyDepolarizationProb(std::vector<double> &probs, double p, int N)
{
    // Total Hilbert space dimension d = 2^N
    // We use long double for intermediate precision
    double d_val = std::pow(2.0, N); 
    double noise_term_singlet = p / d_val;
    double noise_term_other   = 3.0 * p / d_val;

    std::size_t mixed_dim = probs.size(); // This is 2^(N-1)
    
    // We iterate pairs (2*b, 2*b+1)
    for (std::size_t k = 0; k < mixed_dim; k += 2)
    {
        // Index 2*b -> Singlet
        probs[k]     = (1.0 - p) * probs[k]     + noise_term_singlet;
        // Index 2*b+1 -> Others (Triplet + different basis)
        probs[k + 1] = (1.0 - p) * probs[k + 1] + noise_term_other;
    }

    // Normalization check (usually sum is preserved if input sum=1)
    // Singlet noise sum: (2^(N-2)) * (p/2^N) = p/4
    // Other noise sum:   (2^(N-2)) * (3p/2^N) = 3p/4
    // Total added noise = p. 
    // Total decayed signal = (1-p). Sum = 1. Correct.

    double Z = 0.0;
    for (double v : probs) Z += v;
    if(Z > 0.0)
    {
        for(auto &v : probs) v /= Z;
    }
}



long double combinations(unsigned long long n, int k)
{
    if (k < 0 || n < (unsigned long long)k) return 0.0;
    long double res = 1.0;
    for (int i = 0; i < k; ++i)
    {
        res = res * (long double)(n - i) / (long double)(i + 1);
    }
    return res;
}

// ====================== Exact PT Moments Calculation =======================
void computeAndSaveExactPTMoments(MPS& psi, int N, int depth, double J, double h, double dprob, int experiment_id)
{
    auto r = rightLinkIndex(psi, N-1);
    std::cout << "bond dim at (N-1|N) = " << dim(r) << "\n";
    // --- 0) 把正交中心放到 N-1，保证 cut (N-1|N) 的 Schmidt 信息在 A(N-1) 的右侧link上
    psi.position(N-1);

    // --- 1) 对 psi(N-1) 做 SVD，把 (leftLink + site(N-1)) 当成左侧指标集合
    // 这样 S 的对角元素就是 cut (N-1|N) 的 Schmidt 奇异值 s_i
    auto sNm1 = siteIndex(psi, N-1);

    IndexSet Uis;
    if(N-1 > 1)
    {
        auto l = leftLinkIndex(psi, N-1);
        Uis = IndexSet(l, sNm1);
    }
    else
    {
        Uis = IndexSet(sNm1);
    }

    // 3) 对 psi(N-1) 做 SVD：返回 (U,S,V)
    //    这样 S 的对角就是 cut (N-1|N) 的 Schmidt 奇异值 s_i
    ITensor U, S, V;
    std::tie(U, S, V) = svd(psi(N-1), Uis);

    // --- 2) 取出 S 的“奇异值指标”来循环（不要用 dim(1) 这种不稳写法）
    // S 通常是对角张量，两个指标同维；取第一个即可
    auto sidx = inds(S)(1);
    int d = dim(sidx);

    double sum_lam1  = 0.0; // Σ λ
    double sum_lam2  = 0.0; // Σ λ^2
    double sum_lam3  = 0.0; // Σ λ^3
    double sum_lam5  = 0.0; // Σ λ^5

    double sum_sqrt_lam = 0.0; // Σ √λ = Σ s_i (Trace Norm for Negativity)

    // std::cout << "Spectral Debug: ";
    for(int i = 1; i <= d; ++i)
    {
        // Schmidt 奇异值 s_i
        // 注意：elt(S, i, i) 这里的 (i,i) 对应 sidx 的两个对角位置
        double si = elt(S, sidx(i), sidx(i));
        
        // if(i <= 5) std::cout << si << " "; // Print top 5 singular values

        // λ_i = s_i^2
        double lam  = si * si;
        double lam2 = lam * lam;          // λ^2
        double lam3 = lam2 * lam;         // λ^3
        double lam5 = lam2 * lam2 * lam;  // λ^5

        sum_lam1 += lam;
        sum_lam2 += lam2;
        sum_lam3 += lam3;
        sum_lam5 += lam5;
        
        sum_sqrt_lam += si;
    }
    // std::cout << "...\n";

    // Apply formulas
    double M2 = sum_lam1 * sum_lam1;
    double M3 = sum_lam3;
    double M4 = sum_lam2 * sum_lam2;
    double M5 = sum_lam5;

    // Negativity Calculation
    // ||rho^TB||_1 = (sum s_i)^2
    // LogNeg = ln ||rho^TB||_1
    double trace_norm = sum_sqrt_lam * sum_sqrt_lam;
    double log_neg = std::log(trace_norm);
    
    std::cout << "Exact PT Moments: M2=" << M2 << " M3=" << M3 
              << " LogNeg=" << log_neg << " TraceNorm=" << trace_norm << "\n";
    if (std::abs(M3 - 1.0) < 1e-9) {
         std::cout << "[WARNING] M3 is 1.0. This implies the state is a PRODUCT STATE (no entanglement) across the cut.\n";
    }

    // Save to distinct file

    std::ostringstream oss;
    oss << "TIM_exact/n" << N
        << "_depth" << depth
        << "_J" << std::fixed << std::setprecision(2) << J
        << "_h" << std::fixed << std::setprecision(2) << h
        << "_exact.jsonl";

    std::string filename = oss.str();
    
    // Ensure directory exists (quick hack: simple check or system call, but user asked for C++ code)
    // We assume TIM_exact folder exists or we just write to root if not.
    // For safety, let's just write to same folder with different prefix if needed, 
    // but here I'll stick to the requested structure.
    
    // Try to open, if fail, maybe folder missing.
    std::ofstream ofs(filename, std::ios::app);
    if(ofs)
    {
        ofs << "{";
        ofs << "\"experiment_id\":" << experiment_id << ",";
        ofs << "\"N\":" << N << ",";
        ofs << "\"depth\":" << depth << ",";
        ofs << "\"type\":\"exact_PT_moments\",";
        ofs << "\"partition\":\"1..N-1|N\",";
        ofs << "\"M2\":" << std::setprecision(20) << M2 << ",";
        ofs << "\"M3\":" << std::setprecision(20) << M3 << ",";
        ofs << "\"M4\":" << std::setprecision(20) << M4 << ",";
        ofs << "\"M5\":" << std::setprecision(20) << M5;
        ofs << "}\n";
    }
    else
    {
        std::cerr << "Warning: Could not write exact moments to " << filename << "\n";
    }
}

// ====================== main =======================

int
main(int argc, char* argv[])
{
    // 默认参数
    int N        = 15;   // qubit 数
    int depth    = 4;    // brickwork 深度
    int nSamples = 50;   // 采样次数
    int maxDim   = 300;  // 最大 bond dimension
    double dprob = 0.0;
    double h = 1.0;
    int experiment_id = 0;  // 默认 0

    if(argc > 1) N        = std::stoi(argv[1]);
    if(argc > 2) depth    = std::stoi(argv[2]);
    if(argc > 3) nSamples = std::stoi(argv[3]);
    if(argc > 4) maxDim   = std::stoi(argv[4]);
    if(argc > 5) dprob    = std::stod(argv[5]);
    if(argc > 6) h        = std::stod(argv[6]);
    if(argc > 7) experiment_id = std::stoi(argv[7]);
    // double J = 1.0;
    // for (N = 5; N < 10; N++){
    auto sites = SpinHalf(N,{"ConserveQNs=",false});

    // 初态：|00...0> 的 MPS
    // InitState init(sites);
    // for(int i = 1; i <= N; ++i) init.set(i,"Dn");
    // MPS psi(init);

    // 构造 GHZ
    // makeGHZMPS(psi,sites,maxDim);

    double J = 1.0;
    // double h = 1.0;
    
    // --- OPTION 1: Read from file ---
    /*
    MPS psi;
    std::ostringstream oss;
    oss << "MPS/ising_N" << N
        << "_J" << std::fixed << std::setprecision(2) << J
        << "_h" << std::fixed << std::setprecision(2) << h
        << ".mps";

    std::string fname = oss.str();
    // Try to read, but if it fails or we want fresh, see below
    readFromFile(fname, psi);
    if(length(psi) != N)
    {
        std::cerr << "Error: loaded psi length = " << length(psi)
                << " but N = " << N << "\n";
        return 1;
    }
    */

    // --- OPTION 2: Generate fresh Ground State (Guarantees Entanglement) ---
    std::cout << "Generating Ising Ground State for N=" << N << " J=" << J << " h=" << h << "..." << std::endl;
    MPS psi = makeTFIMGroundState(sites, J, h);

    // Compute exact moments first (before Brickwork)
    computeAndSaveExactPTMoments(psi, N, depth, J, h, dprob, experiment_id);

    // brickwork 浅层 Haar 随机线路
    std::random_device rd;
    std::mt19937 rng(rd());
    // Use N-1 to ensure the circuit only acts on subsystem A, leaving qubit N (subsystem B) untouched.
    applyBrickworkCircuit(psi,N-1,depth,rng,maxDim);

    // 一次性算出整个标准基概率分布
    
    // Compute exact moments first
    // computeAndSaveExactPTMoments(psi, N, depth, J, h, dprob, experiment_id);

    auto probs = computeProbDistribution(psi);
    std::size_t dim = probs.size();
    applyDepolarizationProb(probs, dprob, N);
    // ===== 统计每个结果出现次数 =====
    std::vector<std::size_t> counts(dim,0);

    // 使用同一个离散分布对象做多次采样
    std::discrete_distribution<std::size_t> dist(probs.begin(),probs.end());

std::vector<std::size_t> checklist = {1000000};
std::size_t d = 1 << N;
// std::vector<std::size_t> checklist = {d, 2*d, 5*d, 10*d, 20*d, 50*d, 100*d};
// std::vector<std::size_t> checklist = {50*d};
std::size_t check_idx = 0; // 指向下一个要触发的 checklist 元素

    auto calc_moment = [&](int t, std::size_t curSamples) -> double {
        long double total_sum = 0.0;
        // dim is 2^(N-1), but we iterate over basis b which has size 2^(N-2)
        // dim = 2 * num_b
        std::size_t num_b = dim / 2; 

        for (std::size_t b = 0; b < num_b; ++b) {
            // probs structure: [2*b] -> Singlet (r=-1), [2*b+1] -> Others (r=+1)
            unsigned long long n_minus = counts[2 * b];     
            unsigned long long n_plus  = counts[2 * b + 1]; 
            
            if (n_minus + n_plus < (unsigned long long)t) continue;
            
            long double sum_for_b = 0.0;
            // sum_{k=0}^t (-1)^k C(n_minus, k) C(n_plus, t-k)
            for(int k=0; k<=t; ++k) {
                if ((unsigned long long)k > n_minus || (unsigned long long)(t - k) > n_plus) continue;
                
                long double term = combinations(n_minus, k) * combinations(n_plus, t - k);
                if (k % 2 == 1) sum_for_b -= term;
                else            sum_for_b += term;
            }
            total_sum += sum_for_b;
        }
        
        long double denom = combinations(curSamples, t);
        if(denom <= 1e-18) return 0.0;
        
        // Prefactor: d_A1^(t-1) * d_A2^t
        // d_A1 = 2^(N-2), d_A2 = 2
        // If N < 2, handled elsewhere, but N>=2 assumed
        long double d_A1 = std::pow(2.0, N - 2);
        long double d_A2 = 2.0;
        long double prefactor = std::pow(d_A1, t - 1) * std::pow(d_A2, t);
        
        return (double)(prefactor * total_sum / denom);
    };

    auto write_checkpoint = [&](std::size_t curSamples) -> int {
        double val_M2 = calc_moment(2, curSamples);
        double val_M3 = calc_moment(3, curSamples);
        double val_M4 = calc_moment(4, curSamples);
        double val_M5 = calc_moment(5, curSamples);

        // === 写入文件：nSamples 用 curSamples（也就是 s+1）===
        std::ostringstream oss;
        oss << "TIM_complete/n" << N
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
        ofs << "\"M2\":" << std::setprecision(20) << val_M2 << ",";
        ofs << "\"M3\":" << std::setprecision(20) << val_M3 << ",";
        ofs << "\"M4\":" << std::setprecision(20) << val_M4 << ",";
        ofs << "\"M5\":" << std::setprecision(20) << val_M5;
        ofs << "}\n";

        return 0;
    };

    for(int s = 0; s < nSamples; ++s)
    {
        std::size_t x = dist(rng); // x 是 0..2^N-1 的整数，对应某个 bitstring
        // 统计次数
        counts[x]++;
        std::size_t curSamples = s + 1;
        if (check_idx < checklist.size() && curSamples == checklist[check_idx]) {
            if (write_checkpoint(curSamples) != 0) return 1;
            ++check_idx;
        }
        
    }


    return 0;
}
