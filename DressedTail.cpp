/*-----------------*/
/* DressedTail.cpp */
/*-----------------*/
// 261110Cl 新規作成: EBSD のビーム選択の新方式 (着衣つきの裾) の 1 方向の解。研究版 (tools/BeamSelStudy/native_bench/dressed_fused.inc、v19) の式の写し。
//   段 1: H_SQ (nS × nQ) と X₀ = −w ∘ H_QS (w_q = conj(λ_q)/(|λ_q|² + δ²)) を共有の差の表から単精度で作り、H_b = H_SS + H_SQ X₀ の固有分解 (単精度の一般)、
//         近縮退の組ごとのモードの打ち切り (組の重み ‖V_C a_C‖、a = V_b⁻¹e₀)、X₀V_k = X₀ V_k。
//   疎な積: E·X₀V (裾どうしの結合 = 差だけの関数 U(d) の stencil、逆格子で近い行の塊の順、足し込み先はレジスタ)。
//   段 2: R = T + E X₀V (T = −X₀V / w)、Y = −R ∘ conj(d)/(|d|² + δ²) (d = λ_q − γ_j)、E_δ、K = V†(H_SS V + H_SQ Y) + (Y†Y)Γ + Y†E_δ、G = V†V + Y†Y、
//         G⁻¹K の固有分解、初期値、源 F_W = V†F'_SS V + C + C† + χ†diag(w)χ、B = Z†F_W Z。
//   CPU の版: AVX-512 の版 (__AVX512F__) は X を bf16 で持ち gather + compress で当たる相手を集める。U も bf16 (VDPBF16PS) は CPUID で AVX512_BF16 を確かめたときだけ。
//   それ以外 (AVX2・generic・ARM64) は同じ当たる相手と式を、X を単精度にしたスカラーの経路で (自動のベクトル化に任せる)。
#ifndef NOMINMAX
#define NOMINMAX // windows.h の min/max のマクロを止める (std::min/max と Eigen のため)
#endif
#include "stdafx.h"
#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
#define EIGENFUNCS_EXPORTS
#include <vector>
#include <complex>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <new>
#include <bit>
#include <../Eigen/Core>
#include <../Eigen/Dense>
#include <../Eigen/Eigenvalues>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#include <intrin.h>
#define DT_X86 1
#endif
#include "DressedTail.h"

using namespace Eigen;
namespace {
using cd = std::complex<double>;
using cf = std::complex<float>;
using MZ = Matrix<cd, Dynamic, Dynamic>;
using MC = Matrix<cf, Dynamic, Dynamic>;
using VZ = Matrix<cd, Dynamic, 1>;

constexpr int kAbiVersion = 1;

// /fp:fast でも消えない非有限値の検査 (指数部のビットがすべて 1)
inline bool IsFiniteBits(double x) { return (std::bit_cast<uint64_t>(x) & 0x7FF0000000000000ull) != 0x7FF0000000000000ull; }

inline uint32_t Bf16Bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7FFF + ((u >> 16) & 1); return u >> 16; } // 最近接偶数への丸め

#if defined(DT_X86)
bool CpuHasAvx512Bf16()
{
	int r[4];
	__cpuidex(r, 0, 0);
	if (r[0] < 7) return false;
	__cpuidex(r, 7, 1);
	return ((r[0] >> 5) & 1) != 0; // CPUID.(EAX=7, ECX=1):EAX[5] = AVX512_BF16
}
#endif

struct Tables
{
	std::vector<cf> diffU, diffF, phase; int DH = 0, DK = 0, DL = 0;
	std::vector<int> dHkl, offLin; std::vector<float> dRe, dIm; std::vector<uint32_t> bfRe, bfIm; int nD = 0;
	int H = 0, K = 0, L = 0, PH = 0, PK = 0, PL = 0;
	int nPhaseCols = 0, nSites = 0; std::vector<double> wsite;
	long long Bytes() const
	{
		return (long long)(diffU.capacity() + diffF.capacity() + phase.capacity()) * sizeof(cf) + (long long)(dHkl.capacity() + offLin.capacity()) * sizeof(int)
			+ (long long)(dRe.capacity() + dIm.capacity()) * sizeof(float) + (long long)(bfRe.capacity() + bfIm.capacity()) * sizeof(uint32_t) + (long long)wsite.capacity() * sizeof(double);
	}
};

struct Workspace
{
	std::vector<cf> hsq, x0, t, ex0, y, pc, w, invw; std::vector<uint32_t> xb; std::vector<float> xr, xi;
	std::vector<int> map3D, hq, ht, ord; std::vector<int64_t> key; std::vector<float> hre, him, yr, yi;
	std::vector<int> eq, ej; std::vector<cd> ev;
	size_t mapBox = 0; const Tables* mapOwner = nullptr;
	long long Bytes() const
	{
		return (long long)(hsq.capacity() + x0.capacity() + t.capacity() + ex0.capacity() + y.capacity() + pc.capacity() + w.capacity() + invw.capacity()) * sizeof(cf)
			+ (long long)xb.capacity() * 4 + (long long)(xr.capacity() + xi.capacity() + hre.capacity() + him.capacity() + yr.capacity() + yi.capacity()) * sizeof(float)
			+ (long long)(map3D.capacity() + hq.capacity() + ht.capacity() + ord.capacity() + eq.capacity() + ej.capacity()) * sizeof(int) + (long long)key.capacity() * 8 + (long long)ev.capacity() * sizeof(cd);
	}
};

#if defined(__AVX512F__)
// 足し込み先をレジスタに置く積和 (X は bf16 の組、U は単精度)。列を 16 個 × NV の束に分けて当たる相手の一覧を走る
template <int NV>
inline void StencilAccum(int nh, const int* hq, const float* hre, const float* him, const uint32_t* xb, int kp, int c0, float* yr, float* yi)
{
	__m512 ar[NV], ai[NV];
	for (int v = 0; v < NV; v++) { ar[v] = _mm512_setzero_ps(); ai[v] = _mm512_setzero_ps(); }
	const __m512i hiMask = _mm512_set1_epi32((int)0xFFFF0000u);
	for (int j = 0; j < nh; j++)
	{
		const uint32_t* Xb = xb + (size_t)hq[j] * kp + c0;
		__m512 va = _mm512_set1_ps(hre[j]), vb = _mm512_set1_ps(him[j]);
		for (int v = 0; v < NV; v++)
		{
			__m512i w = _mm512_loadu_si512(Xb + 16 * v);
			__m512 a_r = _mm512_castsi512_ps(_mm512_slli_epi32(w, 16)), a_i = _mm512_castsi512_ps(_mm512_and_si512(w, hiMask));
			ar[v] = _mm512_fnmadd_ps(vb, a_i, _mm512_fmadd_ps(va, a_r, ar[v]));
			ai[v] = _mm512_fmadd_ps(vb, a_r, _mm512_fmadd_ps(va, a_i, ai[v]));
		}
	}
	for (int v = 0; v < NV; v++) { _mm512_storeu_ps(yr + c0 + 16 * v, ar[v]); _mm512_storeu_ps(yi + c0 + 16 * v, ai[v]); }
}

#if defined(__clang__)
#define DT_HAS_BF16_PATH 1
// U も bf16 (組の内積 VDPBF16PS): 実部 = (x_r, x_i)·(u_r, −u_i)、虚部 = (x_r, x_i)·(u_i, u_r)。CPUID で AVX512_BF16 を確かめたときだけ呼ぶ
template <int NV>
__attribute__((target("avx512bf16,avx512f,avx512bw,avx512vl")))
inline void StencilAccumBf(int nh, const int* hq, const uint32_t* bre, const uint32_t* bim, const uint32_t* xb, int kp, int c0, float* yr, float* yi)
{
	__m512 ar[NV], ai[NV];
	for (int v = 0; v < NV; v++) { ar[v] = _mm512_setzero_ps(); ai[v] = _mm512_setzero_ps(); }
	for (int j = 0; j < nh; j++)
	{
		const uint32_t* Xb = xb + (size_t)hq[j] * kp + c0;
		__m512bh br = (__m512bh)_mm512_set1_epi32((int)bre[j]), bi = (__m512bh)_mm512_set1_epi32((int)bim[j]);
		for (int v = 0; v < NV; v++)
		{
			__m512bh w = (__m512bh)_mm512_loadu_si512(Xb + 16 * v);
			ar[v] = _mm512_dpbf16_ps(ar[v], w, br);
			ai[v] = _mm512_dpbf16_ps(ai[v], w, bi);
		}
	}
	for (int v = 0; v < NV; v++) { _mm512_storeu_ps(yr + c0 + 16 * v, ar[v]); _mm512_storeu_ps(yi + c0 + 16 * v, ai[v]); }
}
__attribute__((target("avx512bf16,avx512f,avx512bw,avx512vl")))
void AccumRowBf(int n, const int* hq, const uint32_t* bre, const uint32_t* bim, const uint32_t* xb, int kp, int nv, float* yr, float* yi)
{
	int v0 = 0;
	for (; v0 + 8 <= nv; v0 += 8) StencilAccumBf<8>(n, hq, bre, bim, xb, kp, 16 * v0, yr, yi);
	for (; v0 + 4 <= nv; v0 += 4) StencilAccumBf<4>(n, hq, bre, bim, xb, kp, 16 * v0, yr, yi);
	for (; v0 < nv; v0++) StencilAccumBf<1>(n, hq, bre, bim, xb, kp, 16 * v0, yr, yi);
}
#endif
#endif

bool g_bf16Checked = false, g_bf16Available = false;
bool Bf16Available()
{
#if defined(DT_HAS_BF16_PATH)
	if (!g_bf16Checked) { g_bf16Available = CpuHasAvx512Bf16(); g_bf16Checked = true; } // 同じ値を書くだけなので並列の競合は無害
	return g_bf16Available;
#else
	return false;
#endif
}

// 疎な積 EX₀V = E·X₀V (nQ × nM、列優先)。X₀V = ws.t
void SparseProduct(const Tables& c, Workspace& ws, bool useUBf16, int nQ, int nM, const int* qHkl, const double* Dq)
{
	const int nk = 2 * c.K + 1, nl = 2 * c.L + 1;
	const int nkP = 2 * (c.K + c.PK) + 1, nlP = 2 * (c.L + c.PL) + 1; // 余白つきの箱 (行 + 差がいつも箱の中 → 境界の検査なし)
	const size_t box = (size_t)(2 * (c.H + c.PH) + 1) * nkP * nlP;
	if (ws.mapOwner != &c || ws.mapBox != box || ws.map3D.size() != box) { ws.map3D.assign(box, -1); ws.mapBox = box; ws.mapOwner = &c; }
	auto MI = [&](int h, int k, int l) { return ((size_t)(h + c.H + c.PH) * nkP + k + c.K + c.PK) * nlP + l + c.L + c.PL; };
	for (int r = 0; r < nQ; r++) ws.map3D[MI(qHkl[3 * r], qHkl[3 * r + 1], qHkl[3 * r + 2])] = r;
	// 行の並べ替え: 4×4×4 の指数の箱の番号 → 箱の中の (h, k, l)
	const int nh = 2 * c.H + 1, nkT = (nk + 3) / 4, nlT = (nl + 3) / 4;
	const int64_t nLin = (int64_t)nh * nk * nl;
	ws.key.resize(nQ); ws.ord.resize(nQ);
	for (int r = 0; r < nQ; r++)
	{
		int hh = qHkl[3 * r] + c.H, kk = qHkl[3 * r + 1] + c.K, ll = qHkl[3 * r + 2] + c.L;
		int64_t tile = ((int64_t)(hh >> 2) * nkT + (kk >> 2)) * nlT + (ll >> 2);
		ws.key[r] = tile * nLin + ((int64_t)hh * nk + kk) * nl + ll; ws.ord[r] = r;
	}
	std::sort(ws.ord.begin(), ws.ord.end(), [&](int a, int b) { return ws.key[a] < ws.key[b]; });
	ws.hq.resize(c.nD + 16); ws.ht.resize(c.nD + 16); ws.hre.resize(c.nD + 16); ws.him.resize(c.nD + 16);
	ws.ex0.resize((size_t)nQ * nM);
	const int* mp = ws.map3D.data();
#if defined(__AVX512F__)
	const int kp = (nM + 15) & ~15, nv = kp / 16;
	ws.yr.resize(kp); ws.yi.resize(kp);
	ws.xb.assign((size_t)nQ * kp, 0u);
	for (int col = 0; col < nM; col++)
	{
		const cf* xc = ws.t.data() + (size_t)col * nQ;
		for (int r = 0; r < nQ; r++) { float dv = (float)Dq[r]; cf x = xc[r]; ws.xb[(size_t)r * kp + col] = Bf16Bits(x.real() * dv) | (Bf16Bits(x.imag() * dv) << 16); }
	}
	const __m512i iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), neg1 = _mm512_set1_epi32(-1);
	for (int o = 0; o < nQ; o++)
	{
		int r = ws.ord[o], n = 0;
		// 当たる相手の一覧: 16 個の差ずつ表を gather で引き、裾にある (≥ 0) ものを compress で詰める (差の番号の順は保つ)
		__m512i vbase = _mm512_set1_epi32((int)MI(qHkl[3 * r], qHkl[3 * r + 1], qHkl[3 * r + 2]));
		for (int t0 = 0; t0 < c.nD; t0 += 16)
		{
			__mmask16 lm = t0 + 16 <= c.nD ? (__mmask16)0xFFFF : (__mmask16)((1u << (c.nD - t0)) - 1);
			__m512i idx = _mm512_add_epi32(vbase, _mm512_maskz_loadu_epi32(lm, c.offLin.data() + t0));
			__m512i v = _mm512_mask_i32gather_epi32(neg1, lm, idx, mp, 4);
			__mmask16 m = _mm512_mask_cmpge_epi32_mask(lm, v, _mm512_setzero_si512());
			_mm512_mask_compressstoreu_epi32(ws.hq.data() + n, m, v);
			_mm512_mask_compressstoreu_epi32(ws.ht.data() + n, m, _mm512_add_epi32(_mm512_set1_epi32(t0), iota));
			n += _mm_popcnt_u32(m);
		}
#if defined(DT_HAS_BF16_PATH)
		if (useUBf16)
		{
			uint32_t* bre = (uint32_t*)ws.hre.data(); uint32_t* bim = (uint32_t*)ws.him.data();
			for (int j = 0; j < n; j++) { int t = ws.ht[j]; bre[j] = c.bfRe[t]; bim[j] = c.bfIm[t]; }
			AccumRowBf(n, ws.hq.data(), bre, bim, ws.xb.data(), kp, nv, ws.yr.data(), ws.yi.data());
		}
		else
#endif
		{
			for (int j = 0; j < n; j++) { int t = ws.ht[j]; ws.hre[j] = c.dRe[t]; ws.him[j] = c.dIm[t]; }
			int v0 = 0;
			for (; v0 + 8 <= nv; v0 += 8) StencilAccum<8>(n, ws.hq.data(), ws.hre.data(), ws.him.data(), ws.xb.data(), kp, 16 * v0, ws.yr.data(), ws.yi.data());
			for (; v0 + 4 <= nv; v0 += 4) StencilAccum<4>(n, ws.hq.data(), ws.hre.data(), ws.him.data(), ws.xb.data(), kp, 16 * v0, ws.yr.data(), ws.yi.data());
			for (; v0 < nv; v0++) StencilAccum<1>(n, ws.hq.data(), ws.hre.data(), ws.him.data(), ws.xb.data(), kp, 16 * v0, ws.yr.data(), ws.yi.data());
		}
		float dw = (float)Dq[r];
		for (int col = 0; col < nM; col++) ws.ex0[r + (size_t)col * nQ] = cf(ws.yr[col] * dw, ws.yi[col] * dw);
	}
#else
	(void)useUBf16;
	// AVX-512 の無い版: X は単精度 (行優先の実部・虚部)、当たる相手と足す順は同じ
	const int kp = nM;
	ws.yr.resize(kp); ws.yi.resize(kp);
	ws.xr.resize((size_t)nQ * kp); ws.xi.resize((size_t)nQ * kp);
	for (int col = 0; col < nM; col++)
	{
		const cf* xc = ws.t.data() + (size_t)col * nQ;
		for (int r = 0; r < nQ; r++) { float dv = (float)Dq[r]; ws.xr[(size_t)r * kp + col] = xc[r].real() * dv; ws.xi[(size_t)r * kp + col] = xc[r].imag() * dv; }
	}
	for (int o = 0; o < nQ; o++)
	{
		int r = ws.ord[o], n = 0;
		const size_t base = MI(qHkl[3 * r], qHkl[3 * r + 1], qHkl[3 * r + 2]);
		for (int t = 0; t < c.nD; t++)
		{
			int qv = mp[(ptrdiff_t)base + c.offLin[t]];
			if (qv >= 0) { ws.hq[n] = qv; ws.hre[n] = c.dRe[t]; ws.him[n] = c.dIm[t]; n++; }
		}
		float* yr = ws.yr.data(); float* yi = ws.yi.data();
		std::fill(yr, yr + kp, 0.f); std::fill(yi, yi + kp, 0.f);
		for (int j = 0; j < n; j++)
		{
			const float* Xr = ws.xr.data() + (size_t)ws.hq[j] * kp; const float* Xi = ws.xi.data() + (size_t)ws.hq[j] * kp;
			const float ur = ws.hre[j], ui = ws.him[j];
			for (int col = 0; col < kp; col++) { yr[col] += ur * Xr[col] - ui * Xi[col]; yi[col] += ur * Xi[col] + ui * Xr[col]; }
		}
		float dw = (float)Dq[r];
		for (int col = 0; col < nM; col++) ws.ex0[r + (size_t)col * nQ] = cf(yr[col] * dw, yi[col] * dw);
	}
#endif
	for (int r = 0; r < nQ; r++) ws.map3D[MI(qHkl[3 * r], qHkl[3 * r + 1], qHkl[3 * r + 2])] = -1;
}

int DressedSolveImpl(const Tables& c, Workspace& ws, const DressedTailSettings& st, int nS, int nQ, const int* sHkl, const int* qHkl, const double* Ds, const double* Dq,
	const cd* hss, const cd* lamQ, const int* qPhaseCol, double sqrtP0, cd* gammaOut, cd* sourceOut, int capacity, int* nMOut, double* seconds)
{
	using clk = std::chrono::steady_clock;
	auto t0 = clk::now();
	auto lap = [&](int i) { auto t1 = clk::now(); if (seconds) seconds[i] += std::chrono::duration<double>(t1 - t0).count(); t0 = t1; };
	const int dnk = 2 * c.DK + 1, dnl = 2 * c.DL + 1;
	auto DI = [&](int h, int k, int l) { return ((size_t)(h + c.DH) * dnk + (k + c.DK)) * dnl + (l + c.DL); };
	const double delta = 1.0 / (3.141592653589793 * st.RegDepthNm), d2 = delta * delta;
	ws.w.resize(nQ); ws.invw.resize(nQ);
	for (int r = 0; r < nQ; r++)
	{
		cd l = lamQ[r]; cd w = std::conj(l) / (std::norm(l) + d2);
		ws.w[r] = cf(w); ws.invw[r] = cf(1.0 / w);
	}
	// 段 1: H_SQ (nS × nQ) と X₀ = −w ∘ H_QS (nQ × nS)、単精度で差の表から直接
	ws.hsq.resize((size_t)nS * nQ); ws.x0.resize((size_t)nQ * nS);
	for (int r = 0; r < nQ; r++)
	{
		int hq = qHkl[3 * r], kq = qHkl[3 * r + 1], lq = qHkl[3 * r + 2]; double dq = Dq[r];
		cf* col = ws.hsq.data() + (size_t)r * nS;
		for (int s = 0; s < nS; s++) col[s] = c.diffU[DI(hq - sHkl[3 * s], kq - sHkl[3 * s + 1], lq - sHkl[3 * s + 2])] * (float)(dq * Ds[s]); // H_SQ(s, r) = U(q − s) D_s D_q
	}
	for (int s = 0; s < nS; s++)
	{
		int hs = sHkl[3 * s], ks = sHkl[3 * s + 1], ls = sHkl[3 * s + 2]; double ds = Ds[s];
		cf* col = ws.x0.data() + (size_t)s * nQ;
		for (int r = 0; r < nQ; r++) col[r] = -ws.w[r] * (c.diffU[DI(hs - qHkl[3 * r], ks - qHkl[3 * r + 1], ls - qHkl[3 * r + 2])] * (float)(Dq[r] * ds)); // H_QS(r, s) = U(s − q) D_q D_s
	}
	lap(5);
	Map<MC> HSQ(ws.hsq.data(), nS, nQ), X0(ws.x0.data(), nQ, nS);
	MZ Hb = Map<const MZ>(hss, nS, nS) + MC(HSQ * X0).cast<cd>();
	lap(6);
	ComplexEigenSolver<MC> esf(Hb.cast<cf>(), true);
	if (esf.info() != Success) return DT_FAIL_STAGE1_EVD;
	VZ gb = esf.eigenvalues().cast<cd>(); MZ Vb = esf.eigenvectors().cast<cd>();
	// モードの打ち切り (近縮退の組、組の重み ‖V_C a_C‖、a = V_b⁻¹e₀)
	if (st.ModeTau > 0)
	{
		VZ e0 = VZ::Zero(nS); e0(0) = 1; VZ a = PartialPivLU<MZ>(Vb).solve(e0);
		std::vector<int> par(nS);
		for (int j = 0; j < nS; j++) par[j] = j;
		auto find = [&](int x) { while (par[x] != x) x = par[x] = par[par[x]]; return x; };
		if (st.DegTol > 0)
			for (int i = 0; i < nS; i++)
				for (int j = i + 1; j < nS; j++)
					if (std::abs(gb(i) - gb(j)) < st.DegTol) { int ri = find(i), rj = find(j); if (ri != rj) par[ri] = rj; }
		std::vector<VZ> gv(nS); std::vector<char> used(nS, 0); std::vector<double> gw(nS, 0); double wmax = 0;
		for (int j = 0; j < nS; j++) { int r = find(j); if (!used[r]) { gv[r] = VZ::Zero(nS); used[r] = 1; } gv[r] += Vb.col(j) * a(j); }
		for (int r = 0; r < nS; r++) if (used[r]) { gw[r] = gv[r].norm(); wmax = std::max(wmax, gw[r]); }
		std::vector<int> keep;
		for (int j = 0; j < nS; j++) if (gw[find(j)] >= st.ModeTau * wmax) keep.push_back(j);
		const int nK = (int)keep.size();
		MZ Vk(nS, nK); VZ gk(nK);
		for (int t = 0; t < nK; t++) { Vk.col(t) = Vb.col(keep[t]); gk(t) = gb(keep[t]); }
		Vb = Vk; gb = gk;
	}
	const int nM = (int)Vb.cols();
	lap(7);
	ws.t.resize((size_t)nQ * nM);
	Map<MC> X0V(ws.t.data(), nQ, nM);
	X0V.noalias() = X0 * Vb.cast<cf>(); // X₀V_k (= −w ∘ T)
	lap(8);
	SparseProduct(c, ws, st.AllowUBf16 && Bf16Available(), nQ, nM, qHkl, Dq);
	lap(2);
	// 段 2: R = T + E X₀V (T = −X₀V / w)、Y、E_δ
	ws.y.resize((size_t)nQ * nM);
	Map<MC> Y(ws.y.data(), nQ, nM);
	ws.eq.clear(); ws.ej.clear(); ws.ev.clear();
	for (int j = 0; j < nM; j++)
	{
		cd g = gb(j);
		const cf* xv = ws.t.data() + (size_t)j * nQ; const cf* ex = ws.ex0.data() + (size_t)j * nQ; cf* yc = ws.y.data() + (size_t)j * nQ;
		for (int q = 0; q < nQ; q++)
		{
			cd R = cd(-xv[q] * ws.invw[q] + ex[q]);
			cd d = lamQ[q] - g;
			double den = std::norm(d) + d2;
			yc[q] = cf(-R * std::conj(d) / den);
			double f = d2 / den;
			if (f > st.EdeltaThreshold) { ws.eq.push_back(q); ws.ej.push_back(j); ws.ev.push_back(R * f); }
		}
	}
	MZ YY = MZ::Zero(nM, nM); // Y†Y (倍精度で、行の塊ごとに)
	for (int r0 = 0; r0 < nQ; r0 += 2048)
	{
		int nr = std::min(2048, nQ - r0);
		MZ Yc = Y.middleRows(r0, nr).cast<cd>();
		YY.noalias() += Yc.adjoint() * Yc;
	}
	MZ HSQY = MC(HSQ * Y).cast<cd>();
	Map<MC> FSQ(ws.x0.data(), nS, nQ); // F'_SQ を X₀ の領域に作る (X₀ はもう使わない)
	for (int r = 0; r < nQ; r++)
	{
		int hq = qHkl[3 * r], kq = qHkl[3 * r + 1], lq = qHkl[3 * r + 2]; double dq = Dq[r];
		cf* col = ws.x0.data() + (size_t)r * nS;
		for (int s = 0; s < nS; s++) col[s] = c.diffF[DI(sHkl[3 * s] - hq, sHkl[3 * s + 1] - kq, sHkl[3 * s + 2] - lq)] * (float)(Ds[s] * dq); // F'_SQ(s, r) = F(s − q) D_s D_q
	}
	MZ FSQY = MC(FSQ * Y).cast<cd>();
	Map<const MZ> HSS(hss, nS, nS);
	MZ K = Vb.adjoint() * (HSS * Vb + HSQY);
	for (int j = 0; j < nM; j++) K.col(j) += YY.col(j) * gb(j);
	for (size_t t = 0; t < ws.eq.size(); t++) K.col(ws.ej[t]) += Y.row(ws.eq[t]).adjoint().cast<cd>() * ws.ev[t];
	MZ G = Vb.adjoint() * Vb + YY;
	VZ e0(nM);
	for (int j = 0; j < nM; j++) e0(j) = std::conj(Vb(0, j)) * sqrtP0;
	LLT<MZ> ch(G);
	if (ch.info() != Success) return DT_FAIL_GRAM;
	MZ GK = ch.solve(K);
	ComplexEigenSolver<MZ> es(GK, true);
	if (es.info() != Success) return DT_FAIL_FINAL_EVD;
	PartialPivLU<MZ> lu(es.eigenvectors());
	VZ aa = lu.solve(ch.solve(e0));
	MZ Z = es.eigenvectors() * aa.asDiagonal();
	// 源: F_W = V†F'_SS V + C + C† + χ†diag(w)χ (C = V†(F'_SQ Y)、χ = Φ Y)
	MZ FSS(nS, nS);
	for (int b2 = 0; b2 < nS; b2++)
		for (int a2 = 0; a2 < nS; a2++)
			FSS(a2, b2) = cd(c.diffF[DI(sHkl[3 * a2] - sHkl[3 * b2], sHkl[3 * a2 + 1] - sHkl[3 * b2 + 1], sHkl[3 * a2 + 2] - sHkl[3 * b2 + 2])]) * (Ds[a2] * Ds[b2]);
	MZ cross = Vb.adjoint() * FSQY;
	const int ns = c.nSites, chunk = 256;
	MC chi = MC::Zero(ns, nM);
	ws.pc.resize((size_t)ns * chunk);
	for (int r0 = 0; r0 < nQ; r0 += chunk)
	{
		int nr = std::min(chunk, nQ - r0);
		for (int rr = 0; rr < nr; rr++)
		{
			int r = r0 + rr, col = qPhaseCol[r]; float dq = (float)Dq[r];
			cf* dst = ws.pc.data() + (size_t)rr * ns;
			if (col < 0 || col >= c.nPhaseCols) { std::fill(dst, dst + ns, cf(0)); continue; }
			const cf* src = c.phase.data() + (size_t)col * ns;
			for (int i = 0; i < ns; i++) dst[i] = src[i] * dq;
		}
		chi.noalias() += Map<MC>(ws.pc.data(), ns, nr) * Y.middleRows(r0, nr);
	}
	MC chiW = chi;
	for (int i = 0; i < ns; i++) chiW.row(i) *= (float)c.wsite[i];
	MZ FW = Vb.adjoint() * FSS * Vb + cross + cross.adjoint() + MC(chi.adjoint() * chiW).cast<cd>();
	MZ B = Z.adjoint() * FW * Z;
	if (nM > capacity) return DT_FAIL_CAPACITY;
	for (int j = 0; j < nM; j++) if (!IsFiniteBits(es.eigenvalues()(j).real()) || !IsFiniteBits(es.eigenvalues()(j).imag())) return DT_FAIL_NONFINITE;
	for (int k = 0; k < nM * nM; k++) if (!IsFiniteBits(B.data()[k].real()) || !IsFiniteBits(B.data()[k].imag())) return DT_FAIL_NONFINITE;
	Map<VZ>(gammaOut, nM) = es.eigenvalues();
	Map<MZ>(sourceOut, nM, nM) = B;
	*nMOut = nM;
	lap(4);
	return DT_OK;
}
} // namespace

extern "C" {
EIGEN_FUNCS_API int _DressedTailAbiVersion() { return kAbiVersion; }

EIGEN_FUNCS_API int _DressedTailCapabilities()
{
	int caps = 0;
#if defined(__AVX512F__)
	caps |= 1;
#endif
	if (Bf16Available()) caps |= 2;
#if defined(__AVX2__)
	caps |= 4;
#endif
	return caps;
}

EIGEN_FUNCS_API void* _DressedTailCreateTables(const double* diffU, const double* diffF, int diffLength, int DH, int DK, int DL,
	const int* dHkl, const float* dRe, const float* dIm, int nD, int H, int K, int L,
	const double* phase, int nPhaseCols, int nSites, const double* wsite, int* status)
{
	if (status) *status = DT_OK;
	if (!diffU || !diffF || diffLength <= 0 || DH < 0 || DK < 0 || DL < 0 || (long long)(2 * DH + 1) * (2 * DK + 1) * (2 * DL + 1) != diffLength
		|| nD < 0 || (nD > 0 && (!dHkl || !dRe || !dIm)) || H < 0 || K < 0 || L < 0 || nPhaseCols < 0 || nSites < 0 || (nPhaseCols > 0 && nSites > 0 && (!phase || !wsite)))
	{
		if (status) *status = DT_FAIL_ARGS; return nullptr;
	}
	try
	{
		auto t = new Tables();
		t->DH = DH; t->DK = DK; t->DL = DL;
		t->diffU.resize(diffLength); t->diffF.resize(diffLength);
		for (int i = 0; i < diffLength; i++) { t->diffU[i] = cf((float)diffU[2 * i], (float)diffU[2 * i + 1]); t->diffF[i] = cf((float)diffF[2 * i], (float)diffF[2 * i + 1]); }
		t->nD = nD; t->dHkl.assign(dHkl, dHkl + 3 * (size_t)nD); t->dRe.assign(dRe, dRe + nD); t->dIm.assign(dIm, dIm + nD);
		t->H = H; t->K = K; t->L = L;
		for (int i = 0; i < nD; i++) { t->PH = std::max(t->PH, std::abs(dHkl[3 * i])); t->PK = std::max(t->PK, std::abs(dHkl[3 * i + 1])); t->PL = std::max(t->PL, std::abs(dHkl[3 * i + 2])); }
		const int nkP = 2 * (K + t->PK) + 1, nlP = 2 * (L + t->PL) + 1;
		t->offLin.resize(nD); t->bfRe.resize(nD); t->bfIm.resize(nD);
		for (int i = 0; i < nD; i++)
		{
			t->offLin[i] = (dHkl[3 * i] * nkP + dHkl[3 * i + 1]) * nlP + dHkl[3 * i + 2];
			uint32_t ur = Bf16Bits(dRe[i]), ui = Bf16Bits(dIm[i]), uin = Bf16Bits(-dIm[i]);
			t->bfRe[i] = ur | (uin << 16); t->bfIm[i] = ui | (ur << 16);
		}
		t->nPhaseCols = nPhaseCols; t->nSites = nSites;
		t->phase.resize((size_t)nPhaseCols * nSites);
		for (size_t i = 0; i < t->phase.size(); i++) t->phase[i] = cf((float)phase[2 * i], (float)phase[2 * i + 1]);
		t->wsite.assign(wsite, wsite + nSites);
		return t;
	}
	catch (const std::bad_alloc&) { if (status) *status = DT_FAIL_OOM; return nullptr; }
	catch (...) { if (status) *status = DT_FAIL_UNKNOWN; return nullptr; }
}

EIGEN_FUNCS_API void _DressedTailFreeTables(void* tables) { delete static_cast<Tables*>(tables); }
EIGEN_FUNCS_API long long _DressedTailTablesBytes(const void* tables) { return tables ? static_cast<const Tables*>(tables)->Bytes() : 0; }

EIGEN_FUNCS_API void* _DressedTailCreateWorkspace(int* status)
{
	if (status) *status = DT_OK;
	try { return new Workspace(); }
	catch (const std::bad_alloc&) { if (status) *status = DT_FAIL_OOM; return nullptr; }
	catch (...) { if (status) *status = DT_FAIL_UNKNOWN; return nullptr; }
}
EIGEN_FUNCS_API void _DressedTailFreeWorkspace(void* workspace) { delete static_cast<Workspace*>(workspace); }
EIGEN_FUNCS_API long long _DressedTailWorkspaceBytes(const void* workspace) { return workspace ? static_cast<const Workspace*>(workspace)->Bytes() : 0; }

EIGEN_FUNCS_API int _DressedTailSolve(const void* tables, void* workspace, const DressedTailSettings* settings,
	int nS, int nQ, const int* sHkl, const int* qHkl, const double* Ds, const double* Dq,
	const double* hss, const double* lamQ, const int* qPhaseCol, double sqrtP0,
	double* gammaOut, double* sourceOut, int capacity, int* nM, double* seconds)
{
	if (nM) *nM = 0;
	if (!tables || !workspace || !settings || settings->AbiVersion != kAbiVersion || nS <= 0 || nQ < 0 || !sHkl || !Ds || !hss || !gammaOut || !sourceOut || !nM
		|| (nQ > 0 && (!qHkl || !Dq || !lamQ || !qPhaseCol)) || !(settings->RegDepthNm > 0))
		return DT_FAIL_ARGS;
	if (capacity < nS) return DT_FAIL_CAPACITY;
	try
	{
		int rc = DressedSolveImpl(*static_cast<const Tables*>(tables), *static_cast<Workspace*>(workspace), *settings, nS, nQ, sHkl, qHkl, Ds, Dq,
			reinterpret_cast<const cd*>(hss), reinterpret_cast<const cd*>(lamQ), qPhaseCol, sqrtP0,
			reinterpret_cast<cd*>(gammaOut), reinterpret_cast<cd*>(sourceOut), capacity, nM, seconds);
		if (rc != DT_OK) *nM = 0;
		return rc;
	}
	catch (const std::bad_alloc&) { *nM = 0; return DT_FAIL_OOM; }
	catch (...) { *nM = 0; return DT_FAIL_UNKNOWN; }
}
} // extern "C"
