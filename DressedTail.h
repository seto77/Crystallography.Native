/*---------------*/
/* DressedTail.h */
/*---------------*/
// 261110Cl 新規作成: EBSD のビーム選択の新方式 (着衣つきの裾、作者の判断 2026-11-10「N300・τ3e−3・一体化」) の 1 方向の解の C ABI。
//   研究版 (tools/BeamSelStudy/native_bench/dressed_fused.inc、v19) の式をそのまま写し、大域の設定と thread_local をやめた
//   (設定は呼ぶたびに構造体で、作業領域は明示の手綱で渡す。codex2 第 31 回)。設計は .project-guidance/ReciPro/ReciPro_ビーム選択_新方式_移植計画_261110.md。
//   行列はすべて列優先・複素倍精度 (C# の System.Numerics.Complex[] と同じ並び = 実部と虚部の double の組)。
#pragma once
#include "EigenFuncs.h" // EIGEN_FUNCS_API

extern "C" {
	// 戻り値 (DressedTailSolve・Create の status)
	enum DressedTailStatus
	{
		DT_OK = 0,
		DT_FAIL_STAGE1_EVD = 1,   // 段 1 (H_b) の固有分解の失敗
		DT_FAIL_FINAL_EVD = 2,    // 最後 (G⁻¹K) の固有分解の失敗
		DT_FAIL_GRAM = 3,         // G = V†V + Y†Y が正定値でない
		DT_FAIL_CAPACITY = 4,     // 出力の容量 (capacity) が明示の本数 nS より小さい
		DT_FAIL_ARGS = 5,         // 引数の不正 (null・負の大きさ・ABI の版の違い)
		DT_FAIL_OOM = 6,          // メモリの確保の失敗
		DT_FAIL_NONFINITE = 7,    // 結果に非有限値
		DT_FAIL_UNKNOWN = 9       // その他の例外
	};

	// 設定 (ABI の版 1)。既定値は C# 側 (BetheMethod.DressedTailSettings) が持つ
	struct DressedTailSettings
	{
		int AbiVersion;           // = 1
		int AllowUBf16;           // 1 = CPU が対応すれば疎な積の U も bf16 (VDPBF16PS)。0 = U は単精度
		double ModeTau;           // モードの打ち切り τ (入口の励起の組の重み ‖V_C a_C‖ が最大のこの割合未満の組を捨てる。0 = 打ち切らない)
		double DegTol;            // 近縮退の組の幅 |γ_i − γ_j|
		double EdeltaThreshold;   // E_δ の恒等式で残す要素の倍率 δ²/(|λ − γ|² + δ²) の閾値
		double RegDepthNm;        // 正則化の深さ (δ = 1/(π·RegDepthNm))
		double Reserved[4];
	};

	EIGEN_FUNCS_API int _DressedTailAbiVersion();
	// bit 0 = AVX-512 の経路で作った版、bit 1 = U の bf16 (AVX512_BF16) が使える、bit 2 = AVX2 の版
	EIGEN_FUNCS_API int _DressedTailCapabilities();

	// エネルギーごとの共有の表 (中で単精度に写して持つ。呼ぶ側の配列は呼び出しの後に返してよい)。
	//   diffU・diffF: プールの差の箱 (半幅 DH, DK, DL) の U と F (複素倍精度)。番号 = ((h + DH)(2DK + 1) + k + DK)(2DL + 1) + l + DL
	//   dHkl・dRe・dIm (nD): 裾どうしの結合 (|U(d)| ≥ η max|U|) の差の一覧と U(d) (単精度)
	//   H・K・L: 裾の行の 3 次元の表の半幅 (プールの最大 |h|・|k|・|l|)
	//   phase (nPhaseCols × nSites、列 = 候補の列、複素倍精度): 源の求積の位相、wsite (nSites): 点の重み
	EIGEN_FUNCS_API void* _DressedTailCreateTables(const double* diffU, const double* diffF, int diffLength, int DH, int DK, int DL,
		const int* dHkl, const float* dRe, const float* dIm, int nD, int H, int K, int L,
		const double* phase, int nPhaseCols, int nSites, const double* wsite, int* status);
	EIGEN_FUNCS_API void _DressedTailFreeTables(void* tables);
	EIGEN_FUNCS_API long long _DressedTailTablesBytes(const void* tables);

	// 作業領域 (1 つの作業が専有する。中の配列は使い回し、Free で返す)
	EIGEN_FUNCS_API void* _DressedTailCreateWorkspace(int* status);
	EIGEN_FUNCS_API void _DressedTailFreeWorkspace(void* workspace);
	EIGEN_FUNCS_API long long _DressedTailWorkspaceBytes(const void* workspace);

	// 1 方向を解く。sHkl・qHkl (3 × nS・3 × nQ)、Ds・Dq (流束座標の D)、hss (nS × nS、対角 = λ_S)、lamQ (nQ)、qPhaseCol (nQ、位相の表の列、−1 = 0)、
	//   sqrtP0 = √P(000)。出力: gammaOut (capacity 以上)、sourceOut (capacity² 以上、nM × nM を列優先で)、nM。seconds (長さ 9 以上か null) に段ごとの時間を足す
	//   (2 疎な積、4 段 2、5 ブロックの構築、6 H_SQ X₀、7 固有分解と打ち切り、8 X₀V_k)。失敗なら nM = 0
	EIGEN_FUNCS_API int _DressedTailSolve(const void* tables, void* workspace, const DressedTailSettings* settings,
		int nS, int nQ, const int* sHkl, const int* qHkl, const double* Ds, const double* Dq,
		const double* hss, const double* lamQ, const int* qPhaseCol, double sqrtP0,
		double* gammaOut, double* sourceOut, int capacity, int* nM, double* seconds);
} // extern "C"
