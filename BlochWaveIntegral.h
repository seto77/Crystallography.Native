#pragma once
#include <complex>

// 261106Ch 追加: ∫₀ᵗ exp(λz) dz。managed BlochWaveIntegral と同じ |λt| 判定・6次 exprel 級数。
// exponential は既存の等間隔深さの累積指数を再利用する。λ=0 の極限も級数で評価する。
inline std::complex<double> IntegrateBlochExponential(
    std::complex<double> lambda, double thickness, std::complex<double> exponential)
{
    const auto z = lambda * thickness;
    if (std::norm(z) <= 1e-4)
        return thickness * (1.0 + z * (0.5 + z * (1.0 / 6 + z * (1.0 / 24
            + z * (1.0 / 120 + z * (1.0 / 720 + z / 5040.0))))));
    return (exponential - 1.0) / lambda;
}
