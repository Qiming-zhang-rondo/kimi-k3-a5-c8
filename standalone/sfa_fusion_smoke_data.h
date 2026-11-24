#pragma once
#include "sfa_smoke_data.h"
#include <utility>

// Host-only isolated epilogue data/reference. Cold states here are supplied
// normalized FP32 states, NOT outputs from the real C4 reader. Passing these
// checks establishes no end-to-end C4 accuracy or performance result.
// Even hotPresent==0 keeps a fully valid compact128 hot dispatch: this tests
// epilogue exclusion, not an empty-hot native scheduling path.
namespace sfa_fusion_smoke {
inline constexpr std::size_t rows = sfa_fixed::queries * sfa_fixed::heads;
inline constexpr std::size_t outputElements = rows * sfa_fixed::latentDim;
inline constexpr std::size_t outputBytes = outputElements * sizeof(float);
static_assert(outputBytes == 524288);

struct Data {
    sfa_smoke::Data hot;
    std::vector<float> coldOutput, coldMax, coldSum;
    std::vector<uint8_t> coldPresent, hotPresent;
    std::string name;
};

inline sfa_fixed::Tiling MakeTiling(float scale = sfa_fixed::softmaxScale)
{
    auto tiling = sfa_fixed::MakeTiling(scale);
    tiling.baseParams.returnSoftmaxLse = 0; // No combined stats ABI in this candidate.
    return tiling;
}

inline bool ValidateTiling(const sfa_fixed::Tiling& tiling)
{
    if (tiling.baseParams.returnSoftmaxLse != 0) return false;
    auto baseline = tiling;
    baseline.baseParams.returnSoftmaxLse = 1;
    return sfa_fixed::ValidateTiling(baseline);
}

inline bool ValidateData(const Data& data)
{
    using namespace sfa_fixed;
    const auto& hot = data.hot;
    if (hot.query.size() != rows * latentDim || hot.key.size() != physicalKv * latentDim ||
        hot.queryRope.size() != rows * ropeDim || hot.keyRope.size() != physicalKv * ropeDim ||
        hot.indices.size() != queries * sparseCompact || data.coldOutput.size() != outputElements ||
        data.coldMax.size() != rows || data.coldSum.size() != rows ||
        data.coldPresent.size() != queries || data.hotPresent.size() != queries) return false;
    const int32_t table[2] = {0, 1}, qlen[1] = {4}, kvlen[1] = {256};
    if (!ValidateIndicesAndLengths(hot.indices.data(), hot.indices.size(), table, 2, qlen, 1, kvlen, 1))
        return false;
    for (std::size_t q = 0; q < queries; ++q)
        if (data.coldPresent[q] > 1 || data.hotPresent[q] > 1) return false;
    return true;
}

inline Data MakeData(const std::string& name, bool zeroQuery, int hotVariant = 0)
{
    Data data;
    data.name = name;
    data.hot = sfa_smoke::MakeData(zeroQuery, hotVariant);
    data.coldOutput.resize(outputElements);
    data.coldMax.resize(rows);
    data.coldSum.resize(rows);
    data.coldPresent.assign(sfa_fixed::queries, 1);
    data.hotPresent.assign(sfa_fixed::queries, 1);
    for (std::size_t row = 0; row < rows; ++row) {
        data.coldMax[row] = zeroQuery ? 0.0F : 0.13F * float(int(row % 9) - 4);
        data.coldSum[row] = zeroQuery ? 1920.0F : 1700.0F + float(row % 127);
        for (std::size_t d = 0; d < sfa_fixed::latentDim; ++d)
            data.coldOutput[row * sfa_fixed::latentDim + d] =
                0.07F + 0.04F * float(row / sfa_fixed::heads) + 0.003F * float(row % sfa_fixed::heads) +
                0.011F * float(int((d * 7 + row * 3) % 23) - 11);
    }
    return data;
}

inline void PoisonAbsentCold(Data& data)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (std::size_t row = 0; row < rows; ++row) {
        if (data.coldPresent[row / sfa_fixed::heads]) continue;
        data.coldMax[row] = nan;
        data.coldSum[row] = nan;
        std::fill_n(data.coldOutput.begin() + row * sfa_fixed::latentDim, sfa_fixed::latentDim, nan);
    }
}

inline std::vector<Data> MakeCases()
{
    std::vector<Data> cases;
    cases.push_back(MakeData("mixed_zero", true));
    cases.push_back(MakeData("mixed_nonzero", false));
    cases.push_back(MakeData("changed_hot", false, 1));
    auto cold = MakeData("cold_only_flag", false);
    std::fill(cold.hotPresent.begin(), cold.hotPresent.end(), 0);
    cases.push_back(std::move(cold));
    auto hot = MakeData("hot_only_flag", false);
    std::fill(hot.coldPresent.begin(), hot.coldPresent.end(), 0);
    PoisonAbsentCold(hot);
    cases.push_back(std::move(hot));
    auto empty = MakeData("both_empty", true);
    std::fill(empty.coldPresent.begin(), empty.coldPresent.end(), 0);
    std::fill(empty.hotPresent.begin(), empty.hotPresent.end(), 0);
    PoisonAbsentCold(empty);
    cases.push_back(std::move(empty));
    auto extreme = MakeData("extreme_cold", false);
    for (std::size_t row = 0; row < rows; ++row) {
        extreme.coldMax[row] = (row % 2 == 0) ? 1.0e30F : -1.0e30F;
        extreme.coldSum[row] = (row % 2 == 0) ? 2048.0F : 1.0F;
    }
    cases.push_back(std::move(extreme));
    auto flags = MakeData("per_query_presence", false);
    flags.coldPresent = {1, 0, 1, 0};
    flags.hotPresent = {1, 1, 0, 0};
    PoisonAbsentCold(flags);
    cases.push_back(std::move(flags));
    return cases;
}

// The independent math uses double weights. Only the hot normalized output
// follows the original BF16 output contract; cold output is never BF16-rounded.
inline std::vector<float> NativeReference(
    const Data& data, const std::vector<uint16_t>& nativeHotOut,
    const std::vector<float>& nativeMax, const std::vector<float>& nativeSum)
{
    if (!ValidateData(data) || nativeHotOut.size() != outputElements ||
        nativeMax.size() != rows || nativeSum.size() != rows)
        throw std::invalid_argument("Invalid fusion data/native-state shape or compact indices");
    std::vector<float> output(outputElements, 0.0F);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (std::size_t row = 0; row < rows; ++row) {
        const bool hp = data.hotPresent[row / sfa_fixed::heads] != 0;
        const bool cp = data.coldPresent[row / sfa_fixed::heads] != 0;
        if (!hp && !cp) continue;
        const bool hotValid = !hp || (std::isfinite(nativeMax[row]) && std::isfinite(nativeSum[row]) && nativeSum[row] > 0);
        const bool coldValid = !cp || (std::isfinite(data.coldMax[row]) && std::isfinite(data.coldSum[row]) && data.coldSum[row] > 0);
        if (!hotValid || !coldValid) {
            std::fill_n(output.begin() + row * sfa_fixed::latentDim, sfa_fixed::latentDim, nan);
            continue;
        }
        double hw = hp ? 1.0 : 0.0, cw = cp ? 1.0 : 0.0;
        if (hp && cp) {
            const double common = std::max(double(nativeMax[row]), double(data.coldMax[row]));
            hw = double(nativeSum[row]) * std::exp(double(nativeMax[row]) - common);
            cw = double(data.coldSum[row]) * std::exp(double(data.coldMax[row]) - common);
            const double denominator = hw + cw;
            hw /= denominator;
            cw /= denominator;
        }
        for (std::size_t d = 0; d < sfa_fixed::latentDim; ++d) {
            const std::size_t i = row * sfa_fixed::latentDim + d;
            // Branching avoids 0*NaN from absent states.
            const double hot = hp ? double(sfa_smoke::FromBf16(nativeHotOut[i])) : 0.0;
            const double cold = cp ? double(data.coldOutput[i]) : 0.0;
            output[i] = float(hw * hot + cw * cold);
        }
    }
    return output;
}

inline std::vector<float> Reference(const Data& data)
{
    if (!ValidateData(data)) throw std::invalid_argument("Invalid fusion data shape or compact indices");
    const auto hot = sfa_smoke::Reference(data.hot);
    std::vector<uint16_t> rounded;
    rounded.reserve(hot.output.size());
    for (const float x : hot.output) rounded.push_back(sfa_smoke::ToBf16(x));
    return NativeReference(data, rounded, hot.maximum, hot.sum);
}

inline sfa_smoke::CheckResult Check(const Data& data, const std::vector<float>& actual,
                                  const std::vector<float>& reference, double tolerance = 0.001)
{
    sfa_smoke::CheckResult result;
    if (!ValidateData(data) || actual.size() != outputElements || reference.size() != outputElements ||
        !std::isfinite(tolerance) || tolerance < 0) {
        result.message = "Fusion output/reference/data shape or tolerance mismatch";
        return result;
    }
    double totalError = 0, totalNorm = 0;
    std::size_t firstBadRow = rows;
    for (std::size_t row = 0; row < rows; ++row) {
        double error = 0, norm = 0;
        for (std::size_t d = 0; d < sfa_fixed::latentDim; ++d) {
            const std::size_t i = row * sfa_fixed::latentDim + d;
            const double a = actual[i], r = reference[i];
            if (!std::isfinite(a) || !std::isfinite(r)) {
                ++result.nonfinite;
                firstBadRow = std::min(firstBadRow, row);
                continue;
            }
            error += (a - r) * (a - r);
            norm += r * r;
            result.maxAbs = std::max(result.maxAbs, std::abs(a - r));
        }
        const double rowL2 = std::sqrt(error / std::max(norm, 1e-30));
        if (rowL2 > tolerance) firstBadRow = std::min(firstBadRow, row);
        result.worstRowRelativeL2 = std::max(result.worstRowRelativeL2, rowL2);
        totalError += error;
        totalNorm += norm;
    }
    result.relativeL2 = std::sqrt(totalError / std::max(totalNorm, 1e-30));
    result.passed = result.nonfinite == 0 && result.relativeL2 <= tolerance &&
        result.worstRowRelativeL2 <= tolerance;
    result.message = result.passed ? "All FP32 fusion rows passed reference checks" :
        "Fusion output gate failed; first_bad_row=" + std::to_string(firstBadRow);
    return result;
}
} // namespace sfa_fusion_smoke
