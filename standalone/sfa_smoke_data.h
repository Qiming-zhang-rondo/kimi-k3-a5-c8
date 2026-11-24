#pragma once
#include "sfa_fixed_contract.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sfa_smoke {
inline uint16_t ToBf16(float value) {
    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U) return static_cast<uint16_t>((bits >> 16) | 0x40U);
    return static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
inline float FromBf16(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16; float result;
    std::memcpy(&result, &bits, sizeof(result)); return result;
}
struct Data {
    std::vector<uint16_t> query, key, queryRope, keyRope;
    std::vector<int32_t> indices;
    bool zeroQuery = false;
};
struct ReferenceResult { std::vector<float> output, maximum, sum; };
struct CheckResult {
    bool passed = false;
    double relativeL2 = 0, worstRowRelativeL2 = 0, maxAbs = 0;
    double maxRelativeError = 0, sumRelativeError = 0;
    size_t nonfinite = 0;
    std::string message;
};
inline Data MakeData(bool zeroQuery, int variant) {
    using namespace sfa_fixed;
    Data data; data.zeroQuery = zeroQuery;
    data.query.resize(queries * heads * latentDim);
    data.queryRope.resize(queries * heads * ropeDim);
    data.key.resize(physicalKv * kvHeads * latentDim);
    data.keyRope.resize(physicalKv * kvHeads * ropeDim);
    data.indices.resize(queries * sparseCompact);
    for (size_t i = 0; i < data.query.size(); ++i)
        data.query[i] = ToBf16(zeroQuery ? 0.0F : 0.015F * float((i / latentDim) % 9 + 1) +
                               float(int((i * 7) % 19) - 9) * 0.01F);
    for (size_t i = 0; i < data.queryRope.size(); ++i)
        data.queryRope[i] = ToBf16(zeroQuery ? 0.0F : float(int((i * 3) % 11) - 5) * 0.02F);
    for (size_t token = 0; token < physicalKv; ++token) {
        for (size_t d = 0; d < latentDim; ++d)
            data.key[token * latentDim + d] = ToBf16(0.2F + variant * 0.05F + 0.12F * float(int(token / 64) - 1) +
                float(int((token * 3 + d * 7 + size_t(variant >= 0 ? variant : 0)) % 23) - 11) * 0.02F);
        for (size_t d = 0; d < ropeDim; ++d)
            data.keyRope[token * ropeDim + d] = ToBf16(float(int((token * 5 + d * 3) % 17) - 8) * 0.03F);
    }
    for (size_t q = 0; q < queries; ++q)
        for (size_t j = 0; j < sparseCompact; ++j)
            data.indices[q * sparseCompact + j] = int32_t((q * 41 + j * 3) % physicalKv);
    return data;
}
inline ReferenceResult Reference(const Data &data) {
    using namespace sfa_fixed;
    const size_t rows = queries * heads;
    if (data.query.size() != rows * latentDim || data.queryRope.size() != rows * ropeDim ||
        data.key.size() != physicalKv * latentDim || data.keyRope.size() != physicalKv * ropeDim ||
        data.indices.size() != queries * sparseCompact) throw std::invalid_argument("Invalid fixed data shape");
    std::vector<float> q, k, qr, kr;
    for (auto x : data.query) q.push_back(FromBf16(x));
    for (auto x : data.key) k.push_back(FromBf16(x));
    for (auto x : data.queryRope) qr.push_back(FromBf16(x));
    for (auto x : data.keyRope) kr.push_back(FromBf16(x));
    ReferenceResult ref;
    ref.output.resize(rows * latentDim); ref.maximum.resize(rows); ref.sum.resize(rows);
    for (size_t row = 0; row < rows; ++row) {
        std::vector<double> scores(sparseCompact);
        double maximum = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < sparseCompact; ++j) {
            int32_t token = data.indices[(row / heads) * sparseCompact + j];
            if (token < 0 || size_t(token) >= physicalKv) throw std::invalid_argument("Invalid sparse index");
            double score = 0;
            for (size_t d = 0; d < latentDim; ++d) score += double(q[row * latentDim + d]) * k[size_t(token) * latentDim + d];
            for (size_t d = 0; d < ropeDim; ++d) score += double(qr[row * ropeDim + d]) * kr[size_t(token) * ropeDim + d];
            scores[j] = score * softmaxScale; maximum = std::max(maximum, scores[j]);
        }
        double sum = 0;
        for (double &score : scores) { score = std::exp(score - maximum); sum += score; }
        ref.maximum[row] = float(maximum); ref.sum[row] = float(sum);
        for (size_t d = 0; d < latentDim; ++d) {
            double output = 0;
            for (size_t j = 0; j < sparseCompact; ++j)
                output += scores[j] * k[size_t(data.indices[(row / heads) * sparseCompact + j]) * latentDim + d];
            ref.output[row * latentDim + d] = float(output / sum);
        }
    }
    return ref;
}
inline CheckResult Check(const Data &data, const std::vector<uint16_t> &actualOut,
                         const std::vector<float> &maximum, const std::vector<float> &sum) {
    using namespace sfa_fixed;
    CheckResult result; const size_t rows = queries * heads;
    if (actualOut.size() != rows * latentDim || maximum.size() != rows || sum.size() != rows) {
        result.message = "Output/statistics shape mismatch"; return result;
    }
    ReferenceResult ref;
    try { ref = Reference(data); } catch (const std::exception &error) { result.message = error.what(); return result; }
    double totalError = 0, totalNorm = 0; bool statsGood = true;
    for (size_t row = 0; row < rows; ++row) {
        double error = 0, norm = 0;
        for (size_t d = 0; d < latentDim; ++d) {
            size_t i = row * latentDim + d; double actual = FromBf16(actualOut[i]), expected = ref.output[i];
            if (!std::isfinite(actual) || !std::isfinite(expected)) { ++result.nonfinite; continue; }
            error += (actual - expected) * (actual - expected); norm += expected * expected;
            result.maxAbs = std::max(result.maxAbs, std::abs(actual - expected));
        }
        totalError += error; totalNorm += norm;
        result.worstRowRelativeL2 = std::max(result.worstRowRelativeL2, std::sqrt(error / std::max(norm, 1e-30)));
        if (!std::isfinite(maximum[row]) || !std::isfinite(sum[row]) || !std::isfinite(ref.maximum[row]) || !std::isfinite(ref.sum[row])) {
            ++result.nonfinite; statsGood = false; continue;
        }
        double maxError = std::abs(double(maximum[row]) - ref.maximum[row]) / std::max(1.0, std::abs(double(ref.maximum[row])));
        double sumError = std::abs(double(sum[row]) - ref.sum[row]) / std::max(1e-30, double(ref.sum[row]));
        result.maxRelativeError = std::max(result.maxRelativeError, maxError);
        result.sumRelativeError = std::max(result.sumRelativeError, sumError);
        statsGood = statsGood && maxError <= 0.01 && sumError <= 0.01 && sum[row] > 0;
        if (data.zeroQuery) statsGood = statsGood && std::abs(maximum[row]) <= 1e-4 && std::abs(sum[row] - float(sparseCompact)) <= 0.02;
    }
    result.relativeL2 = std::sqrt(totalError / std::max(totalNorm, 1e-30));
    result.passed = result.nonfinite == 0 && result.relativeL2 <= 0.01 && result.worstRowRelativeL2 <= 0.01 && statsGood;
    result.message = result.passed ? "All rows, output, and raw max/sum passed CPU reference checks" : "Output/statistics accuracy gate failed";
    return result;
}
} // namespace sfa_smoke
