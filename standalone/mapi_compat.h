// microapi_compat.h — Map AscendC::MicroAPI to public CANN 9.2.1 SDK API
// The custom_cot 9.1.0 kernel code references AscendC::MicroAPI which is
// a private/extended namespace not present in CANN 9.2.1. The public SDK
// provides the same types directly in namespace AscendC.
// This shim provides namespace-level aliases so the kernel code compiles.
#pragma once

#include "kernel_operator.h"

namespace AscendC {

// Public SDK types (from basic_api/reg_compute/kernel_reg_compute_utils.h)
// #include "basic_api/reg_compute/kernel_reg_compute_utils.h"

namespace MicroAPI {

// Type aliases — same types, different namespace qualification
template<typename... Args>
using CastTrait = AscendC::CastTrait;

// If CastTrait is not a template, use direct alias
// using CastTrait = AscendC::CastTrait;

namespace RegLayout {
    using THREE = AscendC::RegLayout::THREE;
    using ONE = AscendC::RegLayout::ONE;
    using ZERO = AscendC::RegLayout::ZERO;
    using UNKNOWN = AscendC::RegLayout::UNKNOWN;
}

namespace SatMode {
    using SAT = AscendC::SatMode::SAT;
    using NO_SAT = AscendC::SatMode::NO_SAT;
    using UNKNOWN = AscendC::SatMode::UNKNOWN;
}

namespace MaskMergeMode {
    using ZEROING = AscendC::MaskMergeMode::ZEROING;
    using NOMIC = AscendC::MaskMergeMode::NOMIC;
    using UNKNOWN = AscendC::MaskMergeMode::UNKNOWN;
}

namespace RoundMode {
    using CAST_RINT = AscendC::RoundMode::CAST_RINT;
    using CAST_RZ = AscendC::RoundMode::CAST_RZ;
    using CAST_RN = AscendC::RoundMode::CAST_RN;
}

} // namespace MicroAPI
} // namespace AscendC
