// Public inline sidecar writer. No private Prolog source, host ABI or launch.
// Include AND instantiate only in the actual mixed Prolog's VEC3510 branch;
// never change the enclosing Prolog task type to AIV_ONLY for this helper.
#pragma once
#include "kernel_operator.h"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510 && defined(__DAV_C310_VEC__)
namespace QsfaPrologRingWrite {
using namespace AscendC;

// Device-internal binding only. The private adapter owns its real host ABI and
// tiling and must validate buffer sizes/capacity before launching the producer.
struct Binding {
    GM_ADDR latentRing = nullptr;       // BF16 [pools,capacity,512]
    GM_ADDR ropeRing = nullptr;         // BF16 [pools,capacity,64]
    GM_ADDR positionTags = nullptr;     // INT64 [pools,capacity]
    GM_ADDR epochTags = nullptr;        // INT64 [pools,capacity]
    GM_ADDR tokenBanks = nullptr;       // INT32 [rows], stable request bank IDs
    GM_ADDR tokenPositions = nullptr;   // INT64 [rows], actual logical positions
    GM_ADDR tokenEpochs = nullptr;      // INT64 [rows], request generations
    GM_ADDR tokenValid = nullptr;       // UINT8/BOOL [rows], excludes graph padding
    uint32_t rows = 0;
    uint32_t pools = 0;
    uint32_t capacity = 0;
    uint32_t enabled = 0;
};

struct Route {
    bool active = false;
    uint64_t ringRow = 0; // bank*capacity + logical_position%capacity
    int64_t position = -1;
    int64_t epoch = -1;
};

template <HardEvent Event>
__aicore__ inline void Fence(event_t id)
{
    SetFlag<Event>(id);
    WaitFlag<Event>(id);
}

class Writer {
public:
    // Call ONCE per helper, using the SAME TPipe as the enclosing Prolog.
    // OFF touches no UB/event allocator. False means invalid initialization;
    // the caller must reject that configuration, never pretend it wrote a ring.
    __aicore__ inline bool Init(TPipe &pipe, const Binding &binding)
    {
        if (configured_) return false;
        configured_ = true;
        if (binding.enabled == 0) return true;
        if (binding.enabled != 1 || binding.rows == 0 || binding.pools == 0 ||
            binding.capacity == 0 || binding.capacity % 128 != 0 ||
            !binding.latentRing || !binding.ropeRing || !binding.positionTags ||
            !binding.epochTags || !binding.tokenBanks || !binding.tokenPositions ||
            !binding.tokenEpochs || !binding.tokenValid) return false;
        if (static_cast<uint64_t>(binding.pools) * binding.capacity >
            (~static_cast<uint64_t>(0)) / (512 * sizeof(bfloat16_t))) return false;
        pipe_ = &pipe;
        rows_ = binding.rows; pools_ = binding.pools; capacity_ = binding.capacity;
        latentGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(binding.latentRing));
        ropeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(binding.ropeRing));
        positionTagsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(binding.positionTags));
        epochTagsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(binding.epochTags));
        banksGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(binding.tokenBanks));
        positionsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(binding.tokenPositions));
        epochsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(binding.tokenEpochs));
        validGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(binding.tokenValid));
        // Each TBuf begins at a 32-byte-aligned UB allocation. Cast output never
        // aliases the private producer's FP32 quantizer input or shared scratch.
        pipe.InitBuffer(castBuf_, 512 * sizeof(bfloat16_t));
        pipe.InitBuffer(bankBuf_, 32); pipe.InitBuffer(positionBuf_, 32);
        pipe.InitBuffer(epochBuf_, 32); pipe.InitBuffer(validBuf_, 32);
        pipe.InitBuffer(tagsBuf_, 64);
        // Reserve IDs with the owning pipeline; never steal EVENT_ID0 from a
        // mixed Prolog. Caller must reserve any hard-coded private IDs too.
        mte2s_ = pipe.AllocEventID<HardEvent::MTE2_S>();
        smte2_ = pipe.AllocEventID<HardEvent::S_MTE2>();
        vmte3_ = pipe.AllocEventID<HardEvent::V_MTE3>();
        mte3v_ = pipe.AllocEventID<HardEvent::MTE3_V>();
        smte3_ = pipe.AllocEventID<HardEvent::S_MTE3>();
        mte3s_ = pipe.AllocEventID<HardEvent::MTE3_S>();
        enabled_ = true;
        return true;
    }

    __aicore__ inline Route ResolveRow(uint32_t token)
    {
        Route route;
        if (!enabled_ || token >= rows_) return route;
        auto bankUb = bankBuf_.Get<int32_t>();
        auto posUb = positionBuf_.Get<int64_t>();
        auto epochUb = epochBuf_.Get<int64_t>();
        auto validUb = validBuf_.Get<uint8_t>();
        const DataCopyExtParams copy4{1,4,0,0,0}, copy8{1,8,0,0,0}, copy1{1,1,0,0,0};
        DataCopyPad(bankUb, banksGm_[token], copy4, DataCopyPadExtParams<int32_t>{false,0,0,0});
        DataCopyPad(posUb, positionsGm_[token], copy8, DataCopyPadExtParams<int64_t>{false,0,0,0});
        DataCopyPad(epochUb, epochsGm_[token], copy8, DataCopyPadExtParams<int64_t>{false,0,0,0});
        DataCopyPad(validUb, validGm_[token], copy1, DataCopyPadExtParams<uint8_t>{false,0,0,0});
        Fence<HardEvent::MTE2_S>(mte2s_);
        const int32_t bank = bankUb.GetValue(0);
        const int64_t position = posUb.GetValue(0), epoch = epochUb.GetValue(0);
        const bool valid = validUb.GetValue(0) != 0;
        Fence<HardEvent::S_MTE2>(smte2_); // next metadata DMA cannot race scalar reads
        if (!valid || bank < 0 || static_cast<uint32_t>(bank) >= pools_ || position < 0 || epoch < 0) return route;
        route.active = true;
        route.ringRow = static_cast<uint64_t>(bank) * capacity_ + static_cast<uint64_t>(position) % capacity_;
        route.position = position; route.epoch = epoch;
        return route;
    }

    // Source starts at local element ZERO. colOffset is ONLY the destination
    // latent column. Caller slices its real physical UB layout explicitly and
    // ensures source readiness. Do not pass a synthetic reloaded GM value.
    __aicore__ inline bool WriteLatentSlice(const Route &route,
        const LocalTensor<float> &exactQuantInput, uint32_t colOffset, uint32_t count)
    {
        if (!enabled_ || !route.active) return true;
        if (count == 0 || colOffset >= 512 || count > 512 - colOffset) return false;
        auto castUb = castBuf_.Get<bfloat16_t>();
        Cast(castUb, exactQuantInput, RoundMode::CAST_RINT, count); // COUNT overload
        Fence<HardEvent::V_MTE3>(vmte3_);
        const DataCopyExtParams bytes{1,static_cast<uint32_t>(count * sizeof(bfloat16_t)),0,0,0};
        DataCopyPad(latentGm_[route.ringRow * 512 + colOffset], castUb, bytes);
        Fence<HardEvent::MTE3_V>(mte3v_); // cast scratch reusable only after this copy
        return true;
    }

    // Source is the ACTUAL post-RoPE BF16 slice, at local element ZERO.
    // Return drains its MTE3 reader before the caller may reuse local storage.
    __aicore__ inline bool WriteRopeSlice(const Route &route,
        const LocalTensor<bfloat16_t> &actualPostRope, uint32_t colOffset, uint32_t count)
    {
        if (!enabled_ || !route.active) return true;
        if (count == 0 || colOffset >= 64 || count > 64 - colOffset) return false;
        Fence<HardEvent::V_MTE3>(vmte3_);
        const DataCopyExtParams bytes{1,static_cast<uint32_t>(count * sizeof(bfloat16_t)),0,0,0};
        DataCopyPad(ropeGm_[route.ringRow * 64 + colOffset], actualPostRope, bytes);
        Fence<HardEvent::MTE3_V>(mte3v_);
        return true;
    }

    // Exactly ONE producer owner publishes each token's tags. Payload slices
    // can be written by multiple cores, but consumers MUST wait for completion
    // of the WHOLE producer kernel on the same stream (or an explicit event).
    // No tag-as-live-lock protocol or cross-core spin barrier is implemented.
    __aicore__ inline void PublishTags(const Route &route, bool uniqueTagOwner)
    {
        if (!enabled_ || !route.active || !uniqueTagOwner) return;
        auto tagsUb = tagsBuf_.Get<int64_t>();
        tagsUb.SetValue(0, route.position); tagsUb.SetValue(4, route.epoch);
        Fence<HardEvent::S_MTE3>(smte3_);
        const DataCopyExtParams oneTag{1,8,0,0,0};
        DataCopyPad(positionTagsGm_[route.ringRow], tagsUb, oneTag);
        DataCopyPad(epochTagsGm_[route.ringRow], tagsUb[4], oneTag);
        Fence<HardEvent::MTE3_S>(mte3s_); // next SetValue cannot overwrite in-flight tag data
    }

    // All helper copies are drained by the methods above. Call at the private
    // producer's vector cleanup point if its TPipe is reused afterward.
    __aicore__ inline void Release()
    {
        if (!enabled_) return;
        pipe_->ReleaseEventID<HardEvent::MTE2_S>(mte2s_);
        pipe_->ReleaseEventID<HardEvent::S_MTE2>(smte2_);
        pipe_->ReleaseEventID<HardEvent::V_MTE3>(vmte3_);
        pipe_->ReleaseEventID<HardEvent::MTE3_V>(mte3v_);
        pipe_->ReleaseEventID<HardEvent::S_MTE3>(smte3_);
        pipe_->ReleaseEventID<HardEvent::MTE3_S>(mte3s_);
        enabled_ = false;
    }

private:
    bool configured_ = false, enabled_ = false;
    TPipe *pipe_ = nullptr;
    uint32_t rows_ = 0, pools_ = 0, capacity_ = 0;
    event_t mte2s_{}, smte2_{}, vmte3_{}, mte3v_{}, smte3_{}, mte3s_{};
    GlobalTensor<bfloat16_t> latentGm_, ropeGm_;
    GlobalTensor<int64_t> positionTagsGm_, epochTagsGm_, positionsGm_, epochsGm_;
    GlobalTensor<int32_t> banksGm_;
    GlobalTensor<uint8_t> validGm_;
    TBuf<TPosition::VECCALC> castBuf_, bankBuf_, positionBuf_, epochBuf_, validBuf_, tagsBuf_;
};
} // namespace QsfaPrologRingWrite
#endif // actual compiler-selected VEC3510 only
