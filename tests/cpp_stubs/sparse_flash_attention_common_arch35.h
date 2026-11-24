// Host regression stub only. This does not emulate AscendC pipelines, DMA,
// barriers, layouts, compilation constraints, or device execution.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#define __aicore__
#define __gm__

namespace AscendC {

template <typename T>
class LocalTensor {
public:
    LocalTensor(T *values, std::size_t size) : values_(values), size_(size) {}

    T GetValue(std::size_t index) const
    {
        if (index >= size_) {
            throw std::out_of_range("LocalTensor GetValue outside host probe allocation");
        }
        return values_[index];
    }

    void SetValue(std::size_t index, T value)
    {
        if (index >= size_) {
            throw std::out_of_range("LocalTensor SetValue outside host probe allocation");
        }
        values_[index] = value;
    }

private:
    T *values_;
    std::size_t size_;
};

} // namespace AscendC
