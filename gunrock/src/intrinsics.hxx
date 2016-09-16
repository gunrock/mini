#pragma once

namespace gunrock {
namespace util {

    __device__ __forceinline__ int LaneId() {
        int ret;
        asm("mov.u32 %0, %laneid;" : "=r"(ret) );
        return ret;
    }

}
}
