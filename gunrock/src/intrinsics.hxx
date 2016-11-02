#pragma once

namespace gunrock {
namespace util {

    __device__ __forceinline__ int LaneId() {
        int ret;
        asm("mov.u32 %0, %laneid;" : "=r"(ret) );
        return ret;
    }

    __device__ __forceinline__ float atomicMin(float* addr, float val)
    {
        int* addr_as_int = (int*)addr;
        int old = *addr_as_int;
        int expected;
        do {
            expected = old;
            old = ::atomicCAS(addr_as_int, expected, __float_as_int(::fminf(val, __int_as_float(expected))));
        } while (expected != old);
        return __int_as_float(old);
    }
}
}
