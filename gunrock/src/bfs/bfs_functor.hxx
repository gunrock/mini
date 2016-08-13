#pragma once
#include "bfs/bfs_problem.hxx"

namespace gunrock {
namespace bfs {

struct bfs_functor_t {

static __device__ __forceinline__ bool cond_filter(int idx, bfs_problem_t::data_slice_t *data, int iteration) {
    return idx != -1;
}

static __device__ __forceinline__ void apply_filter(int idx, bfs_problem_t::data_slice_t *data, int iteration) {
    //data->d_labels[idx] = iteration + 1;
}

static __device__ __forceinline__ bool cond_advance(int src, int dst, bfs_problem_t::data_slice_t *data, int iteration) {
    return atomicCAS(&data->d_labels[dst], -1, iteration + 1) == -1;
}

};

}// end of bfs
}// end of gunrock



