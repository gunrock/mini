#pragma once
#include "sssp/sssp_problem.hxx"
#include "intrinsics.hxx"

using namespace gunrock::util;

namespace gunrock {
namespace sssp {

struct sssp_functor_t {

static __device__ __forceinline__ bool cond_filter(int idx, sssp_problem_t::data_slice_t *data, int iteration) {
    if (idx == -1) return false;
    if (data->d_visited[idx] == iteration)
        return false;
    data->d_visited[idx] = iteration;
    return true;
}

static __device__ __forceinline__ bool cond_advance(int src, int dst, int edge_id, int rank, int output_idx, sssp_problem_t::data_slice_t *data, int iteration) {
    float new_distance = data->d_labels[src]+data->d_weights[edge_id];
    float old_distance = atomicMin(data->d_labels+dst, new_distance);
    //printf("src:%d, dst:%d old:%5f, new:%5f\n", src, dst, old_distance, new_distance);
    if (new_distance < old_distance) {
        return true;
    }
    else
        return false;
}

static __device__ __forceinline__ bool apply_advance(int src, int dst, int edge_id, int rank, int output_idx, sssp_problem_t::data_slice_t *data, int iteration) {
    data->d_preds[dst] = src;
    return true;
}

};

}// end of sssp
}// end of gunrock



