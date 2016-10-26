#pragma once
#include "lspar/lspar_problem.hxx"
#include "intrinsics.hxx"

using namespace gunrock::util;

namespace gunrock {
namespace lspar {

struct minhash_functor_t {

static __device__ __forceinline__ bool cond_advance(int src, int dst, int edge_id, int rank, int output_idx, lspar_problem_t::data_slice_t *data, int iteration) {
    return true;
}

static __device__ __forceinline__ bool apply_advance(int src, int dst, int edge_id, int rank, int output_idx, lspar_problem_t::data_slice_t *data, int iteration) {
    return true;
}

static __device__ __forceinline__ int get_value_to_reduce(int idx, lspar_problem_t::data_slice_t *data, int iteration) {
    return data->d_hashs[idx];
}

};

struct sim_functor_t {

static __device__ __forceinline__ bool cond_advance(int src, int dst, int edge_id, int rank, int output_idx, lspar_problem_t::data_slice_t *data, int iteration) {
    bool flag = (data->d_minwise_hashs[src] == data->d_minwise_hashs[src]);
    data->d_sims[output_idx].sim = flag ? 1 : 0;
    data->d_sims[output_idx].eid = output_idx;
    return flag;
}

static __device__ __forceinline__ bool apply_advance(int src, int dst, int edge_id, int rank, int output_idx, lspar_problem_t::data_slice_t *data, int iteration) {
    return true;
}

};

struct select_functor_t {

static __device__ __forceinline__ bool cond_advance(int src, int dst, int edge_id, int rank, int output_idx, lspar_problem_t::data_slice_t *data, int iteration) {
    int thres = data->d_thresholds[src];
    bool flag = (rank <= thres);
    return flag;
}

static __device__ __forceinline__ bool apply_advance(int src, int dst, int edge_id, int rank, int output_idx, lspar_problem_t::data_slice_t *data, int iteration) {
    return true;
}

};

}// end of lspar
}// end of gunrock



