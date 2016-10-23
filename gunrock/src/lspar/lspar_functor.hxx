#pragma once
#include "lspar/lspar_problem.hxx"
#include "intrinsics.hxx"

using namespace gunrock::util;

namespace gunrock {
namespace lspar {

struct lspar_functor_t {
static __device__ __forceinline__ bool cond_filter(int idx, lspar_problem_t::data_slice_t *data, int iteration) {
}

static __device__ __forceinline__ bool cond_advance(int src, int dst, int edge_id, int output_idx, lspar_problem_t::data_slice_t *data, int iteration) {
}

static __device__ __forceinline__ bool apply_advance(int src, int dst, int edge_id, int output_idx, lspar_problem_t::data_slice_t *data, int iteration) {
}

static __device__ __forceinline__ int get_value_to_reduce(int idx, lspar_problem_t::data_slice_t *data, int iteration) {
}

};

}// end of lspar
}// end of gunrock



