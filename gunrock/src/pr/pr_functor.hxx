#pragma once
#include "pr/pr_problem.hxx"
#include "intrinsics.hxx"

using namespace gunrock::util;

namespace gunrock {
namespace pr {

    struct pr_functor_t {
        static __device__ __forceinline__ bool cond_filter(int idx, pr_problem_t::data_slice_t *data, int iteration) {
            float old_value = data->d_current_ranks[idx];
            float new_value = (data->d_degrees[idx] > 0) ? (0.15f + 0.85f * data->d_reduced_ranks[idx] / data->d_degrees[idx]) : 0.15f;
            if (!isfinite(new_value)) new_value = 0;
            data->d_current_ranks[idx] = new_value;
            return (fabs(new_value-old_value) > (0.001f*old_value));
        }

        static __device__ __forceinline__ bool cond_advance(int src, int dst, int edge_id, int rank, int output_idx, pr_problem_t::data_slice_t *data, int iteration) {
            return true;
        }

        static __device__ __forceinline__ bool apply_advance(int src, int dst, int edge_id, int rank, int output_idx, pr_problem_t::data_slice_t *data, int iteration) {
            return true;
        }

        static __device__ __forceinline__ float get_value_to_reduce(int idx, pr_problem_t::data_slice_t *data, int iteration) {
                return (isfinite(data->d_current_ranks[idx]) ? data->d_current_ranks[idx]:0);
        }
    };

}// end of pr
}// end of gunrock



