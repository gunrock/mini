#pragma once
#include "coloring/coloring_problem.hxx"
#include "intrinsics.hxx"

using namespace gunrock::util;

namespace gunrock {
namespace coloring {

    struct coloring_functor_t {
        static __device__ __forceinline__ bool cond_filter(int idx, coloring_problem_t::data_slice_t *data, int iteration) { 
            int color = 0;
            int hash = data->d_hashs[idx];
            int max_hash = data->d_reduced_max[idx];
            int min_hash = data->d_reduced_min[idx];
            if (hash < min_hash) {
                color = iteration*2+1;
            }
            if (hash > max_hash) {
                color = iteration*2+2;
            }
            if (color > 0) {
                data->d_colors[idx] = color;
                return false;
            } else {
                return true;
            }
        }

        static __device__ __forceinline__ bool cond_advance(int src, int dst, int edge_id, int output_idx, coloring_problem_t::data_slice_t *data, int iteration) {
            return true;
        }

        static __device__ __forceinline__ bool apply_advance(int src, int dst, int edge_id, int output_idx, coloring_problem_t::data_slice_t *data, int iteration) {
            return true;
        }

        static __device__ __forceinline__ int get_value_to_reduce(int idx, coloring_problem_t::data_slice_t *data, int iteration) {
            return data->d_hashs[idx];
        }

    };

}// end of coloring
}// end of gunrock



