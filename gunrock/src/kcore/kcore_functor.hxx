#pragma once
#include "kcore/kcore_problem.hxx"
#include "intrinsics.hxx"

using namespace gunrock::util;

namespace gunrock {
namespace kcore {

    struct deg_less_than_k_functor_t {
        static __device__ __forceinline__ bool cond_filter(int idx, kcore_problem_t::data_slice_t *data, int k) { 
          if (data->d_degrees[idx] < k) {
            data->d_num_cores[idx] = k - 1;
            data->d_degrees[idx] = 0;
            return true;
          }
          else return false;
        }
    };

    struct deg_atleast_k_functor_t {
        static __device__ __forceinline__ bool cond_filter(int idx, kcore_problem_t::data_slice_t *data, int k) {
          return (data->d_degrees[idx] >= k);            
        }
    };

    struct update_deg_functor_t {
        static __device__ __forceinline__ bool cond_advance(int idx, kcore_problem_t::data_slice_t *data, int k) {
          return true;
        }

        static __device__ __forceinline__ bool apply_advance(int src, int dst, int edge_id, int rank, int output_idx, coloring_problem_t::data_slice_t *data, int iteration) {
            atomicAdd(data->d_degrees + dst, -1);
            return (data->d_degrees[dst] > 0);
        }
    };

}// end of kcore
}// end of gunrock