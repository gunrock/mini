#pragma once

#include "problem.hxx"

#include "moderngpu/memory.hxx"

using namespace mgpu;

namespace gunrock {
namespace kcore {

struct kcore_problem_t : problem_t {
  mem_t<int> d_num_cores;
  mem_t<int> d_degrees;
  std::vector<int> num_cores;
  std::vector<int> degrees;
  int largest_k_core;

  struct data_slice_t {
      int *d_num_cores;
      int *d_degrees;

      void init(mem_t<int> &_num_cores, mem_t<int> &_degrees) {
          d_num_cores = _num_cores.data();
          d_degrees = _degrees.data();
      }
  };

  mem_t<data_slice_t> d_data_slice;
  std::vector<data_slice_t> data_slice;

  kcore_problem_t() {}

  kcore_problem_t(const kcore_problem_t& rhs) = delete;
  kcore_problem_t& operator=(const kcore_problem_t& rhs) = delete;

  kcore_problem_t(std::shared_ptr<graph_device_t> rhs, standard_context_t& context) :
      problem_t(rhs),
      largest_k_core(-1),
      data_slice( std::vector<data_slice_t>(1) ) {
          num_cores = std::vector<int>(rhs->num_nodes, 0);
          d_num_cores = fill(0, rhs->num_nodes, context);
          d_degrees = fill(0, rhs->num_nodes, context);
          GetDegrees(d_degrees, context);
          degrees = from_mem(d_degrees);
          data_slice[0].init(d_num_cores, d_degrees);
          d_data_slice = to_mem(data_slice, context);
      }

  void extract() {
      mgpu::dtoh(num_cores, d_num_cores.data(), gslice->num_nodes);
  }

  int cpu(std::vector<int> &validation_num_cores,
           std::vector<int> &row_offsets,
           std::vector<int> &col_indices) {
             // TODO: add CPU_reference impl
             std::vector<bool> to_remove = std::vector<bool>(gslice->num_nodes, false);
             std::vector<bool> to_remain = std::vector<bool>(gslice->num_nodes, true);
             int largest_k_core = -1;
             int num_to_remove, num_to_remain;
             for (int k = 1; k <= gslice->num_nodes; ++k) {
               //reset to_remove
               std::fill(to_remove.begin(), to_remove.end(), false);
               //reset to_remain
               std::fill(to_remain.begin(), to_remain.end(), true);
               while (true) {
                 num_to_remove = 0;
                 for (int vid = 0; vid < gslice->num_nodes; ++vid) {
                   if (degrees[vid] < k && degrees[vid] > 0 && to_remain[vid]) {
                     validation_num_cores[vid] = k - 1;
                     degrees[vid] = 0;
                     to_remove[vid] = true;
                     num_to_remove++;
                   } else to_remove[vid] = false;
                 }
                 num_to_remain = 0;
                 for (int vid = 0; vid < gslice->num_nodes; ++vid) {
                   if (degrees[vid] >= k) {
                     to_remain[vid] = true;
                     num_to_remain++;
                   } else to_remain[vid] = false;
                 }
                 if (num_to_remove == 0) break;
                 else {
                   for (int vid = 0; vid < gslice->num_nodes; ++vid) {
                     if (to_remove[vid]) {
                        int eid_s = row_offsets[vid];
                        int eid_e = row_offsets[vid+1];
                        for (int eid = eid_s; eid < eid_e; ++eid) {
                          int dst_vid = col_indices[eid];
                          degrees[dst_vid]--;
                        }
                        to_remove[vid] = false;
                     }
                   }
                 }
               }
               if (num_to_remain == 0) {
                 largest_k_core = k - 1;
                 break;
               }
             }
             return largest_k_core;
           }
};

} //end kcore
} //end gunrock
