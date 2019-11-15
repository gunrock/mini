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
      largest_k_core(0),
      data_slice( std::vector<data_slice_t>(1) ) {
          num_cores = std::vector<int>(rhs->num_nodes, 0);
          d_num_cores = fill(0, rhs->num_nodes, context);
          d_degrees = fill(0, rhs->num_nodes, context);
          GetDegrees(d_degrees, context);
          degrees = from_mem(d_degrees, context);
          data_slice[0].init(d_num_cores, d_degrees);
          d_data_slice = to_mem(data_slice, context);
      }

  void extract() {
      mgpu::dtoh(num_cores, d_num_cores.data(), gslice->num_nodes);
  }

  void cpu(std::vector<int> &validation_num_cores,
           std::vector<int> &row_offsets,
           std::vector<int> &col_indices) {
             // TODO: add CPU_reference impl
           }
};

} //end kcore
} //end gunrock
