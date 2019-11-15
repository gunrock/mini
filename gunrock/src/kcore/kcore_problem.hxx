#pragma once

#include "problem.hxx"

namespace gunrock {
namespace kcore {

struct kcore_problem_t : problem_t {
  mem_t<int> d_num_cores;
  mem_t<int> d_degrees;

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
      data_slice( std::vector<data_slice_t>(1) ) {
          d_num_cores = fill(0, rhs->num_nodes, context);
          d_degrees = fill(0, rhs->num_nodes, context);
          GetDegrees(d_degrees, context);
          data_slice[0].init(d_num_cores, d_degrees);
          d_data_slice = to_mem(data_slice, context);
      }

  void extract() {
      // TODO: output num_cores for each node
  }
};

} //end kcore
} //end gunrock