#pragma once

#include "problem.hxx"

namespace gunrock {
namespace pr {

struct pr_problem_t : problem_t {
  mem_t<float> d_current_ranks;
  mem_t<float> d_reduced_ranks;
  mem_t<float> d_degrees;
  int max_iter;

  struct data_slice_t {
      float *d_current_ranks;
      float *d_reduced_ranks;
      float *d_degrees;

      void init(mem_t<float> &_current_ranks, mem_t<float> &_reduced_ranks, mem_t<float> &_degrees) {
          d_current_ranks = _current_ranks.data();
          d_reduced_ranks = _reduced_ranks.data();
          d_degrees = _degrees.data();
      }
  };

  mem_t<data_slice_t> d_data_slice;
  std::vector<data_slice_t> data_slice;
  
  pr_problem_t() {}

  pr_problem_t(const pr_problem_t& rhs) = delete;
  pr_problem_t& operator=(const pr_problem_t& rhs) = delete;

  pr_problem_t(std::shared_ptr<graph_device_t> rhs, int max_iter, standard_context_t& context) :
      problem_t(rhs),
      max_iter(max_iter),
      data_slice( std::vector<data_slice_t>(1) ) {
          d_current_ranks = fill(0.15f, rhs->num_nodes, context);
          d_reduced_ranks = fill(0.0f, rhs->num_nodes, context);
          d_degrees = fill(0.0f, rhs->num_nodes, context);
          GetDegrees(d_degrees, context);
          data_slice[0].init(d_current_ranks, d_reduced_ranks, d_degrees);
          d_data_slice = to_mem(data_slice, context);
      }

  void extract() {
      // TODO: output the selected edge ids?
  }
};

} //end coloring
} //end gunrock
