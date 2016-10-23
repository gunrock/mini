#pragma once

#include "problem.hxx"

namespace gunrock {
namespace coloring {

struct coloring_problem_t : problem_t {
  mem_t<int> d_reduced_max;
  mem_t<int> d_reduced_min;
  mem_t<int> d_hashs;
  mem_t<int> d_colors;
  int prime;
  int max_iter;

  struct data_slice_t {
      int *d_reduced_max;
      int *d_reduced_min;
      int *d_hashs;
      int *d_colors;

      void init(mem_t<int> &_reduced_max, mem_t<int> &_reduced_min, mem_t<int> &_hashs, mem_t<int> &_colors) {
          d_reduced_max = _reduced_max.data();
          d_reduced_min = _reduced_min.data();
          d_hashs = _hashs.data();
          d_colors = _colors.data();
      }
      void set_hashs(mem_t<int> &_hashs) {
          d_hashs = _hashs.data();
      }
  };

  mem_t<data_slice_t> d_data_slice;
  std::vector<data_slice_t> data_slice;
  
  coloring_problem_t() {}

  coloring_problem_t(const coloring_problem_t& rhs) = delete;
  coloring_problem_t& operator=(const coloring_problem_t& rhs) = delete;

  coloring_problem_t(std::shared_ptr<graph_device_t> rhs, int prime, int max_iter, standard_context_t& context) :
      problem_t(rhs),
      prime(prime),
      max_iter(max_iter),
      data_slice( std::vector<data_slice_t>(1) ) {
          d_reduced_max = fill(0, rhs->num_nodes, context);
          d_reduced_min = fill(0, rhs->num_nodes, context);
          d_colors = fill(0, rhs->num_nodes, context);
          d_hashs = fill_random(0, prime, rhs->num_nodes, false, context);
          data_slice[0].init(d_reduced_max, d_reduced_min, d_hashs, d_colors);
          d_data_slice = to_mem(data_slice, context);
      }
  void reset_hashs(standard_context_t& context) {
          d_hashs = fill_random(0, prime, gslice->num_nodes, false, context);
          data_slice[0].set_hashs(d_hashs);
          d_data_slice = to_mem(data_slice, context);
  }

  void extract() {
      // TODO: output the selected edge ids?
  }
};

} //end coloring
} //end gunrock
