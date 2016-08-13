#pragma once

#include "problem.hxx"

namespace gunrock {
namespace bfs {

struct bfs_problem_t : problem_t {
  mem_t<int> d_labels;
  std::vector<int> labels;
  int src;

  struct data_slice_t {
      int *d_labels;

      void init(mem_t<int> &_labels) {
        d_labels = _labels.data();
      }
  };

  mem_t<data_slice_t> d_data_slice;
  std::vector<data_slice_t> data_slice;
  
  bfs_problem_t() {}

  bfs_problem_t(const bfs_problem_t& rhs) = delete;
  bfs_problem_t& operator=(const bfs_problem_t& rhs) = delete;

  bfs_problem_t(std::shared_ptr<graph_device_t> rhs, size_t src, standard_context_t& context) :
      problem_t(rhs),
      src(src),
      data_slice( std::vector<data_slice_t>(1) ) {
          labels = std::vector<int>(rhs->num_nodes, -1);
          labels[src] = 0;
          d_labels = to_mem(labels, context);
          data_slice[0].init(d_labels);
          d_data_slice = to_mem(data_slice, context);
  }
};

} //end bfs
} //end gunrock
