#pragma once

#include "problem.hxx"

namespace gunrock {
namespace bfs {

struct bfs_problem_t : problem_t {
  mem_t<int> d_labels;
  std::vector<int> labels;
  size_t     src;

  struct data_slice_t {
      int *d_labels;

      void init(mem_t<int> &_labels) {
        d_labels = _labels.data();
      }
  };

  data_slice_t *data_slice;
  
  bfs_problem_t() : data_slice(nullptr) {}

  bfs_problem_t(const bfs_problem_t& rhs) = delete;
  bfs_problem_t& operator=(const bfs_problem_t& rhs) = delete;

  bfs_problem_t(std::shared_ptr<graph_device_t> rhs, size_t src, standard_context_t& context) :
      problem_t(rhs),
      src(src) {
          labels = std::vector<int>(rhs->num_nodes, -1);
          labels[src] = 0;
          d_labels = to_mem(labels, context);
          data_slice = new data_slice_t();
          data_slice->init(d_labels);
  }
};

} //end bfs
} //end gunrock
