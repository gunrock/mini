#pragma once

#include "problem.hxx"

BEGIN_GUNROCK_NAMESPACE

struct bfs_problem_t : problem_t {
  mem_t<int> d_labels;
  std::vector<int> labels;
  size_t     src;
  
  bfs_problem_t() {}

  bfs_problem_t(const bfs_problem_t& rhs) = delete;
  bfs_problem_t& operator=(const bfs_problem_t& rhs) = delete;

  bfs_problem_t(std::shared_ptr<graph_device_t> rhs, size_t src, standard_context_t& context) :
      problem_t(rhs),
      src(src) {
          labels = std::vector<int>(rhs->num_nodes, -1);
          labels[src] = 0;
          d_labels = to_mem(labels, context);
  }
};

END_GUNROCK_NAMESPACE
