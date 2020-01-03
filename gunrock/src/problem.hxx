#pragma once

#include "graph.hxx"

namespace gunrock {

struct problem_t {
  std::shared_ptr<graph_device_t> gslice;

  problem_t() : gslice(std::make_shared<graph_device_t>())
      {}

  // Disable copy ctor and assignment operator. We don't want to let the
  // user copy only a slice.
  problem_t(const problem_t& rhs) = delete;
  problem_t& operator=(const problem_t& rhs) = delete;

  problem_t(std::shared_ptr<graph_device_t> rhs) {
      gslice = rhs;
  }

              
  void GetDegrees(mem_t<float> &_degrees, standard_context_t &context) {
      float *degrees = _degrees.data();
      int *offsets = gslice->d_row_offsets.data();
      auto f = [=]__device__(int idx) {
          degrees[idx] = offsets[idx+1]-offsets[idx];
      };
      transform(f, gslice->num_nodes, context);
  }
};

}





