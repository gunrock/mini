#pragma once

#include "graph.hxx"

BEGIN_GUNROCK_NAMESPACE

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
};

END_GUNROCK_NAMESPACE





