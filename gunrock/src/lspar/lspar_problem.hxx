#pragma once

#include "problem.hxx"

namespace gunrock {
namespace lspar {

struct lspar_problem_t : problem_t {
  mem_t<float> d_sims;
  mem_t<int> d_eids;
  mem_t<int> d_degrees;
  std::vector<float> sims;
  std::vector<int> eids;
  std::vector<int> degrees;
  int k;
  float e;

  struct data_slice_t {
      float *d_sims;
      int *d_eids;
      int *d_degrees;

      void init(mem_t<float> &_sims, mem_t<int> &_eids, mem_t<int> &_degrees) {
        d_sims = _sims.data();
        d_eids = _eids.data();
        d_degrees = _degrees.data();
      }
  };

  mem_t<data_slice_t> d_data_slice;
  std::vector<data_slice_t> data_slice;
  
  lspar_problem_t() {}

  lspar_problem_t(const lspar_problem_t& rhs) = delete;
  lspar_problem_t& operator=(const lspar_problem_t& rhs) = delete;

  lspar_problem_t(std::shared_ptr<graph_device_t> rhs, int k, float e, standard_context_t& context) :
      problem_t(rhs),
      k(k),
      e(e),
      data_slice( std::vector<data_slice_t>(1) ) {
          sims = std::vector<float>(rhs->num_edges, 0);
          eids = std::vector<int>(rhs->num_edges, -1);
          degrees = std::vector<int>(rhs->num_nodes, 0);
          d_sims = to_mem(sims, context);
          d_eids = to_mem(eids, context);
          d_degrees = to_mem(degrees, context);
          GetDegrees(d_degrees, context);
          data_slice[0].init(d_sims, d_eids, d_degrees);
          d_data_slice = to_mem(data_slice, context);
      } 

  void extract() {
      // TODO: output the selected edge ids?
  }
};

} //end lspar
} //end gunrock
