#pragma once

#include "problem.hxx"

namespace gunrock {
namespace lspar {

struct lspar_problem_t : problem_t {
    struct sim_edge_t {
        int eid;
        float sim;
        sim_edge_t() = default;
    };

  mem_t<sim_edge_t> d_sims;
  mem_t<sim_edge_t> d_sorted_sims;
  mem_t<int> d_thresholds;
  mem_t<int> d_params;
  mem_t<int> d_hashs; // For now only allow k=1, for k>1, needs specialized reduce_op to do struct min/max
  mem_t<int> d_minwise_hashs;
  std::vector<int> thresholds;
  std::vector<int> params;

  struct data_slice_t {
      sim_edge_t *d_sims;
      sim_edge_t *d_sorted_sims;
      int *d_thresholds;
      int *d_hashs;
      int *d_minwise_hashs;

      void init(mem_t<sim_edge_t> &_sims, mem_t<sim_edge_t> &_sorted_sims, mem_t<int> &_thresholds, mem_t<int> &_hashs, mem_t<int> & _minwise_hashs) {
        d_sims = _sims.data();
        d_sorted_sims = _sorted_sims.data();
        d_thresholds = _thresholds.data();
        d_hashs = _hashs.data();
        d_minwise_hashs = _minwise_hashs.data();
      }
  };

  mem_t<data_slice_t> d_data_slice;
  std::vector<data_slice_t> data_slice;
  
  lspar_problem_t() {}

  lspar_problem_t(const lspar_problem_t& rhs) = delete;
  lspar_problem_t& operator=(const lspar_problem_t& rhs) = delete;

  void ComputeThresholds(mem_t<int> &_thres, float e, standard_context_t &context) {
      int *thres = _thres.data();
      auto f = [=]__device__(int idx) {
          thres[idx] = floor(__powf((float)thres[idx], e));
      };
      transform(f, gslice->num_nodes, context);
  }

  void ComputeHashs(mem_t<int> &_params, mem_t<int> &_hashs, standard_context_t &context) {
      int *params = _params.data();
      int *hashs = _hashs.data();
      auto f = [=]__device__(int idx) {
          int p = ldg(params);
          int a = ldg(params+1);
          int b = ldg(params+2);
          hashs[idx] = (b+a*idx)%p;
      };
      transform(f, gslice->num_nodes, context);
  }

  //http://stackoverflow.com/questions/4424374/determining-if-a-number-is-prime/4424496#4424496
  inline bool is_prime( int number )
  {
      if ( ( (!(number & 1)) && number != 2 ) || (number < 2) || (number % 3 == 0 && number != 3) )
          return (false);

      for( int k = 1; 36*k*k-12*k < number;++k)
          if ( (number % (6*k+1) == 0) || (number % (6*k-1) == 0) )
              return (false);
      return true;
  }

  lspar_problem_t(std::shared_ptr<graph_device_t> rhs, int prime, int k, float e, standard_context_t& context) :
      problem_t(rhs),
      data_slice( std::vector<data_slice_t>(1) ) {
          thresholds = std::vector<int>(rhs->num_nodes, 0);
          d_hashs = fill<int>(0, rhs->num_nodes, context);
          d_sims = fill<sim_edge_t>(sim_edge_t(), rhs->num_edges, context);
          d_thresholds = to_mem(thresholds, context);
          GetDegrees(d_thresholds, context);
          //compute threshold using transform
          ComputeThresholds(d_thresholds, e, context);
          //test prime, if not prime exit.
          if (!is_prime(prime)) {
              exit(1);
          }
          //store prime in d_params[0], store i\in(1..k) a_i in d_params[1..k]
          params[0] = prime;
          params[1] = std::rand()%(prime-1)+1;
          params[2] = std::rand()%prime;
          //std::generate(params.begin()+1, params.begin()+k+1, (std::rand()%(prime-1))+1);
          //b_i in d_params[k+1..2k]
          //std::generate(params.begin()+k+1, params.end(), std::rand()%prime);
          d_params = to_mem(params, context);
          ComputeHashs(d_params, d_hashs, context);
          d_minwise_hashs = fill(0, rhs->num_nodes, context);
          d_sorted_sims = fill<sim_edge_t>(sim_edge_t(), rhs->num_edges, context);
          data_slice[0].init(d_sims, d_sorted_sims, d_thresholds, d_hashs, d_minwise_hashs);
          d_data_slice = to_mem(data_slice, context);
      } 

  void extract() {
      // TODO: output the selected edge ids?
  }
};

} //end lspar
} //end gunrock
