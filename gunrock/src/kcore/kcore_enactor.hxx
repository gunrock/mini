#pragma once
#include "graph.hxx"
#include "frontier.hxx"

#include "kcore_problem.hxx"
#include "kcore_functor.hxx"

#include "filter.hxx"
#include "advance.hxx"

#include "enactor.hxx"
#include "test_utils.hxx"

using namespace mgpu;

using namespace gunrock::oprtr::filter;
using namespace gunrock::oprtr::advance;

namespace gunrock {
namespace kcore {

struct kcore_enactor_t : enactor_t {

    //Constructor
    kcore_enactor_t(standard_context_t &context, int num_nodes, int num_edges) :
        enactor_t(context, num_nodes, num_edges)
    {
    }

    kcore_enactor_t(const kcore_enactor_t& rhs) = delete;
    kcore_enactor_t& operator=(const kcore_enactor_t& rhs) = delete;

    void init_frontier(std::shared_ptr<kcore_problem_t> kcore_problem) {
        std::vector<int> node_idx(kcore_problem.get()->gslice->num_nodes);
        int count = 0;
        generate(node_idx.begin(), node_idx.end(), [&](){ return count++; });
        buffers[0]->load(node_idx);
    }

    //Enact
    void enact(std::shared_ptr<kcore_problem_t> kcore_problem, standard_context_t &context) {
        int num_nodes = kcore_problem.get()->gslice->num_nodes;
        int frontier_length = num_nodes;
        int selector = 0;
        for (int k = 1; k <= num_nodes; ++k) {
          init_frontier(kcore_problem);
          selector = 0;
          while (true) {
            int num_to_remove = filter_kernel<kcore_problem_t, deg_less_than_k_functor_t>
                (kcore_problem,
                 buffers[selector],
                 buffers[selector^1],
                 k,
                 context);
            if (num_to_remove == 0) break;
            else {
              advance_forward_kernel<kcore_problem_t, update_deg_functor_t,
                /*idempotent=*/false, /*has_output=*/false>
                (kcore_problem,
                 buffers[selector^1],
                 buffers[selector],
                 k,
                 context);
              frontier_length = filter_kernel<kcore_problem_t, deg_atleast_k_functor_t>
                (kcore_problem,
                 buffers[selector],
                 buffers[selector^1],
                 k,
                 context);
            }
            frontier_length = filter_kernel<kcore_problem_t, deg_atleast_k_functor_t>
                (kcore_problem,
                 buffers[selector],
                 buffers[selector^1],
                 k,
                 context);
          }
          if (frontier_length == 0) {
            std::cout << "largest k-core: " << k - 1 << std::endl;
            kcore_problem.largest_k_core = k - 1;
            break;
          }
        }
    }
};

} //kcore
} //gunrock
