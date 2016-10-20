#pragma once
#include "graph.hxx"
#include "frontier.hxx"

#include "bfs_problem.hxx"
#include "bfs_functor.hxx"

#include "filter.hxx"
#include "advance.hxx"

#include "enactor.hxx"

using namespace mgpu;

using namespace gunrock::oprtr::advance;
using namespace gunrock::oprtr::filter;

namespace gunrock {
namespace bfs {

struct bfs_enactor_t : enactor_t {

    //Constructor
    bfs_enactor_t(standard_context_t &context, int num_nodes, int num_edges) :
        enactor_t(context, num_nodes, num_edges)
    {
    }

    bfs_enactor_t(const bfs_enactor_t& rhs) = delete;
    bfs_enactor_t& operator=(const bfs_enactor_t& rhs) = delete;

    void init_frontier(std::shared_ptr<bfs_problem_t> bfs_problem) {
        int src = bfs_problem->src;
        std::vector<int> node_idx(1, src);
        buffers[0]->load(node_idx);
    }
    
    //Enact
    void enact(std::shared_ptr<bfs_problem_t> bfs_problem, standard_context_t &context) {
        init_frontier(bfs_problem);

        int frontier_length = 1;
        int selector = 0;

        for (int iteration = 0; ; ++iteration) {
            frontier_length = advance_forward_kernel<bfs_problem_t, bfs_functor_t, false, true>
                (bfs_problem,
                 buffers[selector],
                 buffers[selector^1],
                 iteration,
                 context);
            selector ^= 1;
            frontier_length = filter_kernel<bfs_problem_t, bfs_functor_t>
                (bfs_problem,
                 buffers[selector],
                 buffers[selector^1],
                 iteration,
                 context);
            if (!frontier_length) break;
            selector ^= 1;
        }
    }
   
};

} //bfs
} //gunrock
