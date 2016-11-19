#pragma once
#include "graph.hxx"
#include "frontier.hxx"

#include "bfs_problem.hxx"
#include "bfs_functor.hxx"

#include "filter.hxx"
#include "advance.hxx"

#include "enactor.hxx"

#include "test_utils.hxx"

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
    void enact_pushpull(std::shared_ptr<bfs_problem_t> bfs_problem, float threshold, standard_context_t &context) {
        init_frontier(bfs_problem);

        int frontier_length = 1;
        int selector = 0;
        int num_nodes = bfs_problem.get()->gslice->num_nodes;
        int num_unvisited = num_nodes - 1;
        int iteration;

        for (iteration = 0; ; ++iteration) {
            //std::cout << "push " << iteration << std::endl;
            frontier_length = advance_forward_kernel<bfs_problem_t, bfs_functor_t, false, true>
                (bfs_problem,
                 buffers[selector],
                 buffers[selector^1],
                 iteration,
                 context);
            selector ^= 1;
            if (!frontier_length) break;
            frontier_length = filter_kernel<bfs_problem_t, bfs_functor_t>
                (bfs_problem,
                 buffers[selector],
                 buffers[selector^1],
                 iteration,
                 context);
            num_unvisited -= frontier_length;

            if ((float)num_unvisited < (float)frontier_length * threshold) break;
            if (!frontier_length) break;
            selector ^= 1;
        }
        std::cout << "pushed iterations: " << iteration << std::endl;

        if (frontier_length) {
            // break due to push-pull switch, enter pull-based traversal

            // Preparation:
            ++iteration;
            // Generate unvisited array as input frontier
            frontier_length = gen_unvisited_kernel<bfs_problem_t, bfs_functor_t>(bfs_problem, unvisited[selector^1], unvisited[selector], 0, context);

            int new_frontier_length;
            mem_t<int> bitmap_array = mgpu::fill<int>(0, num_nodes, context);
            buffers[selector]->load(bitmap_array);
            // Generate bitmap array as auxiliary frontier
            sparse_to_dense_kernel<bfs_problem_t, bfs_functor_t>(bfs_problem, buffers[selector^1], buffers[selector], iteration, context);

            for (;;++iteration) {
                //std::cout << "pull " << iteration << std::endl;
                buffers[selector^1]->load(bitmap_array);
                advance_backward_kernel<bfs_problem_t, bfs_functor_t>(
                        bfs_problem,
                        unvisited[selector],
                        buffers[selector],
                        buffers[selector^1],
                        iteration,
                        context);

                new_frontier_length = filter_kernel<bfs_problem_t, bfs_functor_t>(
                        bfs_problem,
                        unvisited[selector],
                        unvisited[selector^1],
                        iteration,
                        context);

                if (!new_frontier_length || new_frontier_length == frontier_length) break;
                frontier_length = new_frontier_length;
                selector ^= 1;

            }

            std::cout << "total iterations: " << iteration << std::endl;

        } else {
            // break due to finished BFS. No pull-based traversal
        }
    }
   
};

} //bfs
} //gunrock
