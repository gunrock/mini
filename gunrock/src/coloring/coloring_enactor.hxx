#pragma once
#include "graph.hxx"
#include "frontier.hxx"

#include "coloring_problem.hxx"
#include "coloring_functor.hxx"

#include "filter.hxx"
#include "neighborhood.hxx"

#include "enactor.hxx"

using namespace mgpu;

using namespace gunrock::oprtr::filter;
using namespace gunrock::oprtr::neighborhood;

namespace gunrock {
namespace coloring {

struct coloring_enactor_t : enactor_t {

    //Constructor
    coloring_enactor_t(standard_context_t &context, int num_nodes, int num_edges) :
        enactor_t(context, num_nodes, num_edges)
    {
    }

    coloring_enactor_t(const coloring_enactor_t& rhs) = delete;
    coloring_enactor_t& operator=(const coloring_enactor_t& rhs) = delete;

    void init_frontier(std::shared_ptr<coloring_problem_t> coloring_problem,
                       std::shared_ptr<frontier_t<int> > frontier) {
        std::vector<int> node_idx(coloring_problem.get()->gslice->num_nodes);
        int count = 0;
        generate(node_idx.begin(), node_idx.end(), [&](){ return count++; });
        frontier->load(node_idx);
        buffers[0]->load(node_idx);
    }
    
    //Enact
    void enact(std::shared_ptr<coloring_problem_t> coloring_problem, standard_context_t &context) {
        std::shared_ptr<frontier_t<int> > full_frontier(std::make_shared<frontier_t<int> >(context, coloring_problem.get()->gslice->num_nodes));
        init_frontier(coloring_problem, full_frontier);

        int frontier_length = coloring_problem.get()->gslice->num_nodes;

        //one neighborhood to compute min, one neighborhood to compute max
        //one filter to fill in colors, return unset color, store frontier
        //reset hashs, until all colors are set
        int selector = 0;
        int iteration = 0;
        int *reduced_max = coloring_problem->d_reduced_max.data();
        int *reduced_min = coloring_problem->d_reduced_min.data();
        while (frontier_length > 0 && iteration < coloring_problem.get()->max_iter) {
            neighborhood_kernel<coloring_problem_t, coloring_functor_t, int, mgpu::maximum_t<int>, false >
                (coloring_problem,
                 full_frontier,
                 full_frontier,
                 reduced_max,
                 std::numeric_limits<int>::min(),
                 iteration,
                 context);

            neighborhood_kernel<coloring_problem_t, coloring_functor_t, int, mgpu::minimum_t<int>, false >
                (coloring_problem,
                 full_frontier,
                 full_frontier,
                 reduced_min,
                 std::numeric_limits<int>::max(),
                 iteration,
                 context);

            frontier_length = filter_kernel<coloring_problem_t, coloring_functor_t>
                (coloring_problem,
                 buffers[selector],
                 buffers[selector^1],
                 iteration,
                 context);

            selector ^= 1;
            ++iteration;
        } 
    }
   
};

} //coloring
} //gunrock
