#pragma once
#include "graph.hxx"
#include "frontier.hxx"

#include "lspar_problem.hxx"
#include "lspar_functor.hxx"

#include "filter.hxx"
#include "advance.hxx"
#include "neighborhood.hxx"

#include "enactor.hxx"

using namespace mgpu;

using namespace gunrock::oprtr::advance;
using namespace gunrock::oprtr::filter;

namespace gunrock {
namespace lspar {

struct lspar_enactor_t : enactor_t {

    //Constructor
    lspar_enactor_t(standard_context_t &context, int num_nodes, int num_edges) :
        enactor_t(context, num_nodes, num_edges)
    {
    }

    lspar_enactor_t(const lspar_enactor_t& rhs) = delete;
    lspar_enactor_t& operator=(const lspar_enactor_t& rhs) = delete;

    void init_frontier(std::shared_ptr<lspar_problem_t> lspar_problem) {
        std::vector<int> node_idx(lspar_problem.get()->gslice->num_nodes);
        int count = 0;
        generate(node_idx.begin(), node_idx.end(), [&](){ return count++; });
        buffers[0]->load(node_idx);
    }
    
    //Enact
    void enact(std::shared_ptr<lspar_problem_t> lspar_problem, standard_context_t &context) {
        init_frontier(lspar_problem);

        int frontier_length = lspar_problem.get()->gslice->num_nodes;

        typename Problem::data_slice_t *data = problem.get()->d_data_slice.data();
        int k = lspar_problem->k;
        float e = lspar_problem->e;

        //one compute step to compute k hash with (a,b,P) values
        //one neighborhood reduction to compute minwise hash
        //one advance to compute per-edge sim values

        frontier_length = advance_forward_kernel<lspar_problem_t, lspar_functor_t, false, false>
            (lspar_problem,
             buffers[0],
             buffers[1],
             iteration,
             context);

        //now we get all the sims stored, time to do segsort
        
        //now assign eids according to the d^e order. this is a filter. but
        //what to output? maybe a tuple of (source node, dst node) would be
        //nice.
    }
   
};

} //lspar
} //gunrock
