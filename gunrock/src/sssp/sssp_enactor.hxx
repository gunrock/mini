#pragma once
#include "graph.hxx"
#include "frontier.hxx"

#include "sssp_problem.hxx"
#include "sssp_functor.hxx"

#include "filter.hxx"
#include "advance.hxx"

#include "enactor.hxx"
#include "test_utils.hxx"

using namespace mgpu;

using namespace gunrock::oprtr::advance;
using namespace gunrock::oprtr::filter;

namespace gunrock {
namespace sssp {

struct sssp_enactor_t : enactor_t {

    //Constructor
    sssp_enactor_t(standard_context_t &context, int num_nodes, int num_edges, float queue_sizing) :
        enactor_t(context, num_nodes, num_edges, queue_sizing)
    {
    }

    sssp_enactor_t(const sssp_enactor_t& rhs) = delete;
    sssp_enactor_t& operator=(const sssp_enactor_t& rhs) = delete;

    void init_frontier(std::shared_ptr<sssp_problem_t> sssp_problem) {
        int src = sssp_problem->src;
        std::vector<int> node_idx(1, src);
        buffers[0]->load(node_idx);
    }
    
    //Enact
    void enact(std::shared_ptr<sssp_problem_t> sssp_problem, standard_context_t &context) {
        init_frontier(sssp_problem);

        int frontier_length = 1;
        int selector = 0;
        int num_nodes = sssp_problem.get()->gslice->num_nodes;
        int iteration;

        for (iteration = 0; ; ++iteration) {
            frontier_length = advance_forward_kernel<sssp_problem_t, sssp_functor_t, false, true>
                (sssp_problem,
                 buffers[selector],
                 buffers[selector^1],
                 iteration,
                 context);

        //display_device_data(sssp_problem.get()->d_labels.data(), sssp_problem.get()->gslice->num_nodes);

        //display_device_data(buffers[selector^1].get()->data()->data(), buffers[selector^1]->size());
            selector ^= 1;
            frontier_length = filter_kernel<sssp_problem_t, sssp_functor_t>
                (sssp_problem,
                 buffers[selector],
                 buffers[selector^1],
                 iteration,
                 context);
            if (!frontier_length) break;
            //display_device_data(buffers[selector^1].get()->data()->data(), buffers[selector^1]->size());

        //cout << "===========\n";
            selector ^= 1;
        }
    }
   
};

} //sssp
} //gunrock
