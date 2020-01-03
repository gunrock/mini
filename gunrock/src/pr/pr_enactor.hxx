#pragma once
#include "graph.hxx"
#include "frontier.hxx"

#include "pr_problem.hxx"
#include "pr_functor.hxx"

#include "filter.hxx"
#include "neighborhood.hxx"

#include "enactor.hxx"
#include "test_utils.hxx"

using namespace mgpu;

using namespace gunrock::oprtr::filter;
using namespace gunrock::oprtr::neighborhood;

namespace gunrock {
namespace pr {

struct pr_enactor_t : enactor_t {

    //Constructor
    pr_enactor_t(standard_context_t &context, int num_nodes, int num_edges) :
        enactor_t(context, num_nodes, num_edges)
    {
    }

    pr_enactor_t(const pr_enactor_t& rhs) = delete;
    pr_enactor_t& operator=(const pr_enactor_t& rhs) = delete;

    void init_frontier(std::shared_ptr<pr_problem_t> pr_problem) {
        std::vector<int> node_idx(pr_problem.get()->gslice->num_nodes);
        int count = 0;
        generate(node_idx.begin(), node_idx.end(), [&](){ return count++; });
        buffers[0]->load(node_idx);
    }
    
    //Enact
    void enact(std::shared_ptr<pr_problem_t> pr_problem, standard_context_t &context) {
        init_frontier(pr_problem);

        int frontier_length = pr_problem.get()->gslice->num_nodes;

        int selector = 0;
        int iteration = 0;
        float *reduced_ranks = pr_problem->d_reduced_ranks.data();
        float *current_ranks = pr_problem->d_current_ranks.data();
        while (frontier_length > 0 && iteration < pr_problem.get()->max_iter) {
            std::shared_ptr<frontier_t<int> > dummy_frontier(std::make_shared<frontier_t<int> >(context, pr_problem.get()->gslice->num_nodes));

            neighborhood_kernel<pr_problem_t, pr_functor_t, float, mgpu::plus_t<float>, false, false>
                (pr_problem,
                 buffers[selector],
                 dummy_frontier,
                 reduced_ranks,
                 0.0f,
                 iteration,
                 context);

            //std::cout << "neighborhood reduction.\n"; 
            //display_device_data(pr_problem.get()->d_current_ranks.data(), pr_problem.get()->gslice->num_nodes);
            //display_device_data(pr_problem.get()->d_reduced_ranks.data(), pr_problem.get()->gslice->num_nodes);

            frontier_length = filter_kernel<pr_problem_t, pr_functor_t>
                (pr_problem,
                 buffers[selector],
                 buffers[selector^1],
                 iteration,
                 context);

            std::cout << "finished iteration:" << iteration << " output length: " << frontier_length << std::endl;

            ++iteration;
            selector^=1;
        }
        display_device_data(pr_problem.get()->d_current_ranks.data(), 10);
    }
};

} //pr
} //gunrock
