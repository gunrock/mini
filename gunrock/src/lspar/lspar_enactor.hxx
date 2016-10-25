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
        std::vector<int> edge_idx(lspar_problem.get()->gslice->num_edges);
        int count = 0;
        generate(edge_idx.begin(), edge_idx.end(), [&](){ return count++; });
        buffers[0]->load(edge_idx);
        buffers[1]->load(edge_idx);
    }

    
    //Enact
    int enact(std::shared_ptr<lspar_problem_t> lspar_problem, standard_context_t &context) {
        init_frontier(lspar_problem);

        int frontier_length = lspar_problem.get()->gslice->num_nodes;

        typename Problem::data_slice_t *data = problem.get()->d_data_slice.data();

        //one neighborhood reduction to compute minwise hash
        int selector = 0;
        int iteration = 0;
        int *minwise_hashs = lspar_problem.get()->d_minwise_hashs.data();
        neighborhood_kernel<lspar_problem_t, minhash_functor_t, int, mgpu::minimum_t<int>, false >
            (lspar_problem,
             buffers[0],
             buffers[1],
             minwise_hashs,
             std::numeric_limits<int>::max(),
             iteration,
             context);

        //one advance to compute per-edge sim values
        advance_forward_kernel<lspar_problem_t, sim_functor_t, false, false>
            (lspar_problem,
             buffers[0],
             buffers[1],
             iteration,
             context);

        //now we get all the sims stored, time to do segsort
        sim_edge_t *sims = lspar_problem.get()->d_sims.data();
        sim_edge_t *sortd_sims = lspar_problem.get()->d_sorted_sims.data();
        int *offsets = lspar_problem.get()->gslice->d_sorted_sims.data();
        int num_edges = lspar_problem.get()->gslice->num_edges;
        int num_nodes = lspar_problem.get()->gslice->num_nodes;
        segmented_sort_indices(sims, sorted_sims, num_edges,offsets,num_nodes,less_sim(), context);
       
        //advance to tag selected edges
        //output could be a tuple (sid,did)
        advance_forward_kernel<lspar_problem_t, select_functor_t, false, true>
            (lspar_problem,
             buffers[0],
             buffers[1],
             iteration,
             context);

        //directly load a transform_compact 
        auto compact = transform_compact(num_edges, context);
        sim_edge_t *input_data = buffers[1].get()->data()->data();
        typename Problem::data_slice_t *data = problem.get()->d_data_slice.data();
        frontier_length = compact.upsweep([=]__device__(int idx) {
                int item = input_data[idx];
                return item != -1;
                });
        output->resize(stream_count);
        sim_edge_t *sims = lspar_problem.get()->d_sims.data();
        sim_edge_t *sortd_sims = lspar_problem.get()->d_sorted_sims.data();
        compact.downsweep([=]__device__(int dest_idx, int source_idx) {
                sim[dest_idx] = sorted_sims[source_idx];
                });

        return frontier_length;
    }

};

} //lspar
} //gunrock
