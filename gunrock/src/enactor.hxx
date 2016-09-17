#pragma once

#include "frontier.hxx"
#include "problem.hxx"
#include "graph.hxx"

using namespace mgpu;

namespace gunrock {

struct enactor_t {
    std::vector< std::shared_ptr<frontier_t<int> > > buffers;
    std::shared_ptr<frontier_t<int> > indices;
    std::shared_ptr<frontier_t<int> > bitmap;

    enactor_t(standard_context_t &context, int num_nodes, int num_edges) {
        init(context, num_nodes, num_edges);
      }

    void init(standard_context_t &context, int num_nodes, int num_edges) {
        std::shared_ptr<frontier_t<int> > input_frontier(std::make_shared<frontier_t<int> >(context, num_edges));
        std::shared_ptr<frontier_t<int> > output_frontier(std::make_shared<frontier_t<int> >(context, num_edges));
        buffers.push_back(input_frontier);
        buffers.push_back(output_frontier);

        indices = std::make_shared<frontier_t<int> >(context, num_edges);
        bitmap = std::make_shared<frontier_t<int> >(context, num_nodes);
        auto gen_idx = [=]__device__(int index) {
            return index;
        };
        mem_t<int> indices_array = mgpu::fill_function<int>(gen_idx, num_nodes, context);
        indices->load(indices_array);

        mem_t<int> bitmap_array = mgpu::fill<int>(0, num_nodes, context);
        bitmap->load(bitmap_array);

    }

  // Disable copy ctor and assignment operator. We don't want to let the
  // user copy only a slice.
  enactor_t(const enactor_t& rhs) = delete;
  enactor_t& operator=(const enactor_t& rhs) = delete;

};

}





