#include "graph.hxx"
#include "frontier.hxx"
#include "bfs/bfs_problem.hxx"
#include "bfs/bfs_functor.hxx"
#include "test_utils.hxx"

#include "filter.hxx"
#include "advance.hxx"

#include <algorithm>
#include <cstdlib>

using namespace gunrock;
using namespace gunrock::bfs;
using namespace gunrock::oprtr::filter;
using namespace gunrock::oprtr::advance;

int main(int argc, char** argv) {
    std::string filename;
    CommandLineArgs args(argc, argv);
    args.GetCmdLineArgument("file", filename);

    int src = 0;
    args.GetCmdLineArgument("src", src);

    bool idempotence;
    idempotence = args.CheckCmdLineFlag("idempotence");

    standard_context_t context;
    
    std::shared_ptr<graph_t> graph = load_graph(filename.c_str());
    std::shared_ptr<graph_device_t> d_graph(std::make_shared<graph_device_t>());
    graph_to_device(d_graph, graph, context);

    std::shared_ptr<bfs_problem_t> test_p(std::make_shared<bfs_problem_t>(d_graph, src, context));

    std::shared_ptr<frontier_t<int> > input_frontier(std::make_shared<frontier_t<int> >(context, d_graph->num_edges*5) );
    std::vector<int> node_idx(1, src);
    input_frontier->load(node_idx);
    std::shared_ptr<frontier_t<int> > output_frontier(std::make_shared<frontier_t<int> >(context, d_graph->num_edges*5) );
    std::vector< std::shared_ptr<frontier_t<int> > > buffers;
    buffers.push_back(input_frontier);
    buffers.push_back(output_frontier);
    mem_t<unsigned char> visited_mask = idempotence ? mgpu::fill<unsigned char>(0, d_graph->num_nodes, context) : mem_t<unsigned char>(1,context);

    test_timer_t timer;
    timer.start();
    int frontier_length = 1;
    int selector = 0;
    for (int iteration = 0; ; ++iteration) {
        if (idempotence)
            frontier_length = advance_kernel<bfs_problem_t, bfs_functor_t, true>(test_p, buffers[selector], buffers[selector^1], iteration, context);
        else
            frontier_length = advance_kernel<bfs_problem_t, bfs_functor_t, false>(test_p, buffers[selector], buffers[selector^1], iteration, context);
        //std::cout << frontier_length << std::endl;
        //display_device_data(buffers[selector^1].get()->data()->data(), frontier_length);
        //display_device_data(visited_mask.data(), d_graph->num_nodes);
        //display_device_data(test_p.get()->d_labels.data(), test_p.get()->gslice->num_nodes);
        if (!frontier_length) break;
        selector ^= 1;
        if (idempotence)
            uniquify_kernel<bfs_problem_t, bfs_functor_t>(test_p, visited_mask.data(), buffers[selector], buffers[selector^1], iteration, context);
        else
            filter_kernel<bfs_problem_t, bfs_functor_t>(test_p, buffers[selector], buffers[selector^1], iteration, context);
        selector ^= 1;
    }
    cout << "elapsed time: " << timer.end() << "s." << std::endl;

    //display_device_data(test_p.get()->d_labels.data(), test_p.get()->gslice->num_nodes);

    /*std::vector<int> test_uniq(10);
    std::generate(test_uniq.begin(), test_uniq.end(), []{return std::rand()%5;});
    input_frontier->load(test_uniq);
    std::vector<unsigned char> mask(10,0);
    mem_t<unsigned char> visited_mask = to_mem(mask, context);
    uniquify_kernel<bfs_problem_t, bfs_functor_t>(test_p, visited_mask.data(), input_frontier, output_frontier, 0, context);
    std::cout << output_frontier->size() << std::endl;*/
    //display_device_data(output_frontier.get()->data()->data(), output_frontier->size());
}



