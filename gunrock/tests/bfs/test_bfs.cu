#include "graph.hxx"
#include "frontier.hxx"
#include "bfs/bfs_problem.hxx"
#include "bfs/bfs_functor.hxx"
#include "test_utils.hxx"

#include "filter.hxx"
#include "advance.hxx"
#include "neighborhood.hxx"

#include <algorithm>
#include <cstdlib>

using namespace gunrock;
using namespace gunrock::bfs;
using namespace gunrock::oprtr::filter;
using namespace gunrock::oprtr::advance;

int main(int argc, char** argv) {

    // read in graph file
    std::string filename;
    CommandLineArgs args(argc, argv);
    args.GetCmdLineArgument("file", filename);

    // read in source node from cmd line
    int src = 0;
    args.GetCmdLineArgument("src", src);

    // CUDA context is used for all mgpu transforms
    standard_context_t context;
   
    // Load graph data to device
    std::shared_ptr<graph_t> graph = load_graph(filename.c_str());
    std::shared_ptr<graph_device_t> d_graph(std::make_shared<graph_device_t>());
    graph_to_device(d_graph, graph, context);

    // Initializes bfs problem object
    std::shared_ptr<bfs_problem_t> bfs_problem(std::make_shared<bfs_problem_t>(d_graph, src, context));

    // Initializes ping-pong buffers
    std::shared_ptr<frontier_t<int> > input_frontier(std::make_shared<frontier_t<int> >(context, d_graph->num_edges) );         
    std::shared_ptr<frontier_t<int> > output_frontier(std::make_shared<frontier_t<int> >(context, d_graph->num_edges) );
    std::vector< std::shared_ptr<frontier_t<int> > > buffers;
    buffers.push_back(input_frontier);
    buffers.push_back(output_frontier);

    // Generate unvisited array as input frontier
    std::shared_ptr<frontier_t<int> > init_indices(std::make_shared<frontier_t<int> >(context, d_graph->num_nodes));
    auto gen_idx = [=]__device__(int index) {
        return index;
    };
    mem_t<int> indices = mgpu::fill_function<int>(gen_idx, d_graph->num_nodes, context);
    init_indices->load(indices);
    gen_unvisited_kernel<bfs_problem_t, bfs_functor_t>(bfs_problem, init_indices, buffers[0], 0, context);
    mem_t<int> bitmap_array = mgpu::fill<int>(0, d_graph->num_nodes, context);
    std::shared_ptr<frontier_t<int> > bitmap(std::make_shared<frontier_t<int> >(context, d_graph->num_nodes) );
    bitmap->load(bitmap_array);
    
    // Generate bitmap array as auxiliary frontier
    std::vector<int> node_idx(1, src);
    output_frontier->load(node_idx);
    sparse_to_dense_kernel<bfs_problem_t, bfs_functor_t>(bfs_problem, buffers[1], bitmap, 0, context);

    test_timer_t timer;
    timer.start();
    int frontier_length = d_graph->num_nodes - 1;
    int selector = 0;

    
    //display_device_data(buffers[0].get()->data()->data(), buffers[0]->size());
    //display_device_data(bitmap.get()->data()->data(), d_graph->num_nodes); 

    for (int iteration = 0; ; ++iteration) {
        advance_backward_kernel<bfs_problem_t, bfs_functor_t>(
                bfs_problem,
                buffers[selector],
                bitmap,
                buffers[selector^1],
                iteration,
                context);

        selector ^= 1;

        //display_device_data(bitmap.get()->data()->data(), d_graph->num_nodes); 

        filter_kernel<bfs_problem_t, bfs_functor_t>(
                bfs_problem,
                buffers[selector],
                buffers[selector^1],
                iteration,
                context);

        selector ^= 1;

        if (!buffers[selector]->size()) break;
    }

    cout << "elapsed time: " << timer.end() << "s." << std::endl;

    display_device_data(bfs_problem.get()->d_labels.data(), bfs_problem.get()->gslice->num_nodes);

    /*for (int iteration = 0; ; ++iteration) {
        if (idempotence)
            frontier_length = advance_kernel<bfs_problem_t, bfs_functor_t, true>(test_p, buffers[selector], buffers[selector^1], iteration, context);
        else
            frontier_length = advance_kernel<bfs_problem_t, bfs_functor_t, false>(test_p, buffers[selector], buffers[selector^1], iteration, context);
        //std::cout << frontier_length << std::endl;
        //display_device_data(buffers[selector^1].get()->data()->data(), frontier_length);
        //display_device_data(visited_mask.data(), d_graph->num_nodes);
        //display_device_data(test_p.get()->d_labels.data(), test_p.get()->gslice->num_nodes);
        //gen_unvisited_kernel(test_p, buffers[selector^1], unvisited, context);
        //gen_bitmap_kernel(buffers[selector^1], bitmap, context);
        //display_device_data(bitmap.get()->data()->data(), d_graph->num_nodes);
        selector ^= 1;
        if (idempotence)
            uniquify_kernel<bfs_problem_t, bfs_functor_t>(test_p, visited_mask.data(), buffers[selector], buffers[selector^1], iteration, context);
        else
            filter_kernel<bfs_problem_t, bfs_functor_t>(test_p, buffers[selector], buffers[selector^1], iteration, context);
        //std::cout << buffers[selector^1]->size() << std::endl;
        if (!buffers[selector^1]->size()) break;
        selector ^= 1;
    }
    cout << "elapsed time: " << timer.end() << "s." << std::endl;

    display_device_data(test_p.get()->d_labels.data(), test_p.get()->gslice->num_nodes);*/

    /*std::vector<int> test_uniq(4000);
    int num_nodes = d_graph->num_nodes;
    std::generate(test_uniq.begin(), test_uniq.end(), [=]{return std::rand()%num_nodes;});
    input_frontier->load(test_uniq);
    std::vector<unsigned char> mask(4000,0);
    mem_t<unsigned char> visited_mask = to_mem(mask, context);
    uniquify_kernel<bfs_problem_t, bfs_functor_t>(test_p, visited_mask.data(), input_frontier, output_frontier, 0, context);
    std::cout << output_frontier->size() << std::endl;*/
    //display_device_data(output_frontier.get()->data()->data(), output_frontier->size());
}



