#include "graph.hxx"
#include "frontier.hxx"
#include "bfs/bfs_problem.hxx"
#include "bfs/bfs_functor.hxx"
#include "test_utils.hxx"

#include "filter.hxx"
#include "advance.hxx"

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

    standard_context_t context;
    
    std::shared_ptr<graph_t> graph = load_graph(filename.c_str());
    std::shared_ptr<graph_device_t> d_graph(std::make_shared<graph_device_t>());
    graph_to_device(d_graph, graph, context);

    std::shared_ptr<bfs_problem_t> test_p(std::make_shared<bfs_problem_t>(d_graph, src, context));

    std::shared_ptr<frontier_t<int> > input_frontier(std::make_shared<frontier_t<int> >(context, graph->num_edges) );
    std::vector<int> node_idx(1, src);
    input_frontier->load(node_idx);
    std::shared_ptr<frontier_t<int> > output_frontier(std::make_shared<frontier_t<int> >(context, 0, 1) );
    std::vector< std::shared_ptr<frontier_t<int> > > buffers;
    buffers.push_back(input_frontier);
    buffers.push_back(output_frontier);


    int frontier_length = 1;
    int selector = 0;
    for (int iteration = 0; ; ++iteration) {
        frontier_length = advance_kernel<bfs_problem_t, bfs_functor_t>(test_p, buffers[selector], buffers[selector^1], iteration, context);
        if (!frontier_length) break;
        selector ^= 1;
        filter_kernel<bfs_problem_t, bfs_functor_t>(test_p, buffers[selector], buffers[selector^1], iteration, context);
        selector ^= 1;
    }

    display_device_data(test_p.get()->d_labels.data(), test_p.get()->gslice->num_nodes);
}



