#include "graph.hxx"
#include "frontier.hxx"
#include "bfs/bfs_problem.hxx"
#include "bfs/bfs_functor.hxx"
#include "test_utils.hxx"

#include "filter.hxx"

using namespace gunrock;
using namespace gunrock::bfs;
using namespace gunrock::oprtr::filter;

int main(int argc, char** argv) {
    std::string filename;
    CommandLineArgs args(argc, argv);
    args.GetCmdLineArgument("file", filename);

    standard_context_t context;
    
    std::shared_ptr<graph_t> graph = load_graph(filename.c_str());
    std::shared_ptr<graph_device_t> d_graph(std::make_shared<graph_device_t>());
    graph_to_device(d_graph, graph, context);


    std::shared_ptr<frontier_t<int> > frontier(std::make_shared<frontier_t<int> >(context, graph->num_nodes, 1.5f) );

    std::vector<int> node_idx(graph->num_nodes, -1);
    node_idx[1] = 1;
    node_idx[2] = 2;
    node_idx[3] = 3;
    frontier->load(node_idx);
    display_device_data(frontier.get()->data()->data(), frontier->size());

    std::shared_ptr<bfs_problem_t> test_p(std::make_shared<bfs_problem_t>(d_graph, 0, context));
    display_device_data(test_p.get()->d_labels.data(), test_p.get()->gslice->num_nodes);

    std::shared_ptr<frontier_t<int> > output_frontier(std::make_shared<frontier_t<int> >(context, 0, 1) );
    launch_kernel<int, int, bfs_problem_t, bfs_functor_t>(test_p, frontier, output_frontier, context);

    display_device_data(output_frontier.get()->data()->data(), output_frontier->size());

}



