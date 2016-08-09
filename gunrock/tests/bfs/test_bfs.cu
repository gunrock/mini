#include "graph.hxx"
#include "frontier.hxx"
#include "bfs/bfs_problem.hxx"
#include "test_utils.hxx"

using namespace gunrock;

int main(int argc, char** argv) {
    std::string filename;
    CommandLineArgs args(argc, argv);
    args.GetCmdLineArgument("file", filename);

    standard_context_t context;
    
    std::shared_ptr<graph_t> graph = load_graph(filename.c_str());
    std::shared_ptr<graph_device_t> d_graph(std::make_shared<graph_device_t>());
    graph_to_device(d_graph, graph, context);

    frontier_t<int> frontier(context, graph->num_edges, 1.5f);
    frontier.load(d_graph->d_col_indices);
    display_device_data(frontier.data()->data(), frontier.size());

    std::vector<int> source {0};
    frontier.load(source);
    display_device_data(frontier.data()->data(), frontier.size());

    bfs_problem_t test_p(d_graph, 2, context);
    display_device_data(test_p.d_labels.data(), test_p.gslice->num_nodes);
}



