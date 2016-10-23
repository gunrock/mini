#include "coloring/coloring_enactor.hxx"
#include "test_utils.hxx"

#include <algorithm>
#include <cstdlib>

using namespace gunrock;
using namespace gunrock::coloring;

int main(int argc, char** argv) {

    // read in graph file
    std::string filename;
    CommandLineArgs args(argc, argv);
    args.GetCmdLineArgument("file", filename);

    // read in source node from cmd line
    int seed = 15485863;
    args.GetCmdLineArgument("seed", seed);
    int max_iter = 10;
    args.GetCmdLineArgument("max_iter", max_iter);

    cout<<"seed: " << seed << "max_iter: " << max_iter << std::endl;

    // CUDA context is used for all mgpu transforms
    standard_context_t context;
   
    // Load graph data to device
    std::shared_ptr<graph_t> graph = load_graph(filename.c_str());
    std::shared_ptr<graph_device_t> d_graph(std::make_shared<graph_device_t>());
    graph_to_device(d_graph, graph, context);

    // Initializes coloring problem object
    std::shared_ptr<coloring_problem_t> coloring_problem(std::make_shared<coloring_problem_t>(d_graph, seed, max_iter, context));

    std::shared_ptr<coloring_enactor_t> coloring_enactor(std::make_shared<coloring_enactor_t>(context, d_graph->num_nodes, d_graph->num_edges));

    test_timer_t timer;
    timer.start();
    coloring_enactor->enact(coloring_problem, context);
    cout << "elapsed time: " << timer.end() << "s." << std::endl;

    //coloring_problem->extract();
    display_device_data(coloring_problem.get()->d_colors.data(), coloring_problem.get()->gslice->num_nodes);
}



