#include "pr/pr_enactor.hxx"
#include "test_utils.hxx"

#include <algorithm>
#include <cstdlib>

using namespace gunrock;
using namespace gunrock::pr;

int main(int argc, char** argv) {

    // read in graph file
    std::string filename;
    CommandLineArgs args(argc, argv);
    args.GetCmdLineArgument("file", filename);

    // read in source node from cmd line
    int max_iter = 50;
    args.GetCmdLineArgument("max_iter", max_iter);

    // CUDA context is used for all mgpu transforms
    standard_context_t context;
   
    // Load graph data to device
    std::shared_ptr<graph_t> graph = load_graph(filename.c_str(), true);
    std::shared_ptr<graph_device_t> d_graph(std::make_shared<graph_device_t>());
    graph_to_device(d_graph, graph, context);

    // Initializes coloring problem object
    std::shared_ptr<pr_problem_t> pr_problem(std::make_shared<pr_problem_t>(d_graph, max_iter, context));

    std::shared_ptr<pr_enactor_t> pr_enactor(std::make_shared<pr_enactor_t>(context, d_graph->num_nodes, d_graph->num_edges));

    test_timer_t timer;
    timer.start();
    pr_enactor->enact(pr_problem, context);
    cout << "elapsed time: " << timer.end() << "s." << std::endl;

    //pr_problem->extract();
    //display_device_data(pr_problem.get()->d_colors.data(), pr_problem.get()->gslice->num_nodes);
}



