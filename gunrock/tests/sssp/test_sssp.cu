#include "sssp/sssp_enactor.hxx"
#include "test_utils.hxx"

#include <algorithm>
#include <cstdlib>

using namespace gunrock;
using namespace gunrock::sssp;

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

    // Initializes sssp problem object
    std::shared_ptr<sssp_problem_t> sssp_problem(std::make_shared<sssp_problem_t>(d_graph, src, context));

    std::shared_ptr<sssp_enactor_t> sssp_enactor(std::make_shared<sssp_enactor_t>(context, d_graph->num_nodes, d_graph->num_edges));

    test_timer_t timer;
    timer.start();
    sssp_enactor->enact(sssp_problem, context);
    cout << "elapsed time: " << timer.end() << "s." << std::endl;

    sssp_problem->extract();
    //display_device_data(sssp_problem.get()->d_labels.data(), sssp_problem.get()->gslice->num_nodes);
    //display_device_data(sssp_problem.get()->d_preds.data(), sssp_problem.get()->gslice->num_nodes);
}



