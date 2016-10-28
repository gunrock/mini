#include "lspar/lspar_enactor.hxx"
#include "test_utils.hxx"

#include <algorithm>
#include <cstdlib>

using namespace gunrock;
using namespace gunrock::lspar;

int main(int argc, char** argv) {

    // read in graph file
    std::string filename;
    CommandLineArgs args(argc, argv);
    args.GetCmdLineArgument("file", filename);

    int prime = 15485863;
    args.GetCmdLineArgument("prime", prime);
    int k = 1;
    float e = 0.5f;
    args.GetCmdLineArgument("e", e);

    // CUDA context is used for all mgpu transforms
    standard_context_t context;
   
    // Load graph data to device
    std::shared_ptr<graph_t> graph = load_graph(filename.c_str(), true);
    std::shared_ptr<graph_device_t> d_graph(std::make_shared<graph_device_t>());
    graph_to_device(d_graph, graph, context);

    // Initializes lspar problem object
    std::shared_ptr<lspar_problem_t> lspar_problem(std::make_shared<lspar_problem_t>(d_graph, prime, k, e, context));
    std::shared_ptr<lspar_enactor_t> lspar_enactor(std::make_shared<lspar_enactor_t>(context, d_graph->num_nodes, d_graph->num_edges));

    test_timer_t timer;
    timer.start();
    int selected_edge_num = lspar_enactor->enact(lspar_problem, context);
    cout << "elapsed time: " << timer.end() << "s." << std::endl;
    cout << "selected " << selected_edge_num << " edges." << std::endl;

    //lspar_problem->extract();
    //display_device_data(lspar_problem.get()->d_thresholds.data(), lspar_problem.get()->gslice->num_nodes);
    //display_device_data(lspar_problem.get()->d_hashs.data(), lspar_problem.get()->gslice->num_nodes);
    //display_device_data(lspar_problem.get()->d_minwise_hashs.data(), lspar_problem.get()->gslice->num_nodes);
}



