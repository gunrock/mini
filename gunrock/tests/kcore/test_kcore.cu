#include "kcore/kcore_enactor.hxx"
#include "test_utils.hxx"

#include <algorithm>
#include <cstdlib>

using namespace gunrock;
using namespace gunrock::kcore;

int main(int argc, char** argv) {

    // read in graph file
    std::string filename;
    CommandLineArgs args(argc, argv);
    args.GetCmdLineArgument("file", filename);

    // CUDA context is used for all mgpu transforms
    standard_context_t context;

    // Load graph data to device
    std::shared_ptr<graph_t> graph = load_graph(filename.c_str(),/*_undir=*/true,/*_random_edge_value=*/false);
    std::shared_ptr<graph_device_t> d_graph(std::make_shared<graph_device_t>());
    graph_to_device(d_graph, graph, context);

    // Initializes kcore problem object
    std::shared_ptr<kcore_problem_t> kcore_problem(std::make_shared<kcore_problem_t>(d_graph, context));

    std::shared_ptr<kcore_enactor_t> kcore_enactor(std::make_shared<kcore_enactor_t>(context, d_graph->num_nodes, d_graph->num_edges));

    test_timer_t timer;
    timer.start();
    kcore_enactor->enact(kcore_problem, context);
    cout << "elapsed time: " << timer.end() << "s." << std::endl;

    std::vector<int> validation_num_cores = std::vector<int>(d_graph->num_nodes, 0);
    kcore_problem->extract();
    int ref_largest_k_core = kcore_problem->cpu(validation_num_cores, graph->csr->offsets, graph->csr->indices);

    if (ref_largest_k_core != kcore_problem.get()->largest_k_core) {
        cout << "Validation Error for largest k-core. ref: " << ref_largest_k_core << " gpu: " << kcore_problem.get()->largest_k_core << endl;
    }
    //display_device_data(kcore_problem.get()->d_num_cores.data(), kcore_problem.get()->gslice->num_nodes);

    if (!validate(kcore_problem.get()->num_cores, validation_num_cores))
        cout << "Validation Error." << endl;
    else
        cout << "Correct." << endl;
}
