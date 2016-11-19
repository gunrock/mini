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

    float queue_sizing=1.0f;
    args.GetCmdLineArgument("queue-sizing", queue_sizing);

    bool undirected = args.CheckCmdLineFlag("undirected");

    // CUDA context is used for all mgpu transforms
    standard_context_t context;
   
    // Load graph data to device
    std::shared_ptr<graph_t> graph = load_graph(filename.c_str(), undirected, false);
    std::shared_ptr<graph_device_t> d_graph(std::make_shared<graph_device_t>());
    graph_to_device(d_graph, graph, context);

    // Initializes sssp problem object
    std::shared_ptr<sssp_problem_t> sssp_problem(std::make_shared<sssp_problem_t>(d_graph, src, context));

    std::shared_ptr<sssp_enactor_t> sssp_enactor(std::make_shared<sssp_enactor_t>(context, d_graph->num_nodes, d_graph->num_edges, queue_sizing));

    test_timer_t timer;
    timer.start();
    sssp_enactor->enact(sssp_problem, context);
    cout << "elapsed time: " << timer.end() << "s." << std::endl;

    std::vector<int> validation_preds = std::vector<int>(d_graph->num_nodes, -1);
    sssp_problem->extract();
    sssp_problem->cpu(validation_preds, graph->csr->offsets, graph->csr->indices, graph->csr->edge_weights);

    if (!validate(sssp_problem.get()->preds, validation_preds))
        cout << "Validation Error." << endl;
    else
        cout << "Correct," << endl;

    //display_device_data(sssp_problem.get()->d_preds.data(), sssp_problem.get()->gslice->num_nodes);
    
    //int lala = 0;
    //for (int p : validation_preds)
        //std::cout << lala++ <<":"<<p << " ";
    //std::cout << std::endl;
}



