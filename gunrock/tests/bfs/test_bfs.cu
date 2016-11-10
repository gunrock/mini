#include "bfs/bfs_enactor.hxx"
#include "test_utils.hxx"

#include <algorithm>
#include <cstdlib>

using namespace gunrock;
using namespace gunrock::bfs;

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
    std::shared_ptr<graph_t> graph = load_graph(filename.c_str(),true,false);
    std::shared_ptr<graph_device_t> d_graph(std::make_shared<graph_device_t>());
    graph_to_device(d_graph, graph, context); 

    //float alpha = d_graph->num_nodes; // set default to prevent push-pull switch to happen
    float alpha = 1.0f/d_graph->num_nodes; // set default to prevent push-pull switch to happen
    args.GetCmdLineArgument("alpha", alpha);

    // Initializes bfs problem object
    std::shared_ptr<bfs_problem_t> bfs_problem(std::make_shared<bfs_problem_t>(d_graph, src, context));

    std::shared_ptr<bfs_enactor_t> bfs_enactor(std::make_shared<bfs_enactor_t>(context, d_graph->num_nodes, d_graph->num_edges));

    test_timer_t timer;
    timer.start();
    bfs_enactor->enact_baseline(bfs_problem, context);
    //bfs_enactor->enact_pushpull(bfs_problem, alpha, context);
    cout << "elapsed time: " << timer.end() << "s." << std::endl;

    std::vector<int> validation_labels = std::vector<int>(d_graph->num_nodes, -1);
    bfs_problem->extract();
    bfs_problem->cpu(validation_labels, graph->csr->offsets, graph->csr->indices);
    //display_device_data(bfs_problem.get()->d_labels.data(), bfs_problem.get()->gslice->num_nodes);

    if (!validate(bfs_problem.get()->labels, validation_labels))
        cout << "Validation Error." << endl;
    else
        cout << "Correct." << endl;
}



