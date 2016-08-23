#pragma once
#include <vector>
#include <memory>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include <tuple>
#include <iostream>

#include "moderngpu/meta.hxx"
#include "moderngpu/context.hxx"
#include "moderngpu/memory.hxx"

using namespace mgpu;

namespace gunrock {

struct csr_t {
  int num_nodes;
  int num_edges;
  std::vector<int> offsets;
  std::vector<int> indices;
  std::vector<float> edge_weights;
};

struct edgelist_t {
  int num_edges;
  std::vector<std::tuple<int, int, float>> edges;
};

struct graph_t {
  int num_nodes;
  int num_edges;

  std::shared_ptr<csr_t> csr;
  std::shared_ptr<csr_t> csc;
  std::shared_ptr<edgelist_t> edgelist;
};

struct graph_device_t {
  int num_nodes;
  int num_edges;
  mem_t<int> d_row_offsets;
  mem_t<int> d_col_indices;
  mem_t<float> d_col_values;
  mem_t<int> d_col_offsets;
  mem_t<int> d_row_indices;
  mem_t<float> d_row_values;
  
  // update during each advance iteration
  // to store the scaned row offsets of
  // the current frontier
  mem_t<int> d_scanned_row_offsets;
  // row lengths for each node in frontier
  // coarse fine row offsets array is used for dynamic grouping advance
  // int2.x:coarse rank, int2.y:fine rank
  mem_t<int> d_row_lengths;
  mem_t<int2> d_scanned_coarse_fine_row_offsets;

  graph_device_t() :
      num_nodes(0),
      num_edges(0)
      {}
};

void graph_to_device(std::shared_ptr<graph_device_t> d_graph, std::shared_ptr<graph_t> graph,
                     standard_context_t &context) {
  d_graph->num_nodes = graph->num_nodes;
  d_graph->num_edges = graph->num_edges;
  d_graph->d_row_offsets = to_mem(graph->csr->offsets, context);
  d_graph->d_col_indices = to_mem(graph->csr->indices, context);
  d_graph->d_col_values = to_mem(graph->csr->edge_weights, context);
  d_graph->d_col_offsets = to_mem(graph->csc->offsets, context);
  d_graph->d_row_indices = to_mem(graph->csc->indices, context);
  d_graph->d_row_values = to_mem(graph->csc->edge_weights, context);
  d_graph->d_scanned_row_offsets = mem_t<int>(graph->num_nodes, context);
  // TODO: should really allocate when strategy is dynamic grouping.
  d_graph->d_row_lengths = mem_t<int>(graph->num_nodes, context);
  d_graph->d_scanned_coarse_fine_row_offsets = mem_t<int2>(graph->num_nodes, context);
}

void display_csr(std::shared_ptr<csr_t> csr) {
    std::cout << "offsets: \n";
    for (auto item = csr->offsets.begin(); item != csr->offsets.end(); ++item)
        std::cout << *item << ' ';
    std::cout << std::endl;
    std::cout << "indices: \n";
    for (auto item = csr->indices.begin(); item != csr->indices.end(); ++item)
        std::cout << *item << ' ';
    std::cout << std::endl;
}

std::shared_ptr<graph_t> load_graph(const char *_name, bool _undir = false,
                                    bool _random_edge_value = false) {
  FILE *f = fopen(_name, "r");
  if (!f)
    return false;

  char line[100];
  while (fgets(line, 100, f) && '%' == line[0])
    ;

  int height, width, num_edges;
  if (3 != sscanf(line, "%d %d %d", &height, &width, &num_edges)) {
    printf("Error reading %s\n", _name);
    exit(0);
  }

  int num_vertices = height;

  int edge_size = _undir ? 2 * num_edges : num_edges;
  std::vector<std::tuple<int, int, float>> tuples(edge_size);
  for (int edge = 0; edge < edge_size; ++edge) {
    std::tuple<int, int, float> tuple;
    int num_item;
    if (!fgets(line, 100, f) ||
        2 > (num_item = sscanf(line, "%d %d %f", &std::get<0>(tuple),
                               &std::get<1>(tuple), &std::get<2>(tuple)))) {
      printf("Error reading edge lists %s\n", _name);
      exit(0);
    }
    if (num_item == 2) {
      std::get<2>(tuple) = _random_edge_value ? rand() % 64 : 1.0f;
    }
    --(std::get<0>(tuple)), --(std::get<1>(tuple));
    tuples[edge] = tuple;
    if (_undir) {
      std::swap(std::get<0>(tuple), std::get<1>(tuple));
      tuples[edge + num_edges] = tuple;
    }
  }
  if (_undir) {
    num_edges *= 2;
  }

  sort(tuples.begin(), tuples.begin(),
       [](const std::tuple<int, int, float> &a,
          const std::tuple<int, int, float> &b) -> bool {
         int first = std::get<1>(a);
         int second = std::get<1>(b);
         if (first > second) {
           return true;
         } else if (first < second) {
           return false;
         }
         first = std::get<0>(a);
         second = std::get<0>(b);
         if (first > second) {
           return true;
         } else if (first < second) {
           return false;
         }
         return true;
       });

  // Build csc
  std::vector<int> col_offsets(num_vertices + 1, num_edges);
  std::vector<int> row_indices(num_edges);
  std::vector<float> row_values(num_edges);
  int cur_vertex = -1;
  for (int edge = 0; edge < num_edges; ++edge) {
    while (cur_vertex < std::get<0>(tuples[edge])) {
      col_offsets[++cur_vertex] = edge;
    }
    row_indices[edge] = std::get<1>(tuples[edge]);
    row_values[edge] = std::get<1>(tuples[edge]);
  }

  sort(tuples.begin(), tuples.begin(),
       [](const std::tuple<int, int, float> &a,
          const std::tuple<int, int, float> &b) -> bool {
         int first = std::get<0>(a);
         int second = std::get<0>(b);
         if (first > second) {
           return true;
         } else if (first < second) {
           return false;
         }
         first = std::get<1>(a);
         second = std::get<1>(b);
         if (first > second) {
           return true;
         } else if (first < second) {
           return false;
         }
         return true;
       });

  // Build csr
  std::vector<int> row_offsets(num_vertices + 1, num_edges);
  std::vector<int> col_indices(num_edges);
  std::vector<float> col_values(num_edges);
  cur_vertex = -1;
  for (int edge = 0; edge < num_edges; ++edge) {
    while (cur_vertex < std::get<1>(tuples[edge])) {
      row_offsets[++cur_vertex] = edge;
    }
    col_indices[edge] = std::get<0>(tuples[edge]);
    col_values[edge] = std::get<0>(tuples[edge]);
  }

  // create unique_ptr of csr, csc, and edgelist
  std::shared_ptr<csr_t> csr_ptr(
      new csr_t{num_vertices, num_edges, row_offsets, col_indices, col_values});
  std::shared_ptr<csr_t> csc_ptr(
      new csr_t{num_vertices, num_edges, col_offsets, row_indices, row_values});
  std::shared_ptr<edgelist_t> edgelist_ptr(new edgelist_t{num_edges, tuples});

  // return graph_t
  return std::shared_ptr<graph_t>(
      new graph_t{num_vertices, num_edges, csr_ptr, csc_ptr, edgelist_ptr});
}

}
