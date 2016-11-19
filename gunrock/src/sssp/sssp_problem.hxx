#pragma once

#include "problem.hxx"
#include <queue>
#include <utility>

namespace gunrock {
namespace sssp {

struct sssp_problem_t : problem_t {
  mem_t<float> d_labels;
  mem_t<int> d_preds;
  mem_t<int> d_visited;
  std::vector<float> labels;
  std::vector<int> preds;
  int src;

  struct data_slice_t {
      float *d_labels;
      int *d_preds;
      float *d_weights;
      int *d_visited;

      void init(mem_t<float> &_labels, mem_t<int> &_preds, mem_t<float> &_weights, mem_t<int> &_visited) {
        d_labels = _labels.data();
        d_preds = _preds.data();
        d_weights = _weights.data();
        d_visited = _visited.data();
      }
  };

  mem_t<data_slice_t> d_data_slice;
  std::vector<data_slice_t> data_slice;
  
  sssp_problem_t() {}

  sssp_problem_t(const sssp_problem_t& rhs) = delete;
  sssp_problem_t& operator=(const sssp_problem_t& rhs) = delete;

  sssp_problem_t(std::shared_ptr<graph_device_t> rhs, size_t src, standard_context_t& context) :
      problem_t(rhs),
      src(src),
      data_slice( std::vector<data_slice_t>(1) ) {
          labels = std::vector<float>(rhs->num_nodes, std::numeric_limits<float>::max());
          preds = std::vector<int>(rhs->num_nodes, -1);
          labels[src] = 0;
          d_labels = to_mem(labels, context);
          d_preds = to_mem(preds, context);
          d_visited = fill(-1, rhs->num_nodes, context);
          data_slice[0].init(d_labels, d_preds, gslice->d_col_values, d_visited);
          d_data_slice = to_mem(data_slice, context);
      }

  void extract() {
      mgpu::dtoh(labels, d_labels.data(), gslice->num_nodes); 
      mgpu::dtoh(preds, d_preds.data(), gslice->num_nodes); 
  }

  void cpu(std::vector<int> &validation_preds,
                  std::vector<int> &row_offsets,
                  std::vector<int> &col_indices,
                  std::vector<float> &col_values) {
      // CPU SSSP for validation (Dijkstra's algorithm)
      typedef std::pair<int, int> ipair;

      // initialize input frontier
      std::priority_queue<ipair, std::vector<ipair>, std::greater<ipair> > pq;
      std::vector<int> dist(row_offsets.size(), std::numeric_limits<int>::max());
      pq.push(std::make_pair(-1, src));
      validation_preds[src] = -1;
      dist[src] = 0;
      while(!pq.empty())
      {
          int u = pq.top().second;
          pq.pop();

          for (int i = row_offsets[u]; i < row_offsets[u+1]; ++i) {
              int v = col_indices[i];
              int w = col_values[i];
              if (dist[v] > dist[u] + w) {
                  validation_preds[v] = u;
                  dist[v] = dist[u] + w;
                  pq.push(std::make_pair(dist[u], v));
              }
          }
      }

  }
};

} //end sssp
} //end gunrock
