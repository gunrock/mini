#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_load_balance.hxx>
#include "frontier.hxx"
#include "intrinsics.hxx"

template<>
struct plus_t<int2> : public std::binary_function<int2, int2, int2> {
	__forceinline__ __host__ __device__ int2 operator()(int2 a, int2 b) const {
    return make_int2(a.x+b.x, a.y+b.y);
  }
};

namespace gunrock {
namespace oprtr {
namespace advance {

    //first scan
    //then lbs (given the option to idempotence or not)
template<typename Problem, typename Functor, bool idempotence, bool has_output>
int advance_forward_kernel(std::shared_ptr<Problem> problem,
              std::shared_ptr<frontier_t<int> > &input,
              std::shared_ptr<frontier_t<int> > &output,
              int iteration,
              standard_context_t &context)
{
    int *input_data = input.get()->data()->data();
    int *scanned_row_offsets = problem.get()->gslice->d_scanned_row_offsets.data();
    int *row_offsets = problem.get()->gslice->d_row_offsets.data();
    mem_t<int> count(1, context);

    auto segment_sizes = [=]__device__(int idx) {
        int count = 0;
        int v = input_data[idx];
        int begin = row_offsets[v];
        int end = row_offsets[v+1];
        count = end - begin;
        return count;
    };
    transform_scan<int>(segment_sizes, (int)input.get()->size(), scanned_row_offsets, plus_t<int>(),
            count.data(), context);

    int front = from_mem(count)[0];
    if(!front) {
        if (has_output) output->resize(front);
        return 0;
    }

    int *col_indices = problem.get()->gslice->d_col_indices.data();
    if (has_output) output->resize(front);
    int *output_data = has_output? output.get()->data()->data() : nullptr;
    typename Problem::data_slice_t *data = problem.get()->d_data_slice.data();
    auto neighbors_expand = [=]__device__(int idx, int seg, int rank) {
        int v = input_data[seg];
        int start_idx = row_offsets[v];
        int neighbor = col_indices[start_idx+rank];
        bool cond = Functor::cond_advance(v, neighbor, start_idx+rank, rank, idx, data, iteration); 
        if (has_output)
            output_data[idx] = idempotence ? neighbor : ((cond && Functor::apply_advance(v, neighbor, start_idx+rank, rank, idx, data, iteration)) ? neighbor : -1);
    };
    transform_lbs(neighbors_expand, front, scanned_row_offsets, (int)input.get()->size(), context);

    if(!has_output) front = 0;

    return front;
}

template<typename Problem, typename Functor>
void sparse_to_dense_kernel(std::shared_ptr<Problem> problem,
                       std::shared_ptr<frontier_t<int> > &sparse, //compact representation
                       std::shared_ptr<frontier_t<int> > &dense, //bitmap representation
                       int iteration,
                       standard_context_t &context)
{
    int *input_data = sparse.get()->data()->data();
    int *output_data = dense.get()->data()->data();
    typename Problem::data_slice_t *data = problem.get()->d_data_slice.data();
    auto f = [=]__device__(int idx) {
        int item = input_data[idx];
        output_data[item] = Functor::cond_sparse_to_dense(item, data, iteration) ? 1 : 0;
    };
    transform(f, (int)sparse.get()->size(), context);
}

template<typename Problem, typename Functor>
int gen_unvisited_kernel(std::shared_ptr<Problem> problem,
                       std::shared_ptr<frontier_t<int> > &indices,
                       std::shared_ptr<frontier_t<int> > &unvisited,
                       int iteration,
                       standard_context_t &context)
{
    auto compact = transform_compact(indices.get()->size(), context);
    int *input_data = indices.get()->data()->data();
    typename Problem::data_slice_t *data = problem.get()->d_data_slice.data();
    int stream_count = compact.upsweep([=]__device__(int idx) {
                int item = input_data[idx];
                return Functor::cond_gen_unvisited(item, data, iteration);
                });
    unvisited->resize(stream_count);
    int *unvisited_data = unvisited.get()->data()->data();
    compact.downsweep([=]__device__(int dest_idx, int source_idx) {
            unvisited_data[dest_idx] = input_data[source_idx];
        });
    return stream_count;
}

template<typename Problem, typename Functor>
int advance_backward_kernel(std::shared_ptr<Problem> problem,
              std::shared_ptr<frontier_t<int> > &unvisited,
              std::shared_ptr<frontier_t<int> > &bitmap,
              std::shared_ptr<frontier_t<int> > &bitmap_out,
              int iteration,
              standard_context_t &context)
{
    int *unvisited_data = unvisited.get()->data()->data();
    int *scanned_row_offsets = problem.get()->gslice->d_scanned_row_offsets.data();
    int *col_offsets = problem.get()->gslice->d_col_offsets.data();
    mem_t<int> count(1, context);

    auto segment_sizes = [=]__device__(int idx) {
        int count = 0;
        int v = unvisited_data[idx];
        int begin = col_offsets[v];
        int end = col_offsets[v+1];
        count = end - begin;
        return count;
    };
    transform_scan<int>(segment_sizes, (int)unvisited.get()->size(), scanned_row_offsets, plus_t<int>(),
            count.data(), context);

    int front = from_mem(count)[0];
    if(!front) return 0;

    int *row_indices = problem.get()->gslice->d_row_indices.data();
    int num_nodes = problem.get()->gslice->num_nodes;
    int num_edges = problem.get()->gslice->num_edges;
    // update the bitmaps of visited nodes, compact the unvisited nodes
    typename Problem::data_slice_t *data = problem.get()->d_data_slice.data();
    int *bitmap_data = bitmap.get()->data()->data();
    int *bitmap_out_data = bitmap_out.get()->data()->data();
    auto neighbors_expand = [=]__device__(int idx, int seg, int rank) {
        int v = unvisited_data[seg];
        int start_idx = col_offsets[v];
        int neighbor = row_indices[start_idx+rank];
        // if the neighbor item in bitmap is set,
        // set unvisited_data[seg] in bitmap too.
        // if early_exit, mark unvisited_data[seg] as -1
        if (bitmap_data[neighbor] && Functor::apply_advance(neighbor, v, start_idx+rank, rank, idx, data, iteration)) {
            bitmap_out_data[v] = 1;
            unvisited_data[seg] = -1;
        }
        if (!Functor::cond_advance(neighbor, v, start_idx+rank, rank, idx, data, iteration))
            return;
    };

    transform_lbs(neighbors_expand, front, scanned_row_offsets, (int)unvisited.get()->size(), context);

    return front;
}

} // end advance
} // end  oprtr
} // end gunrock
