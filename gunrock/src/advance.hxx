#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_load_balance.hxx>
#include "frontier.hxx"

namespace gunrock {
namespace oprtr {
namespace advance {

    //first scan
    //then lbs (given the option to idempotence or not)
template<typename Problem, typename Functor, bool idempotence>
int advance_kernel(std::shared_ptr<Problem> problem,
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
    if(!front) return 0;

    int *col_indices = problem.get()->gslice->d_col_indices.data();
    //output.reset( new frontier_t<int>(context, input.get()->capacity(), front, input.get()->type()) );
    output->resize(front);
    int *output_data = output.get()->data()->data();
    typename Problem::data_slice_t *data = problem.get()->d_data_slice.data();
    auto neighbors_expand = [=]__device__(int idx, int seg, int rank) {
        int v = input_data[seg];
        int start_idx = row_offsets[v];
        int neighbor = col_indices[start_idx+rank];
        bool apply = Functor::cond_advance(v, neighbor, data, iteration);
        output_data[idx] = idempotence ? neighbor : (apply ? neighbor : -1);
        if (apply) Functor::apply_advance(v, neighbor, data, iteration);
    };
    transform_lbs(neighbors_expand, front, scanned_row_offsets, (int)input.get()->size(), context);

    return front;
}

template<typename Problem, typename Functor, bool early_exit>
int advance_reverse_kernel(std::shared_ptr<Problem> problem,
              std::shared_ptr<frontier_t<int> > &unvisited,
              std::shared_ptr<frontier_t<int> > &bitmap,
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
    // update the bitmaps of visited nodes, compact the unvisited nodes
    typename Problem::data_slice_t *data = problem.get()->d_data_slice.data();
    auto neighbors_expand = [=]__device__(int idx, int seg, int rank) {
        int v = unvisited_data[seg];
        if (early_exit && v == -1) {
            // visited and early exit,
            return;
        }
        int start_idx = col_offsets[v];
        int neighbor = row_indices[start_idx+rank];
        // if the neighbor item in bitmap is set,
        // set unvisited_data[seg] in bitmap too.
        // if early_exit, mark unvisited_data[seg] as -1
        int *bitmap_data = bitmap.get()->data()->data();
        if (bitmap_data[neighbor]) {
            bitmap_data[unvisited_data[seg]] = 1;
            if (early_exit)
                unvisited_data[seg] = -1;
        }
        if (Functor::cond_advance(v, neighbor, data, iteration))
            Functor::apply_advance(v, neighbor, data, iteration);
    };

    transform_lbs(neighbors_expand, front, scanned_row_offsets, (int)unvisited.get()->size(), context);

    return front;
}

} // end advance
} // end  oprtr
} // end gunrock
