#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_load_balance.hxx>
#include "frontier.hxx"

namespace gunrock {
namespace oprtr {
namespace advance {

    //first scan
    //then lbs (given the option to idempotence or not)
template<typename Problem, typename Functor>
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
        output_data[idx] = Functor::cond_advance(v, neighbor, data, iteration) ? neighbor : -1;
    };
    transform_lbs(neighbors_expand, front, scanned_row_offsets, (int)input.get()->size(), context);

    return front;
}

} // end advance
} // end  oprtr
} // end gunrock
