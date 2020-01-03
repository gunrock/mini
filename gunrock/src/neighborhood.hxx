#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_segreduce.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_intervalmove.hxx>
#include "frontier.hxx"
#include "intrinsics.hxx" 

namespace gunrock {
namespace oprtr {
namespace neighborhood {

template<typename Problem, typename Functor, typename Value, typename reduce_op, bool has_output, bool push>
int neighborhood_kernel(std::shared_ptr<Problem> problem,
              std::shared_ptr<frontier_t<int> > &input,
              std::shared_ptr<frontier_t<int> > &output,
              Value  *reduced,
              Value identity,
              int iteration,
              standard_context_t &context)
{
    int *input_data = input.get()->data()->data();
    int *scanned_offsets = problem.get()->gslice->d_scanned_row_offsets.data();
    int *offsets = (push) ? problem.get()->gslice->d_row_offsets.data():
                            problem.get()->gslice->d_col_offsets.data();
    mem_t<int> count(1, context);

    auto segment_sizes = [=]__device__(int idx) {
        int count = 0;
        int v = ldg(input_data+idx);
        int begin = ldg(offsets+v);
        int end = ldg(offsets+v+1);
        count = end - begin;
        return count;
    };
    transform_scan<int>(segment_sizes, (int)input.get()->size(), scanned_offsets, plus_t<int>(), count.data(), context);

    int non_zeros = from_mem(count)[0];
    if(!non_zeros) return 0;


    int *col_indices = (push) ? problem.get()->gslice->d_col_indices.data():
                                problem.get()->gslice->d_row_indices.data();
    if (has_output) output->resize(non_zeros);
    int *output_data = (has_output) ? output.get()->data()->data() : nullptr;
    typename Problem::data_slice_t *data = problem.get()->d_data_slice.data();
    
    auto neighborhood_reduce = [=]__device__(int idx,int seg, int rank) {
        int v = ldg(input_data+seg);
        int start = ldg(offsets+v);
        int neighbor = ldg(col_indices+start+rank);
        bool cond = Functor::cond_advance(v, neighbor, start+rank, rank, idx, data, iteration);
        bool apply = Functor::apply_advance(v, neighbor, start+rank, rank, idx, data, iteration);
        if (has_output) output_data[idx] = (cond && apply) ? neighbor : -1;
        auto val = Functor::get_value_to_reduce(neighbor, data, iteration);
        return val;
    };

    lbs_segreduce(neighborhood_reduce, non_zeros, scanned_offsets, (int)input.get()->size(), reduced, reduce_op(), identity, context);

    // scatter reduced value to problem
    // skip for now, since reduced array has been sent in
    /*auto f = [=]__device__(int idx) {
        int item = input_data[idx];
        Value reduced_val = reduced[idx];
        Functor::write_reduced_value(item, reduced_val, data, iteration);
    };
    transform(f, (int)input.get()->size(), context);*/

    return non_zeros;
}

} // neighborhood
} // oprtr
} // gunrock
