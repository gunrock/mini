#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_segreduce.hxx>
#include <moderngpu/kernel_reduce.hxx>
#include "frontier.hxx"
#include "intrinsics.hxx"

namespace gunrock {
namespace oprtr {
namespace neighborhood {

template<typename Problem, typename Functor, typename Value, typename reduce_op>
int neighborhood_kernel(std::shared_ptr<Problem> problem,
              std::shared_ptr<frontier_t<int> > &input,
              std::shared_ptr<frontier_t<int> > &output,
              Value  *reduced,
              Value identity,
              int iteration,
              standard_context_t &context)
{
    int *input_data = input.get()->data()->data();
    int *row_lengths = problem.get()->gslice->d_row_lengths.data();
    int *row_offsets = problem.get()->gslice->d_row_offsets.data();
    mem_t<int> count(1, context);

    auto segment_sizes = [=]__device__(int idx) {
        int count = 0;
        int v = input_data[idx];
        int begin = row_offsets[v];
        int end = row_offsets[v+1];
        count = end - begin;
        row_lengths[idx] = count;
        return count;
    };
    transform_reduce<int>(segment_sizes, (int)input.get()->size(), count.data(), plus_t<int>(), context);

    int non_zeros = from_mem(count)[0];
    if(!non_zeros) return 0;

    int *col_indices = problem.get()->gslice->d_col_indices.data();
    int *sources = problem.get()->gslice->d_csr_srcs.data();
    output->resize(non_zeros);
    int *output_data = output.get()->data()->data();
    typename Problem::data_slice_t *data = problem.get()->d_data_slice.data();
    
    auto neighborhood_reduce = [=]__device__(int idx) {
        int v = ldg(sources+idx);
        int neighbor = ldg(col_indices+idx);
        bool cond = Functor::cond_advance(v, neighbor, data, iteration);
        bool apply = Functor::apply_advance(v, neighbor, data, iteration);
        output_data[idx] = (cond && apply) ? neighbor : -1;
        return Functor::get_value_to_reduce(neighbor);
    };

    transform_segreduce(neighborhood_reduce, non_zeros, row_lengths, (int)input.get()->size(), reduced, reduce_op(), identity, context);

    // scatter reduced value to problem
    auto f = [=]__device__(int idx) {
        int item = input_data[idx];
        Value reduced_val = reduced[idx];
        Functor::write_reduced_value(item, reduced_val, data);
    };
    transform(f, (int)input.get()->size(), context);

    return non_zeros;
}

} // neighborhood
} // oprtr
} // gunrock
