#include <moderngpu/kernel_compact.hxx>
#include "frontier.hxx"

namespace gunrock {
namespace oprtr {
namespace filter {

// filter kernel using transform_compact with full uniquification
// (remove all the failed condition items)
//
template<typename Problem, typename Functor>
void filter_kernel(std::shared_ptr<Problem> problem,
              std::shared_ptr<frontier_t<int> > &input,
              std::shared_ptr<frontier_t<int> > &output,
              int iteration,
              standard_context_t &context)
{
    auto compact = transform_compact(input.get()->size(), context);
    int *input_data = input.get()->data()->data();
    typename Problem::data_slice_t *data = problem.get()->d_data_slice.data();
    int stream_count = compact.upsweep([=]__device__(int idx) {
                int item = input_data[idx];
                return Functor::cond_filter(item, data, iteration);
            });
    //output.reset( new frontier_t<int>(context, input.get()->capacity(), stream_count, input.get()->type()) );
    output->resize(stream_count);
    int *output_data = output.get()->data()->data();
    compact.downsweep([=]__device__(int dest_idx, int source_idx) {
            output_data[dest_idx] = input_data[source_idx];
            Functor::apply_filter(output_data[dest_idx], data, iteration);
        });
}

struct UniquifyFunctor {
    static __device__ __forceinline__ void bitmask_cull(int idx, int *input_data, unsigned char *d_visited_mask) {
        int item = input_data[idx];
        int mask_byte_offset = item >> 3;
        unsigned char mask_bit = 1 << item & 7;
        unsigned char mask_byte = d_visited_mask[mask_byte_offset];
        if (mask_bit & mask_byte) {
            // seen it
            input_data[idx] = -1;
        } else {
            unsigned char new_mask_byte = d_visited_mask[mask_byte_offset];
            new_mask_byte |= mask_byte;
            if (mask_bit & new_mask_byte) {
                // seen it
                input_data[idx] = -1;
            } else {
                new_mask_byte |= mask_bit;
                d_visited_mask[mask_byte_offset] = new_mask_byte;
            }
        }
    }

    //static __device__ __forceinline__ bool history_cull(int idx, 
};

//TODO: END_BITMASK_CULL to control when to stop
//add history warp cull
template<typename Problem, typename ProblemFunctor>
void uniquify_kernel(std::shared_ptr<Problem> problem,
                unsigned char *d_visited_mask,
                std::shared_ptr<frontier_t<int> > &input,
                std::shared_ptr<frontier_t<int> > &output,
                int iteration,
                standard_context_t &context)
{
    auto compact = transform_compact(input.get()->size(), context);
    int *input_data = input.get()->data()->data();
    typename Problem::data_slice_t *data = problem.get()->d_data_slice.data();
    int stream_count = compact.upsweep([=]__device__(int idx) {
                UniquifyFunctor::bitmask_cull(idx, input_data, d_visited_mask);
                return ProblemFunctor::cond_uniq(input_data[idx], data, iteration);
            });
    //output.reset( new frontier_t<int>(context, input.get()->capacity(), stream_count, input.get()->type()) );
    output->resize(stream_count);
    int *output_data = output.get()->data()->data();
    compact.downsweep([=]__device__(int dest_idx, int source_idx) {
            output_data[dest_idx] = input_data[source_idx];
            ProblemFunctor::apply_uniq(output_data[dest_idx], data, iteration);
        });
}

} // end filter
} // end oprtr
} // end gunrock


