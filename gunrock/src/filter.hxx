#include <moderngpu/kernel_compact.hxx>
#include "frontier.hxx"
#include "test_utils.hxx"

namespace gunrock {
namespace oprtr {
namespace filter {

template<typename InputFrontierType, typename OutputFrontierType, typename Problem, typename Functor>
void launch_kernel(std::shared_ptr<Problem> problem,
              std::shared_ptr<frontier_t<InputFrontierType> > input,
              std::shared_ptr<frontier_t<OutputFrontierType> > output,
              standard_context_t &context)
{
    auto compact = transform_compact(input.get()->size(), context);
    InputFrontierType *input_data = input.get()->data()->data();
    typename Problem::data_slice_t *data = problem.get()->data_slice;
    int stream_count = compact.upsweep([=]__device__(int idx) {
                int item = input_data[idx];
                return Functor::cond_filter(item, data);
            });
    output.reset( new frontier_t<OutputFrontierType>(context, stream_count, 1, input.get()->type()) );
    OutputFrontierType *output_data = output.get()->data()->data();
    compact.downsweep([=]__device__(int dest_idx, int source_idx) {
            output_data[dest_idx] = input_data[source_idx];
            Functor::apply_filter(output_data[dest_idx], data);
        });
}

} // end filter
} // end oprtr
} // end gunrock


