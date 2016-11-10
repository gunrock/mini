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
    if(!front) return 0;

    int *col_indices = problem.get()->gslice->d_col_indices.data();
    if (has_output) output->resize(front);
    int *output_data = has_output? output.get()->data()->data() : nullptr;
    typename Problem::data_slice_t *data = problem.get()->d_data_slice.data();
    auto neighbors_expand = [=]__device__(int idx, int seg, int rank) {
        int v = input_data[seg];
        int start_idx = row_offsets[v];
        int neighbor = col_indices[start_idx+rank];
        bool cond = Functor::cond_advance(v, neighbor, start_idx+rank, rank, idx, data, iteration);
        bool apply = Functor::apply_advance(v, neighbor, start_idx+rank, rank, idx, data, iteration);
        if (has_output)
            output_data[idx] = idempotence ? neighbor : ((cond && apply) ? neighbor : -1);
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

template<typename Problem, typename Functor>
struct AdvanceTWCFunctor {
    static __device__ __forceinline__ void expand_cta(
            int idx,
            int input_size,
            int num_edges,
            typename Problem::data_slice_t *data,
            const int *input_data,
            int *output_data,
            int *row_lengths,
            const int *row_offsets,
            const int *col_indices,
            const int2 *row_ranks,
            int cta_gather_threshold,
            int (&warp_comm)[mgpu::warp_size][4],
            int (&cta_comm)[1],
            int nt,
            int iteration,
            bool idempotence) {
        if (idx >= num_edges) return;
        cta_comm[0] = -1;
        while (true) {
            if (idx < input_size && row_lengths[idx] >= cta_gather_threshold) {
                cta_comm[0] = threadIdx.x;
            }
            __syncthreads();
            //printf("blockIdx:%d threadIdx: %d idx:%d, row_len:%d, cta_comm:%d\n",blockIdx.x, threadIdx.x, idx, row_lengths[idx], cta_comm);

            // check
            int owner = cta_comm[0];
            if (owner == nt) {
                break;
            }
            __syncthreads();
            //printf("blockIdx:%d threadIdx: %d idx:%d, row_len:%d, owner:%d\n",blockIdx.x, threadIdx.x, idx, row_lengths[idx], owner);
            break;

            if (owner == threadIdx.x) { // got control of the CTA, command it
                //printf("blockIdx:%d threadIdx: %d idx:%d, row_len:%d\n",blockIdx.x, threadIdx.x, idx, row_lengths[idx]);
                // start
                warp_comm[0][0] = row_offsets[input_data[idx]];
                // queue rank
                warp_comm[0][1] = row_ranks[idx].x;
                // oob
                warp_comm[0][2] = row_offsets[input_data[idx]] + row_lengths[idx];
                // pred
                warp_comm[0][3] = input_data[idx];

                row_lengths[idx] = 0;
                cta_comm[0] = nt;
            }
            __syncthreads();

            // Read commands
            int coop_offset = warp_comm[0][0];
            int coop_rank = warp_comm[0][1] + threadIdx.x;
            int coop_oob = warp_comm[0][2];
            int pred = warp_comm[0][3];

            int neighbor_id;
            bool apply;
            bool cond;
            while (coop_offset + threadIdx.x < coop_oob) {
                // Gather
                neighbor_id = col_indices[coop_offset+threadIdx.x];
                cond = Functor::cond_advance(pred, neighbor_id, coop_offset+threadIdx.x, threadIdx.x, coop_rank, data, iteration);
                apply = Functor::apply_advance(pred, neighbor_id, coop_offset+threadIdx.x, threadIdx.x, coop_rank, data, iteration);
                // Scatter neighbor
                if (output_data != nullptr) output_data[coop_rank] = idempotence ? neighbor_id : ((cond && apply) ? neighbor_id : -1);
                coop_offset += nt;
                coop_rank += nt;
                //printf("pred:%d, neighbor:%d\n", pred, neighbor_id);
            }
            if (coop_offset + idx < coop_oob) {
                // Gather
                neighbor_id = col_indices[coop_offset+threadIdx.x];
                cond = Functor::cond_advance(pred, neighbor_id, coop_offset+threadIdx.x, threadIdx.x, coop_rank, data, iteration);
                apply =  Functor::apply_advance(pred, neighbor_id, coop_offset+threadIdx.x, threadIdx.x, coop_rank, data, iteration);
                // Scatter neighbor
                if (output_data != nullptr) output_data[coop_rank] = idempotence ? neighbor_id : ((cond && apply) ? neighbor_id : -1);
                //printf("pred:%d, neighbor:%d\n", pred, neighbor_id);
            }
        }
    }

    static __device__ __forceinline__ void expand_warp(
            int idx,
            int input_size,
            int num_edges,
            typename Problem::data_slice_t *data,
            const int *input_data,
            int *output_data,
            int *row_lengths,
            const int *row_offsets,
            const int *col_indices,
            const int2 *row_ranks,
            int warp_gather_threshold,
            int (&warp_comm)[mgpu::warp_size][4],
            int nt,
            int iteration,
            int idempotence) {
        int warp_id = threadIdx.x / mgpu::warp_size;
        int lane_id = util::LaneId();
        //cannot have more than num_edges edge to expand
        if (idx >= num_edges) return;
        while (__any(row_lengths[idx] >= warp_gather_threshold)) {
            if (idx < input_size && row_lengths[idx] >= warp_gather_threshold) {
                warp_comm[warp_id][0] = lane_id;
            }
            if (lane_id == warp_comm[warp_id][0]) {
                // got control of the warp
                warp_comm[warp_id][0] = row_offsets[input_data[idx]];
                warp_comm[warp_id][1] = row_ranks[idx].x;
                warp_comm[warp_id][2] = row_offsets[input_data[idx]] + row_lengths[idx];
                warp_comm[warp_id][3] = input_data[idx];
                // unset row length
                row_lengths[idx] = 0;
            }
            int coop_offset = warp_comm[warp_id][0];
            int coop_rank = warp_comm[warp_id][1] + lane_id;
            int coop_oob = warp_comm[warp_id][2];
            int pred = warp_comm[warp_id][3];

            int neighbor_id;
            bool apply;
            bool cond;
            while (coop_offset + mgpu::warp_size < coop_oob) {
                // Gather
                neighbor_id = col_indices[coop_offset+lane_id];
                cond = Functor::cond_advance(pred, neighbor_id, coop_offset+lane_id, lane_id, coop_rank, data, iteration);
                apply = Functor::apply_advance(pred, neighbor_id, coop_offset+lane_id, lane_id, coop_rank, data, iteration);
                // Scatter neighbor
                if (output_data != nullptr) output_data[coop_rank] = idempotence ? neighbor_id : ((cond && apply) ? neighbor_id : -1);
                coop_offset += mgpu::warp_size;
                coop_rank += mgpu::warp_size;
            }
            if (coop_offset + lane_id < coop_oob) {
                // Gather
                neighbor_id = col_indices[coop_offset+lane_id];
                cond = Functor::cond_advance(pred, neighbor_id, coop_offset+lane_id, lane_id, coop_rank, data, iteration);
                apply = Functor::apply_advance(pred, neighbor_id, coop_offset+lane_id, lane_id, coop_rank, data, iteration);
                // Scatter neighbor
                if (output_data != nullptr) output_data[coop_rank] = idempotence ? neighbor_id : ((cond && apply) ? neighbor_id : -1);
            }
        }
    }

    static __device__ __forceinline__ void expand_scan(
            int idx,
            int input_size,
            int num_edges,
            int *input_data,
            int *row_lengths,
            const int *row_offsets,
            int2 *row_ranks,
            int (&gather_offsets)[mgpu::warp_size<<2],
            int (&gather_preds)[mgpu::warp_size<<2],
            int smem_gather_elements,
            int progress) {
        // reuse row_ranks[idx].x as row_progress
        // at this point, it must be 0.
        if (idx >= num_edges) return;
        int scratch_offset = row_ranks[idx].y + row_ranks[idx].x - progress;
        while (idx < input_size && (row_ranks[idx].x < row_lengths[idx]) &&
                (scratch_offset < smem_gather_elements)) {
            gather_offsets[scratch_offset] = row_offsets[input_data[idx]]+row_ranks[idx].x;
            gather_preds[scratch_offset] = input_data[idx];
            row_ranks[idx].x++;
            scratch_offset++;
        }
    }
};

template<typename Problem, typename Functor, bool idempotence, bool has_output>
int advance_twc_kernel(std::shared_ptr<Problem> problem,
              std::shared_ptr<frontier_t<int> > &input,
              std::shared_ptr<frontier_t<int> > &output,
              int warp_gather_threshold,
              int cta_gather_threshold,
              int iteration,
              standard_context_t &context)
{
    // TODO: tune this after finish
    typedef launch_box_t<
        arch_20_cta<128, 1>,
        arch_35_cta<128, 1>,
        arch_52_cta<128, 1> > launch_t;
    int *input_data = input.get()->data()->data();
    int input_size = (int)input.get()->size();
    int *row_lengths = problem.get()->gslice->d_row_lengths.data();
    int *row_offsets = problem.get()->gslice->d_row_offsets.data();
    int2 *coarse_fine_ranks = problem.get()->gslice->d_scanned_coarse_fine_row_offsets.data();
    int *col_indices = problem.get()->gslice->d_col_indices.data();
    mem_t<int2> counts(1, context); 

    auto segment_sizes = [=]__device__(int idx) {
        int2 count;
        int v = input_data[idx];
        int begin = row_offsets[v];
        int end = row_offsets[v+1];
        int len = end - begin;
        row_lengths[idx] = len;
        count.x = (len <= warp_gather_threshold) ? 0 : len;
        count.y = (len <= warp_gather_threshold) ? len : 0; 
        return count;
    };
    transform_scan<int2>(segment_sizes, (int)input.get()->size(), coarse_fine_ranks, plus_t<int2>(), counts.data(), context);

    int coarse_count = from_mem(counts)[0].x;
    int fine_count = from_mem(counts)[0].y;
    std::cout << input_size << " " << coarse_count << " " << fine_count << std::endl;

    // start cta/warp/thread expand
    if (has_output) output->resize(coarse_count+fine_count);
    int *output_data = has_output ? output.get()->data()->data() : nullptr;
    typename Problem::data_slice_t *data = problem.get()->d_data_slice.data();
    int num_edges = problem.get()->gslice->num_edges;
    auto f = [=]__device__(int idx) {
        __shared__ int warp_comm[mgpu::warp_size][4];
        __shared__ int cta_comm[1];
        __shared__ int gather_offsets[mgpu::warp_size<<2]; //128
        __shared__ int gather_preds[mgpu::warp_size<<2]; //128
        int smem_gather_elements = mgpu::warp_size<<2;

        //printf("nt:%d, blockDim:%d\n", launch_t::sm_ptx::nt, blockDim.x);
       
        if (coarse_count > 0) {
        AdvanceTWCFunctor<Problem, Functor>::expand_cta(idx,
                input_size,
                num_edges,
                data,
                input_data,
                output_data,
                row_lengths,
                row_offsets,
                col_indices,
                coarse_fine_ranks,
                cta_gather_threshold,
                warp_comm,
                cta_comm,
                launch_t::sm_ptx::nt,
                iteration,
                idempotence);

        AdvanceTWCFunctor<Problem, Functor>::expand_warp(idx,
                input_size,
                num_edges,
                data,
                input_data,
                output_data,
                row_lengths,
                row_offsets,
                col_indices,
                coarse_fine_ranks,
                warp_gather_threshold,
                warp_comm,
                launch_t::sm_ptx::nt,
                iteration,
                idempotence);
        }

        int progress = 0;
        while (progress < fine_count) {
            AdvanceTWCFunctor<Problem, Functor>::expand_scan(idx,
                input_size,
                num_edges,
                input_data,
                row_lengths,
                row_offsets,
                coarse_fine_ranks,
                gather_offsets,
                gather_preds,
                smem_gather_elements,
                progress);
            int scratch_remainder = (smem_gather_elements < fine_count - progress) ? smem_gather_elements : fine_count - progress;
            __syncthreads();
            bool apply;
            bool cond;
            for (int scratch_offset = threadIdx.x;
                    scratch_offset < scratch_remainder;
                    scratch_offset += launch_t::sm_ptx::nt)
            {
                int neighbor_id = col_indices[gather_offsets[scratch_offset]];
                int pred = gather_preds[scratch_offset]; 
                cond = Functor::cond_advance(pred, neighbor_id, gather_offsets[scratch_offset], gather_offsets[scratch_offset] - row_offsets[idx], coarse_count + progress + scratch_offset, data, iteration);
                apply = Functor::apply_advance(pred, neighbor_id, gather_offsets[scratch_offset], gather_offsets[scratch_offset] - row_offsets[idx], coarse_count + progress + scratch_offset, data, iteration);
                // Scatter neighbor 
                if (has_output) output_data[coarse_count + progress + scratch_offset] = idempotence ? neighbor_id : ((cond && apply) ? neighbor_id : -1);
            }
            progress += smem_gather_elements;
            __syncthreads();
        }
    };
    int num_threads = max(num_edges, cta_gather_threshold);
    //int num_threads = (int)input.get()->size();
    transform<launch_t>(f, num_threads, context);

    return coarse_count + fine_count;
}

} // end advance
} // end  oprtr
} // end gunrock
