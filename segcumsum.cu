#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA_ERROR(call) {                                \
    const cudaError_t error = call;                             \
    if (error != cudaSuccess) {                                 \
        printf("Error: %s:%d, ", __FILE__, __LINE__);           \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                                \
    }                                                           \
}

extern "C" __global__ void segcumsum_kernel(float* values, const int* segment_ids, int size, int max_seg_size, float* block_sums_out, int* block_last_ids_out, bool return_next_level) {
    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    // Load input into shared memory
    if (index < size) {
        shared_data[tid] = values[index];
    } else {
        shared_data[tid] = 0.0f;
    }
    __syncthreads();

    // Inclusive scan (cumulative sum) within each block
    int id_curr = segment_ids[index];
    int id_lookback;
    bool stop = false;
    int stride_limit = min(max_seg_size, blockDim.x);

    for (int stride = 1; stride < stride_limit; stride *= 2) {
        float temp;
        if (tid >= stride)
        {            
            id_lookback = segment_ids[index-stride];
            if (stop)
            {
                temp = 0.0;
            }
            else if (id_curr != id_lookback)
            {
                temp = 0.0;
                stop = true;
            }
            else
            {
                temp = shared_data[tid - stride];
            }
        }
        else
        {
            temp = 0.0f;
        }
        __syncthreads();
        shared_data[tid] += temp;
        __syncthreads();
    }

    // Write results to output
    if (index < size) {
        values[index] = shared_data[tid];
    }

    // Write block sums
    if (return_next_level && (tid == blockDim.x - 1)) {
        block_sums_out[blockIdx.x] = shared_data[tid];
        block_last_ids_out[blockIdx.x] = segment_ids[index];
    }
}



extern "C" __global__ void add_block_sums_kernel(float* output, const float* block_sums, const int* segment_ids, const int* block_last_id, int size) {
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int id_curr = segment_ids[index];

    if ((blockIdx.x >= 1) && (block_last_id[blockIdx.x-1] == id_curr))
    {
        output[index] += block_sums[blockIdx.x-1];
    }
}

extern "C" void segcumsum_wrapper(float* values, const int* segment_ids, int size, int max_seg_size, float* block_sums_out, int* block_last_ids_out, bool return_next_level, int blocks, int threads_per_block, size_t shared_memory_size) {
    segcumsum_kernel<<<blocks, threads_per_block, shared_memory_size>>>(values, segment_ids, size, max_seg_size, block_sums_out, block_last_ids_out, return_next_level);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

extern "C" void add_block_sums_wrapper(float* output, const float* block_sums, const int* segment_ids, const int* block_last_id, int size, int blocks, int threads_per_block) {
    add_block_sums_kernel<<<blocks, threads_per_block>>>(output, block_sums, segment_ids, block_last_id, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
