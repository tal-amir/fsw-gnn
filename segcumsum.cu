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

extern "C" __global__ void segcumsum_kernel(const float* input, const int* segment_ids, int max_seg_size, float* output, float* block_sums, int* block_last_id, int size) {
    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    // Load input into shared memory
    if (index < size) {
        shared_data[tid] = input[index];
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
        output[index] = shared_data[tid];
    }

    // Write block sums
    if (tid == blockDim.x - 1) {
        block_sums[blockIdx.x] = shared_data[tid];
        block_last_id[blockIdx.x] = segment_ids[index];
    }
}



extern "C" __global__ void add_block_sums_kernel(float* output, const float* block_sums, const int* segment_ids, const int* block_last_id, int size) {
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int id_curr = segment_ids[index];

    // if ((blockIdx.x >= 1) && (block_last_id[blockIdx.x-1] == id_curr))
    // {
    //     output[index] += block_sums[blockIdx.x-1];
    // }

    // Calculate the sum of previous block sums
    float block_sum = 0.0f;

    for (int i = blockIdx.x-1; i >= 0; --i) {
        if (block_last_id[i] == id_curr)
            block_sum += block_sums[i];
        else
            break;
    }

    // Adjust the cumulative sum with the block sums
    if (index < size) {
        output[index] += block_sum;
    }
}

extern "C" void segcumsum_wrapper(const float* input, const int* segment_ids, int max_seg_size, float* output, float* block_sums, int* block_last_id, int size, int blocks, int threads_per_block, size_t shared_memory_size) {
    segcumsum_kernel<<<blocks, threads_per_block, shared_memory_size>>>(input, segment_ids, max_seg_size, output, block_sums, block_last_id, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

extern "C" void add_block_sums_wrapper(float* output, const float* block_sums, const int* segment_ids, const int* block_last_id, int size, int blocks, int threads_per_block) {
    add_block_sums_kernel<<<blocks, threads_per_block>>>(output, block_sums, segment_ids, block_last_id, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
