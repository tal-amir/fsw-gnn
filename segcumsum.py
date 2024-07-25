import torch
import ctypes
import os

# Load the shared library
libsegcumsum = ctypes.CDLL(os.path.abspath("libsegcumsum.so"))

# Segmented Cumulative Sum
# This is a wrapper function that calls CUDA kernels
def segcumsum(input_tensor, segment_ids, max_seg_size=None):
    # Determine some CUDA properties
    device_index = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device_index)
    max_threads_per_block = device_properties.max_threads_per_multi_processor # Not sure this is the right property

    threads_per_block = min(512, min(max_threads_per_block, input_tensor.numel()))
    num_blocks = (input_tensor.numel() + threads_per_block - 1) // threads_per_block
    shared_memory_size = threads_per_block * ctypes.sizeof(ctypes.c_float)

    input_tensor = input_tensor.contiguous()
    segment_ids = segment_ids.contiguous()
    output_tensor = torch.empty_like(input_tensor)
    block_sums = torch.empty(num_blocks, device=input_tensor.device, dtype=input_tensor.dtype)
    block_last_id = torch.empty(num_blocks, device=input_tensor.device, dtype=torch.int32)

    # Get pointers to the tensors
    input_ptr = input_tensor.data_ptr()
    segment_ids_ptr = segment_ids.data_ptr()
    output_ptr = output_tensor.data_ptr()
    block_sums_ptr = block_sums.data_ptr()
    block_last_id_ptr = block_last_id.data_ptr()
    
    if max_seg_size is None:
        # Calculate maximal segmet size
        _, counts = torch.unique_consecutive(segment_ids, return_counts=True)
        max_seg_size = int(torch.max(counts))

    # Define the kernel signatures
    libsegcumsum.segcumsum_wrapper.argtypes = [
        ctypes.c_void_p,  # input pointer
        ctypes.c_void_p,  # segment_ids pointer
        ctypes.c_int,     # max_seg_size
        ctypes.c_void_p,  # output pointer
        ctypes.c_void_p,  # block sums pointer
        ctypes.c_void_p,  # block last id pointer
        ctypes.c_int,     # size
        ctypes.c_int,     # blocks
        ctypes.c_int,     # threads_per_block
        ctypes.c_size_t   # shared_memory_size
    ]
    libsegcumsum.segcumsum_wrapper.restype = None

    libsegcumsum.add_block_sums_wrapper.argtypes = [
        ctypes.c_void_p,  # output pointer
        ctypes.c_void_p,  # block_sums pointer
        ctypes.c_void_p,  # segment_ids pointer
        ctypes.c_void_p,  # block_last_id pointer
        ctypes.c_int,     # size
        ctypes.c_int,     # blocks
        ctypes.c_int      # threads_per_block
    ]
    libsegcumsum.add_block_sums_wrapper.restype = None

    # Launch the segcumsum_wrapper
    libsegcumsum.segcumsum_wrapper(
        ctypes.c_void_p(input_ptr),
        ctypes.c_void_p(segment_ids_ptr),
        ctypes.c_int(max_seg_size),
        ctypes.c_void_p(output_ptr),
        ctypes.c_void_p(block_sums_ptr),
        ctypes.c_void_p(block_last_id_ptr),
        ctypes.c_int(input_tensor.numel()),
        ctypes.c_int(num_blocks),
        ctypes.c_int(threads_per_block),
        ctypes.c_size_t(shared_memory_size)
    )

    # Synchronize to ensure all blocks have completed before the second kernel
    torch.cuda.synchronize()

    # Launch the add_block_sums_wrapper
    # libsegcumsum.add_block_sums_wrapper(
    #     ctypes.c_void_p(output_ptr),
    #     ctypes.c_void_p(block_sums_ptr),
    #     ctypes.c_void_p(segment_ids_ptr),
    #     ctypes.c_void_p(block_last_id_ptr),
    #     ctypes.c_int(input_tensor.numel()),
    #     ctypes.c_int(num_blocks),
    #     ctypes.c_int(threads_per_block)
    # )

    # # Synchronize to ensure the final result is ready
    # torch.cuda.synchronize()

    return output_tensor


# This is a slow alternative of segcumsum() to verify the correctness of the results
def segcumsum_slow(x, segment_ids):
    out = torch.empty_like(x)

    for i in range(len(x)):
        if (i == 0):
            out[i] = x[i]
        elif segment_ids[i] == segment_ids[i-1]:
            out[i] = out[i-1] + x[i]
        else:
            out[i] = x[i]
    
    return out
