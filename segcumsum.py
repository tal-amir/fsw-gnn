import torch
import ctypes
import os

# Load the shared library
libsegcumsum = ctypes.CDLL(os.path.abspath("libsegcumsum.so"))

# Segmented Cumulative Sum
# This is a wrapper function that calls CUDA kernels
def segcumsum(input_tensor, segment_ids, max_seg_size=None):
    default_num_threads_per_block = 2

    # Determine some CUDA properties
    device_index = torch.cuda.device(input_tensor.device)
    device_properties = torch.cuda.get_device_properties(device_index)
    max_threads_per_block = device_properties.max_threads_per_multi_processor # Not sure this is the right property    

    threads_per_block = min(default_num_threads_per_block, min(max_threads_per_block, input_tensor.numel()))
    shared_memory_size = threads_per_block * ctypes.sizeof(ctypes.c_float)

    assert threads_per_block > 1, 'threads_per_block must be greater than 1'

    input_tensor = input_tensor.contiguous()
    segment_ids = segment_ids.contiguous()

    if max_seg_size is None:
        # Calculate maximal segmet size
        _, counts = torch.unique_consecutive(segment_ids, return_counts=True)
        max_seg_size = int(torch.max(counts))
        del counts

    # Construct block hierarchy
    tensor_sizes = [ input_tensor.numel(), ]
    num_blocks = []
    max_seg_sizes = [ max_seg_size, ]
    
    # Stop dividing when the whole tensor fits in one block
    while tensor_sizes[-1] > threads_per_block:
        tensor_size_new = (tensor_sizes[-1] + threads_per_block - 1) // threads_per_block
        tensor_sizes.append(tensor_size_new)

        num_blocks.append(tensor_size_new)

        max_seg_size_new = (max_seg_sizes[-1] + threads_per_block - 1) // threads_per_block
        max_seg_sizes.append(max_seg_size_new)

    num_blocks.append(1)

    output_tensors = []
    id_tensors = []

    for i,s in enumerate(tensor_sizes):
        if i == 0:
            output_tensor_new = torch.clone(input_tensor, memory_format=torch.contiguous_format)
            id_tensor_new = segment_ids
        else:
            output_tensor_new = torch.empty(size=(s,), device=input_tensor.device, dtype=input_tensor.dtype, memory_format=torch.contiguous_format)
            id_tensor_new = torch.empty(size=(s,), device=segment_ids.device, dtype=segment_ids.dtype, memory_format=torch.contiguous_format)

        #TODO: Remove
        # output_tensor_new = torch.empty(size=(s,), device=input_tensor.device, dtype=input_tensor.dtype, memory_format=torch.contiguous_format)
        # id_tensor_new = torch.empty(size=(s,), device=segment_ids.device, dtype=segment_ids.dtype, memory_format=torch.contiguous_format)

        # output_tensor_new[:] = 8
        # id_tensor_new[:] = 8

        output_tensors.append(output_tensor_new)
        id_tensors.append(id_tensor_new)


    # Define the kernel signatures
    libsegcumsum.segcumsum_wrapper.argtypes = [
        ctypes.c_void_p,  # values input/output pointer
        ctypes.c_void_p,  # segment_ids pointer
        ctypes.c_int,     # size
        ctypes.c_int,     # max_seg_size
        ctypes.c_void_p,  # block sums output pointer
        ctypes.c_void_p,  # block last ids output pointer
        ctypes.c_bool,    # return_next_level boolean
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


    for i,s in enumerate(tensor_sizes):
        return_next_level = ( i < (len(tensor_sizes) - 1) )

        # Launch the segcumsum_wrapper
        libsegcumsum.segcumsum_wrapper(
            ctypes.c_void_p(output_tensors[i].data_ptr()),
            ctypes.c_void_p(id_tensors[i].data_ptr()),
            ctypes.c_int(tensor_sizes[i]),            
            ctypes.c_int(max_seg_sizes[i]),
            ctypes.c_void_p(output_tensors[i+1].data_ptr() if return_next_level else 0),
            ctypes.c_void_p(id_tensors[i+1].data_ptr() if return_next_level else 0),
            ctypes.c_bool( return_next_level  ),
            ctypes.c_int(num_blocks[i]),
            ctypes.c_int(threads_per_block),
            ctypes.c_size_t(shared_memory_size)
        )

        torch.cuda.synchronize()

    for i in reversed(range(len(tensor_sizes)-1)):
        # print('Running level %d. First level: %d' %(i, len(tensor_sizes)-2) )
        # print('Tensor size: ', tensor_sizes[i], output_tensors[i].numel(), id_tensors[i].numel())
        # print('Num blocks: ', num_blocks[i], output_tensors[i+1].numel(), id_tensors[i+1].numel())
        # print('Num levels: ', len(output_tensors))

        # Launch the add_block_sums_wrapper
        libsegcumsum.add_block_sums_wrapper(
            ctypes.c_void_p(output_tensors[i].data_ptr()),
            ctypes.c_void_p(output_tensors[i+1].data_ptr()),
            ctypes.c_void_p(id_tensors[i].data_ptr()),
            ctypes.c_void_p(id_tensors[i+1].data_ptr()),
            ctypes.c_int(tensor_sizes[i]),
            ctypes.c_int(num_blocks[i]),
            ctypes.c_int(threads_per_block)
        )

        # Synchronize to ensure the final result is ready
        torch.cuda.synchronize()

        # assert output_tensors[i][0] != 2

    return output_tensors[0]


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
