# Segmented Cumulative Sum
#
# Part of our 2023 paper titled "Injective Sliced-Wasserstein Embedding for Sets and Weighted Point-Clouds"
# Tal Amir, Nadav Dym

# segcumsum (Segmented Cumulative Sum) is like the regular cumsum, but has the option to reset the accumulated sum in the middle.
# The input is given in a 1d tensor <values>, and an accompanying tensor of integers <segment_ids>, both of the same size.
# The input in <values> is regarded as an independent collection of segments, on each of which cumsum is calculated independently.
#
# For example: values = <some tensor of length 10>, segment_ids = [1,1,2,2,2,2,3,3,3,4]
#              Here cumsum is calculated independently on values[0:2], values[2:6], values[6:9] and values[9:10]
# The ids of consecutive segments do not have to be monotone increasing, but they do have to be unique for each segment.
# That is, segment_ids = [1,1,1,2,2,1,1] is illegal, and in such cases correct output is not guaranteed.
#
# Time complexity: When calculated without parallelism, O(n log k) with n = values.numel() and k being the maximal segment size.
#                  In practice it is much faster, due to CUDA's multitude of thread blocks.


import torch
import ctypes
import os
import numbers

# Load the segcumsum shared library from the same directory as the current file
mydir = os.path.dirname(os.path.abspath(__file__))
libsegcumsum_path = os.path.join(mydir, "libsegcumsum.so")
libsegcumsum = ctypes.CDLL(libsegcumsum_path)


# This is the main function that calculates the segmented cumsum.
# Input arguments: 
#   max_seg_size: an upper bound on the maximal length of a contiguous segment in <segment_ids>. If not provided, detected automatically.
#   in_place:     if set to True, writes the output directly to <values> instead of allocating new memory.
#   thorough_verify_input: verifies the input for correctness. meant for debugging purposes. in particular, checks <segment_ids>
#                          for repeated ids of different segments, and looks for infs and nans in <values>.
#   always_use_pure_torch: when set to True, always uses the pure torch implementation.
#                          otherwise, when the input is on a cuda device, uses a custom cuda implementation.
#                          the cuda implementation has a better memory bottleneck.
#                          in terms of running time, both are comparable, with two-fold differences for one over
#                          the other or vice versa.
#
# Output: The segmented cumsum of <values> according to <segment_ids>.
def segcumsum(values, segment_ids, max_seg_size=None, in_place=False, thorough_verify_input=False, always_use_pure_torch=False):
    # Verify input device, dtypes and shapes
    assert values.dim() == 1, 'values must be a 1-dimensional tensor'
    assert segment_ids.dim() == 1, 'segment_ids must be a 1-dimensional tensor'
    assert segment_ids.numel() == values.numel(), 'values and segment_ids must contain the same number of elements'
    assert segment_ids.dtype in (torch.int32,torch.int64), 'segment_ids must have int32 or int64 dtype'
    assert values.device == segment_ids.device, 'values and segment_ids must be on the same device'

    # Ensure all data is contiguous
    assert not segment_ids.is_sparse, 'segment_ids cannot be sparse'
    assert segment_ids.is_contiguous(), 'segment_ids must be in contiguous format'

    assert not values.is_sparse, 'values cannot be sparse'
    assert (not in_place) or values.is_contiguous(), 'when in_place==True, values must be in contiguous format'

    num_segments = None

    if max_seg_size is None:
        # Calculate maximal segmet size
        _, counts_consecutive = torch.unique_consecutive(segment_ids, return_counts=True)
        del _
        num_segments = counts_consecutive.numel()
        max_seg_size_real = int(torch.max(counts_consecutive))
        max_seg_size = max_seg_size_real
        del counts_consecutive
    else:
        assert isinstance(max_seg_size, numbers.Number)
        assert max_seg_size >= 1

    if thorough_verify_input:
        if num_segments is None:
            _, counts_consecutive = torch.unique_consecutive(segment_ids, return_counts=True)
            del _
            num_segments = counts_consecutive.numel()
            max_seg_size_real = int(torch.max(counts_consecutive))
            del counts_consecutive

        _, counts_total = torch.unique(segment_ids, return_counts=True)
        del _
        num_segments_unique = counts_total.numel()
        del counts_total

        assert num_segments == num_segments_unique, 'repeated segment IDs detected'
        assert max_seg_size == max_seg_size_real, 'incorrect max_seg_size detected (got %d, correct is %d)' % (max_seg_size, max_seg_size_real)

        assert not torch.isinf(values).any(), "Found infs in ''values''"
        assert not torch.isnan(values).any(), "Found nans in ''values''"

    # Calculate and return the segmented cumsum
    if (values.device.type == 'cuda') and (not always_use_pure_torch):
        return segcumsum_cuda(values, segment_ids, max_seg_size, in_place)
    else:
        return segcumsum_torch(values, segment_ids, max_seg_size, in_place)
    
# torch implementation
def segcumsum_torch(values, segment_ids, max_seg_size, in_place):
    assert values.is_contiguous(), 'in the segcumsum_torch implementation, ''values'' must be in contiguous format'

    if in_place:
        out = values
    else:
        out = torch.clone(values, memory_format=torch.contiguous_format)

    return segcumsum_torch_main(out, segment_ids, max_seg_size)


# main loop of torch implementation
# Note: using torch jit here makes it empirically slower
def segcumsum_torch_main(values, segment_ids, max_seg_size : int):
    n = values.numel()

    stride = 0
    while stride < max_seg_size:
        stride = max(1, 2*stride)
        values[stride:n] += (segment_ids[stride:n] == segment_ids[0:(n-stride)]) * values[0:(n-stride)]
    
    return values


# cuda implementation
def segcumsum_cuda(values, segment_ids, max_seg_size, in_place):
    # Maximal number of CUDA threads to use per block.
    # Note: This is automatically capped by the maximal number supported by the architecture.
    # Set to an arbitrarily large number (e.g. 1e6) to determine automatically.
    max_num_threads_per_block = 1e6

    print('dtype = ', segment_ids.dtype)

    assert values.device.type == 'cuda', 'the tensor ''values'' must be on a CUDA device'
    assert segment_ids.device.type == 'cuda', 'the tensor ''segment_ids'' must be on a CUDA device'
    assert segment_ids.dtype == torch.int64, 'segment_ids must have int64 dtype'
    
    # Process input data types
    if values.dtype == torch.float32:
        dtype_num = 0;
        c_num_type = ctypes.c_float
    elif values.dtype == torch.float64:
        dtype_num = 1;
        c_num_type = ctypes.c_double
    else:
        raise RuntimeError("Unsupported input_tensor dtype ''%s''" % (str(values.dtype)))

    n = values.numel()

    # Determine the maximal number of threads per block supported in the current CUDA device
    cuda_max_threads_per_block = get_max_threads_per_block(values.device.index)

    # Take the smallest multiple of 32 greater or equal to the input size, but no less than 64
    num_threads_2 = max(64, (n+31)//32)

    threads_per_block = min(num_threads_2, max_num_threads_per_block, cuda_max_threads_per_block)
    shared_memory_size = threads_per_block * ctypes.sizeof(c_num_type)

    assert threads_per_block > 1, 'threads_per_block must be greater than 1'

    # Construct block hierarchy
    tensor_sizes = [ n, ]
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
            if in_place:
                output_tensor_new = values
            else:
                output_tensor_new = torch.clone(values, memory_format=torch.contiguous_format)

            id_tensor_new = segment_ids

        else:
            output_tensor_new = torch.empty(size=(s,), device=values.device, dtype=values.dtype, memory_format=torch.contiguous_format)
            id_tensor_new = torch.empty(size=(s,), device=segment_ids.device, dtype=segment_ids.dtype, memory_format=torch.contiguous_format)

        output_tensors.append(output_tensor_new)
        id_tensors.append(id_tensor_new)

    # Define the kernel signatures
    libsegcumsum.segcumsum_wrapper.argtypes = [
        ctypes.c_int64,     # dtype_num
        ctypes.c_void_p,  # values input/output pointer
        ctypes.c_void_p,  # segment_ids pointer
        ctypes.c_int64,     # size
        ctypes.c_int64,     # max_seg_size
        ctypes.c_void_p,  # block sums output pointer
        ctypes.c_void_p,  # block last ids output pointer
        ctypes.c_bool,    # return_next_level boolean
        ctypes.c_int64,     # blocks
        ctypes.c_int64,     # threads_per_block
        ctypes.c_size_t   # shared_memory_size
    ]
    libsegcumsum.segcumsum_wrapper.restype = None

    libsegcumsum.add_block_sums_wrapper.argtypes = [
        ctypes.c_int64,     # dtype_num
        ctypes.c_void_p,  # output pointer
        ctypes.c_void_p,  # block_sums pointer
        ctypes.c_void_p,  # segment_ids pointer
        ctypes.c_void_p,  # block_last_id pointer
        ctypes.c_int64,     # size
        ctypes.c_int64,     # blocks
        ctypes.c_int64      # threads_per_block
    ]
    libsegcumsum.add_block_sums_wrapper.restype = None

    for i,s in enumerate(tensor_sizes):
        return_next_level = ( i < (len(tensor_sizes) - 1) )

        # Launch the segcumsum_wrapper
        libsegcumsum.segcumsum_wrapper(
            ctypes.c_int64(dtype_num),
            ctypes.c_void_p(output_tensors[i].data_ptr()),
            ctypes.c_void_p(id_tensors[i].data_ptr()),
            ctypes.c_int64(tensor_sizes[i]),            
            ctypes.c_int64(max_seg_sizes[i]),
            ctypes.c_void_p(output_tensors[i+1].data_ptr() if return_next_level else 0),
            ctypes.c_void_p(id_tensors[i+1].data_ptr() if return_next_level else 0),
            ctypes.c_bool( return_next_level  ),
            ctypes.c_int64(num_blocks[i]),
            ctypes.c_int64(threads_per_block),
            ctypes.c_size_t(shared_memory_size)
        )


    for i in reversed(range(len(tensor_sizes)-1)):

        # Launch the add_block_sums_wrapper
        libsegcumsum.add_block_sums_wrapper(
            ctypes.c_int64(dtype_num),
            ctypes.c_void_p(output_tensors[i].data_ptr()),
            ctypes.c_void_p(output_tensors[i+1].data_ptr()),
            ctypes.c_void_p(id_tensors[i].data_ptr()),
            ctypes.c_void_p(id_tensors[i+1].data_ptr()),
            ctypes.c_int64(tensor_sizes[i]),
            ctypes.c_int64(num_blocks[i]),
            ctypes.c_int64(threads_per_block)
        )

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


# Returns the maximal number of threads per block supported by the CUDA device with the given index
def get_max_threads_per_block(device_index):
    libsegcumsum.get_max_threads_per_block.argtypes = [ ctypes.c_int ]
    libsegcumsum.get_max_threads_per_block.restype = ctypes.c_int
    return libsegcumsum.get_max_threads_per_block(ctypes.c_int(device_index))

