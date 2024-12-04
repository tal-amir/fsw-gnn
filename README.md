## Instructions

1. Download the files listed below.
2. Experiment with the minimal working examples: `demo_conv.py` or `demo_fsw_embedding.py`  
   In case of any problem, run `build_fsw_embedding`

## Files

`fsw_embedding.py` calculates the Fourier Sliced-Wasserstein Embedding  
`fsw_embedding.cu` CUDA library source code for FSW embedding  
`libfsw_embedding.so` CUDA library binary for FSW embedding  
`build_fsw_embedding` Script for building the CUDA library  
`fsw_conv.py` contains the `FSW_conv` module, which calculates one message passing iteration   
`demo_fsw_embedding.py` a test script that runs `FSW_embedding` on a random point-cloud  
`demo_conv.py` a test script that runs `FSW_conv` on a random graph 
