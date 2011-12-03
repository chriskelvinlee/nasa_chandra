import numpy as np
from pylab import *
import os
import sys
import pycuda.autoinit
import pycuda.driver as cu
import pycuda.compiler as nvcc
from pycuda import gpuarray
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel
import time

"""
# DOES NOT SEEM TO WORK FOR RESONANCE
#Get the input filename from the command line
try:
    file_name = sys.argv[1]; MaxRad = float(sys.argv[2]); Threshold = float(sys.argv[3])
except:
    print "Usage:",sys.argv[0], "infile maxrad threshold"; sys.exit(1)
"""

file_name   = 'png_input/114_ccd7_small.png'
MaxRad      = float(2)
Threshold   = float(10)

# setup input file
IMG_rgb = imread(file_name)
IMG = array( IMG_rgb[:,:,0] )

# Get image data
Lx = np.int32( IMG.shape[0] )
Ly = np.int32( IMG.shape[1] )

# Allocate memory
# size of the box needed to reach the threshold value or maxrad value
#BOX = np.zeros((Lx, Ly), dtype=np.float64) 
BOX = array(IMG)
# normalized array
# NORM = np.zeros((Lx, Ly), dtype=np.float64)
NORM = array(IMG)
# output array
# OUT = np.zeros((Lx, Ly), dtype=np.float64)
OUT = array(IMG)

# Execution configuration
TPBx	= int( 32 )                # Sweet spot num of threads per block
TPBy    = int( 32 )                # 32*32 = 1024 max
nBx     = int( Ly/TPBx )           # Num of thread blocks
nBy     = int( Lx/TPBy ) 


#kernel for first part of algorithm that performs the smoothing
## TO DO ##   
# Implement The kernel
#########
kernel_smooth_source = \
"""
    __global__ void smoothingFilter(int Lx, int Ly, float* IMG, float* BOX, float* NORM, float* OUT)
    {
    // Indexing
    int tid = threadIdx.x;
    int tjd = threadIdx.y;
    int i = blockIdx.x * blockDim.x + tid;
    int j = blockIdx.y * blockDim.y + tjd;
    int stid = tjd * blockDim.x + tid;
    int gtid = j * Ly + i;  

    extern __shared__ float s_IMG[];
    s_IMG[stid] = IMG[gtid];
    __syncthreads();

    OUT[gtid] = IMG[gtid];

    }
    """
#kernel for the second part of the algorithm that normalizes the data
## TO DO ##   
# Implement The kernel
#########

kernel_norm_source = \
"""
    __global__ void normalizeFilter(int Lx, int Ly, float* IMG_norm, float* IMG, float* NORM )
    {
    // Indexing
    int tid = threadIdx.x;
    int tjd = threadIdx.y;
    int i = blockIdx.x * blockDim.x + tid;
    int j = blockIdx.y * blockDim.y + tjd;
    int stid = tjd * blockDim.x + tid;
    int gtid = j * Ly + i;  

    // shared memory for image matrix
    extern __shared__ float s_IMG[];
    s_IMG[stid] = IMG[gtid];
    // shared memory for NORM factors
    extern __shared__ float s_NORM[];
    s_NORM[stid] = NORM[gtid];
    __syncthreads();    

    // Compute all pixels except for image border
	if ( i > 0 && i < Ly-1 && j > 0 && j < Lx-1 )
	{
        // Compute within bounds of block dimensions
        if( tid > 0 && tid < blockDim.x-1 && tjd > 0 && tjd < blockDim.y-1 )
        {
            //perform calculations here
            //IMG_norm[gtid] = s_IMG[stid] / s_NORM[gtid]
            break;
        }
        // Compute block borders with global memory
        else
        {
            //perform calculations
            //IMG_norm[gtid] = IMG[gtid] / NORM[gtid]
            break;
        }
	}
	__syncthreads();
    }
"""

#kernel for the last part of the algorithm that creates the output image
## TO DO ##   
# Implement The kernel
#########
kernel_out_source = \
"""
    __global__ void outFilter( float* IMG_out, float* IMG_norm, float* BOX, int Lx, int Ly )
    {
    // Indexing
    int tid = threadIdx.x;
    int tjd = threadIdx.y;
    int i = blockIdx.x * blockDim.x + tid;
    int j = blockIdx.y * blockDim.y + tjd;
    int stid = tjd * blockDim.x + tid;
    int gtid = j * Ly + i;  

    extern __shared__ float s_IMG_norm[];
    s_IMG_norm[stid] = IMG_norm[gtid];
    __syncthreads();

    // Compute all pixels except for image border
	if ( i > 0 && i < Ly-1 && j > 0 && j < Lx-1 )
	{
        // Compute within bounds of block dimensions
        if( tid > 0 && tid < blockDim.x-1 && tjd > 0 && tjd < blockDim.y-1 )
        {
            //perform calculations here
            break;
        }
        // Compute block borders with global memory
        else
        {
            //perform calculations
            break;
        }
	}
	// Swap references to the images by replacing value
	__syncthreads();
    }
"""

# Initialize kernel
smoothing_kernel = nvcc.SourceModule(kernel_smooth_source).get_function("smoothingFilter")
# normalize_kernel = nvcc.SourceModule(kernel_norm_source).get_function("normalizeFilter")
# out_kernel = nvcc.SourceModule(kernel_out_source).get_function("outFilter")

total_start_time = time.time()
setup_start_time = time.time()

# Allocate memory and constants
smem_size   = int(TPBx*TPBy*4)

# Copy image to device
# ONLY NEED TO SEND IMG, NORM, BOX, OUT ARRAYS TO GLOBAL ONCE
IMG_device          = gpuarray.to_gpu(IMG)
# IMG_norm_device     = gpuarray.to_gpu(IMG)
# IMG_out_device      = gpuarray.to_gpu(IMG)
BOX_device          = gpuarray.to_gpu(BOX)
NORM_device         = gpuarray.to_gpu(NORM)
OUT_device          = gpuarray.to_gpu(OUT)

setup_stop_time = time.time()
smth_kernel_start_time   = cu.Event()
smth_kernel_stop_time    = cu.Event()
norm_kernel_start_time   = cu.Event()
norm_kernel_stop_time    = cu.Event()
out_kernel_start_time    = cu.Event()
out_kernel_stop_time     = cu.Event()

##########
# The kernel will convolve the image with a gaussian weighted sum
# determine the BOX size that allows the sum to reach either the maxRad or 
# threshold values
# This kernel will utilize the IMG and modify the BOX and NORM
##########

smth_kernel_start_time.record()
smoothing_kernel(Lx, Ly, IMG_device, BOX_device, NORM_device, OUT_device, block=( TPBx, TPBy,1 ),  grid=( nBx, nBy ), shared=( smem_size ) )
smth_kernel_stop_time.record()

##########
# This kernel will normalize the image with the value obtained from first kernel
# Normalizing kernel will utilize the NORM and modify the IMG
##########

# Copy normalization factor to host and box size array
# No need because it is already stored on device
# NORM = NORM_device.get()
# BOX = BOX_device.get()

norm_kernel_start_time.record()
# normalize_kernel(Lx, Ly, IMG_device, NORM_device, block=( TPBx, TPBy,1 ),  grid=( nBx, nBy ), shared=( smem_size ) )
norm_kernel_stop_time.record()

# Copy image to host and send to output kernel
# No need because it is already stored on device
# IMG_norm = IMG_norm_device.get()

##########
# This will resmooth the data utilizing the new normalized image
# This kernel will utilize the BOX and IMG_norm and modify the OUT
##########
out_kernel_start_time.record()
# out_kernel(Lx, Ly, BOX_device, IMG_device, OUT_device, block=( TPBx, TPBy,1 ),  grid=( nBx, nBy ), shared=( smem_size ) )
out_kernel_stop_time.record()

# Copy image to host and 
IMG_out = OUT_device.get()


total_stop_time = time.time()
imsave('{}_smoothed_gpu.png'.format(file_name), IMG_out, cmap=cm.gray, vmin=0, vmax=1)

# Print results & save
total_time      = (total_stop_time - total_start_time)
setup_time      = (setup_stop_time - setup_start_time)
smth_ker_time   = (smth_kernel_start_time.time_till(smth_kernel_stop_time) * 1e-3)
norm_ker_time   = (norm_kernel_start_time.time_till(norm_kernel_stop_time) * 1e-3)
out_ker_time    = (out_kernel_start_time.time_till(out_kernel_stop_time) * 1e-3)

print "Total Time: %f"                  % total_time
print "Setup Time: %f"                  % setup_time
print "Kernel (Smooth) Time: %f"        % smth_ker_time
print "Kernel (Normalize) Time: %f"     % norm_ker_time
print "Kernel (Output) Time: %f"        % out_ker_time

