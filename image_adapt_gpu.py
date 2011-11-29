import numpy as np
from pylab import *
from mpi4py import MPI
import pycuda.autoinit
import pycuda.driver as cu
import pycuda.compiler as nvcc
from pycuda import gpuarray
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel
import time


# Read image
file_name       = filename
IMG_rgb         = imread(file_name)
IMG             = array( IMG_rgb[:,:,0] )

# Get image data
Lx              = np.int32( IMG.shape[0] )
Ly              = np.int32( IMG.shape[1] )

# Allocate memory
RAD = np.zeros((Lx, Ly), dtype=np.float64)
NORM = np.zeros((Lx, Ly), dtype=np.float64)
OUT = np.zeros((Lx, Ly), dtype=np.float64)

# Array to hold updated values
# This array can be used to implement the weight sums
# for example gaussian, tophat, cone
# currently set to one for general use.
ww = 1.0#np.ones((Lx, Ly), dtype=np.float64)

# Parameters
Threshold = 30.0
MaxRad = 30.0

# Execution configuration
TPBx	= int( 32 )                # Sweet spot num of threads per block
TPBy    = int( 32 )                # 32*32 = 1024 max
nBx     = int( Ly/TPBx )           # Num of thread blocks
nBy     = int( Lx/TPBy ) 

## TO DO ##   
# I currently don't have anything coded in the kernels, just outline
#########

#kernel for first part of algorithm that performs the smoothing
## TO DO ##   
# Implement The kernel
#########
kernel_smooth_source = \
"""
    __global__ void smoothingFilter( float* next_im, float* curr_im, int Lx, int Ly )
    {
    // Indexing
    int tid = threadIdx.x;
    int tjd = threadIdx.y;
    int i = blockIdx.x * blockDim.x + tid;
    int j = blockIdx.y * blockDim.y + tjd;
    int stid = tjd * blockDim.x + tid;
    int gtid = j * Ly + i;  

    extern __shared__ float s_curr_im[];
    s_curr_im[stid] = curr_im[gtid];
    __syncthreads();

    // Compute all pixels except for image border
	if ( i > 0 && i < Ly-1 && j > 0 && j < Lx-1 )
	{
        // Compute within bounds of block dimensions
        if( tid > 0 && tid < blockDim.x-1 && tjd > 0 && tjd < blockDim.y-1 )
        {
            //perform calculations here
        }
        // Compute block borders with global memory
        else
        {
            //perform calculations
        }
	}
	// Swap references to the images by replacing value
	__syncthreads();
	curr_im[gtid] = next_im[gtid];

    }
    """
#kernel for the second part of the algorithm that normalizes the data
## TO DO ##   
# Implement The kernel
#########
kernel_norm_source = \
"""
    __global__ void normalizeFilter( float* next_im, float* curr_im, int Lx, int Ly, float* NORM )
    {
    // Indexing
    int tid = threadIdx.x;
    int tjd = threadIdx.y;
    int i = blockIdx.x * blockDim.x + tid;
    int j = blockIdx.y * blockDim.y + tjd;
    int stid = tjd * blockDim.x + tid;
    int gtid = j * Ly + i;  

    // shared memory for image matrix
    extern __shared__ float s_curr_im[];
    s_curr_im[stid] = curr_im[gtid];
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
            next_im[gtid] = s_curr_im[stid] / s_NORM[gtid]
        }
        // Compute block borders with global memory
        else
        {
            //perform calculations
            next_im[gtid] = curr_im[gtid] / NORM[gtid]
        }
	}
	// Swap references to the images by replacing value
	// DO WE NEED TO ACTUALLY SWAP REFERENCES?
	__syncthreads();
	curr_im[gtid] = next_im[gtid];

    }
    """

#kernel for the last part of the algorithm that creates the output image
## TO DO ##   
# Implement The kernel
#########
kernel_out_source = \
"""
    __global__ void outFilter( float* next_im, float* curr_im, int Lx, int Ly )
    {
    // Indexing
    int tid = threadIdx.x;
    int tjd = threadIdx.y;
    int i = blockIdx.x * blockDim.x + tid;
    int j = blockIdx.y * blockDim.y + tjd;
    int stid = tjd * blockDim.x + tid;
    int gtid = j * Ly + i;  

    extern __shared__ float s_curr_im[];
    s_curr_im[stid] = curr_im[gtid];
    __syncthreads();

    // Compute all pixels except for image border
	if ( i > 0 && i < Ly-1 && j > 0 && j < Lx-1 )
	{
        // Compute within bounds of block dimensions
        if( tid > 0 && tid < blockDim.x-1 && tjd > 0 && tjd < blockDim.y-1 )
        {
            //perform calculations here
        }
        // Compute block borders with global memory
        else
        {
            //perform calculations
        }
	}
	// Swap references to the images by replacing value
	__syncthreads();
	curr_im[gtid] = next_im[gtid];

    }
    """

# Initialize kernel
smoothing_kernel = nvcc.SourceModule(kernel_smooth_source).get_function("smoothingFilter")
normalize_kernel = nvcc.SourceModule(kernel_norm_source).get_function("normalizeFilter")
out_kernel = nvcc.SourceModule(kernel_out_source).get_function("outFilter")

total_start_time = time.time()
setup_start_time = time.time()

# Allocate memory and constants
smem_size   = int(TPBx*TPBy*4)
curr_im     = array( IMG )
next_im     = array( IMG )

# Copy image to device
curr_im_device = gpuarray.to_gpu(curr_im)
next_im_device = gpuarray.to_gpu(next_im)

setup_stop_time = time.time()

# Calculate mean on device
# DONT NEED THESE TWO LINES?
# m_array = gpuarray.to_gpu(curr_im)
# sum = gpuarray.sum(m_array).get()

kernel_start_time = cu.Event()
kernel_stop_time = cu.Event()

#
# IS THERE A REASON WHY WE HAVE TO USE THREE KERNELS INSTEAD OF ONE?
# REASON BEING THAT EVERY SINGLE TIME WE CALL A KERNEL AND LOADING THE
# IMG, RAD, NORM, AND OUT MATRIX INTO SHARED MEMORY, EACH TIME WILL BE
# COSTLY. LOOKING BACK AT THE SERIAL CODE SUGGESTS THAT WE CAN COMBINE
# THE THREE FOR LOOPS.
## TO DO ##   
#########

# Run the CUDA kernel with the appropriate inputs and outputs
## TO DO ##   
# This kernel will utilize the IMG and modify the RAD and NORM
#########

# Put RAD_device and NORM_device as parameters, you may change
smth_kernel_start_time.record()
smoothing_kernel(next_im_device, curr_im_device, Lx, Ly, RAD_device, NORM_device, block=( TPBx, TPBy,1 ),  grid=( nBx, nBy ), shared=( smem_size ) )
smth_kernel_stop_time.record()

# Run the CUDA kernel with the appropriate inputs and outputs
## TO DO ## 
# Normalizing kernel will utilize the NORM and modify the IMG
#########

# Copy normalization factor to host
NORM = NORM_device.get()

# Takes in 
norm_kernel_start_time.record()
normalize_kernel(next_im_device, curr_im_device, Lx, Ly, NORM, block=( TPBx, TPBy,1 ),  grid=( nBx, nBy ), shared=( smem_size ) )
norm_kernel_stop_time.record()

# Copy image to host and send to output kernel
curr_im = curr_im_device.get()

# Run the CUDA kernel with the appropriate inputs and outputs
## TO DO ##   
# This kernel will utilize the RAD and IMG and modify the OUT
#########
out_kernel_start_time.record()
out_kernel(next_im_device, curr_im, Lx, Ly, block=( TPBx, TPBy,1 ),  grid=( nBx, nBy ), shared=( smem_size ) )
out_kernel_start_time.record()


total_stop_time = time.time()
imsave('{}_smoothed.png'.format(filename), curr_im, cmap=cm.gray, vmin=0, vmax=1)

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

