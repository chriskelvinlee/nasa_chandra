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

#Debug value, if 1 print out debug text file
DEBUG = 1

file_name   = 'extrap_data/11759_ccd3/11759_32x32.png'

# Parameter
Threshold   = np.int32(1)
MaxRad      = np.int32(10)

# Setup input file
IMG_rgb = imread(file_name)
IMG = array( IMG_rgb[:,:,0] )

# Get image data
Lx = np.int32( IMG.shape[0] )
Ly = np.int32( IMG.shape[1] )

# Allocate memory
# Max box smoothing stencil
BOX = np.zeros((Lx, Ly), dtype=np.float32)
# normalized array
NORM = np.zeros((Lx, Ly), dtype=np.float32)
# output array
OUT = np.zeros((Lx, Ly), dtype=np.float32)

# Execution configuration
TPBx	= int( 32 )       
TPBy    = int( 32 )           
nBx     = int( Ly/TPBx )
nBy     = int( Lx/TPBy ) 


#########
## SMOOTHING KERNEL ##   
# First part of algorithm performs adaptive smoothing
#
#########
kernel_smooth_source = \
"""
    __global__ void smoothingFilter(int Lx, int Ly, int Threshold, int MaxRad, 
        float* IMG, float* BOX, float* NORM)
    {
    // Indexing
    int tid = threadIdx.x;
    int tjd = threadIdx.y;
    int i = blockIdx.x * blockDim.x + tid;
    int j = blockIdx.y * blockDim.y + tjd;
    int stid = tjd * blockDim.x + tid;
    int gtid = j * Ly + i;  
    
    // Smoothing params
    float qq    = 1.0;
    float sum   = 0.0;
    float ksum  = 0.0;
    float ss    = qq;

    // Shared memory
    extern __shared__ float s_IMG[];
    s_IMG[stid] = IMG[gtid];
    __syncthreads();
    
    // Compute all pixels except for image border
	if ( i >= 0 && i < Ly && j >= 0 && j < Lx )
	{
	    // Continue until parameters are met
	    while (sum < Threshold && qq < MaxRad)
	    {
	        ss = qq;
	        sum = 0.0;
	        ksum = 0.0;
	        
            // Normal adaptive smoothing
            for (int ii = -ss; ii < ss+1; ii++)
            {
                for (int jj = -ss; jj < ss+1; jj++)
                {
                    // Smoothing stencil must be within bounds
                    if ( (i-ss >= 0) && (i+ss < Ly) && (j-ss >= 0) && (j+ss < Lx) )
                    {
                            sum += IMG[gtid + ii*Ly + jj];
                            ksum += 1.0;                           
                    }
                }
            }
            qq += 1;
        }
        // Store max size of box stencil
        BOX[gtid] = ss;

        // Determine the normalization for each box
        for (int ii = -ss; ii < ss+1; ii++)
        {
            for (int jj = -ss; jj < ss+1; jj++)
            {
                if ( (i-ss >= 0) && (i+ss < Ly) && (j-ss >= 0) && (j+ss < Lx) )
                {
                    if (ksum != 0)
                    {
                        NORM[gtid + ii*Ly + jj] +=  1.0 / 2;
                    }
                }
            }
        }
	}
	__syncthreads();

    }
    """    
#########
## NORMALIZING KERNEL ##   
# Second part of the algorithm applies smoothing
#
#########
kernel_norm_source = \
"""
    __global__ void normalizeFilter(int Lx, int Ly, float* IMG, float* NORM )
    {
    // Indexing
    int tid = threadIdx.x;
    int tjd = threadIdx.y;
    int i = blockIdx.x * blockDim.x + tid;
    int j = blockIdx.y * blockDim.y + tjd;
    int stid = tjd * blockDim.x + tid;
    int gtid = j * Ly + i;  

    // Shared memory for IMG and NORM
    extern __shared__ float s_IMG[];
    extern __shared__ float s_NORM[];
    s_IMG[stid] = IMG[gtid];
    s_NORM[stid] = NORM[gtid];
    __syncthreads();    

    // Compute all pixels except for image border
	if ( i >= 0 && i < Ly && j >= 0 && j < Lx )
	{
        if (NORM != 0)
        {
            // Access from global memory
            IMG[gtid] /= NORM[gtid];
        }
	}
	__syncthreads();
    }
"""

#########
## OUTPUT KERNEL ##   
# kernel for the last part of the algorithm that creates the output image
#
#########
kernel_out_source = \
"""
    __global__ void outFilter( int Lx, int Ly, float* IMG, float* BOX, float* OUT )
    {
    // Indexing
    int tid = threadIdx.x;
    int tjd = threadIdx.y;
    int i = blockIdx.x * blockDim.x + tid;
    int j = blockIdx.y * blockDim.y + tjd;
    int stid = tjd * blockDim.x + tid;
    int gtid = j * Ly + i;  

    // Smoothing params
    float ss    = BOX[gtid];
    float sum   = 0.0;
    float ksum  = 0.0;

    extern __shared__ float s_IMG[];
    s_IMG[stid] = IMG[gtid];
    __syncthreads();

    // Compute all pixels except for image border
	if ( i >= 0 && i < Ly && j >= 0 && j < Lx )
	{
        for (int ii = -ss; ii < ss+1; ii++)
        {
            for (int jj = -ss; jj < ss+1; jj++)
            {
            if ( (i-ss >= 0) && (i+ss < Ly) && (j-ss >= 0) && (j+ss < Lx) )
                {
                        sum += IMG[gtid + ii*Ly + jj];
                        ksum += 1.0;                   
                }
            }
        }
	}  
    if ( ksum != 0 )
    {
        OUT[gtid] = sum / ksum;
    }
	__syncthreads();
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

# Copy image to device
IMG_device          = gpuarray.to_gpu(IMG)
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
smoothing_kernel(Lx, Ly, Threshold, MaxRad, IMG_device, BOX_device, NORM_device,
    block=( TPBx, TPBy,1 ),  grid=( nBx, nBy ), shared=( smem_size ) )
smth_kernel_stop_time.record()

##########
# This kernel will normalize the image with the value obtained from first kernel
# Normalizing kernel will utilize the NORM and modify the IMG
##########

norm_kernel_start_time.record()
normalize_kernel(Lx, Ly, IMG_device, NORM_device,
    block=( TPBx, TPBy,1 ),  grid=( nBx, nBy ), shared=( smem_size ) )
norm_kernel_stop_time.record()

##########
# This will resmooth the data utilizing the new normalized image
# This kernel will utilize the BOX and IMG_norm and modify the OUT
##########
out_kernel_start_time.record()
out_kernel(Lx, Ly, IMG_device, BOX_device, OUT_device,
    block=( TPBx, TPBy,1 ),  grid=( nBx, nBy ), shared=( smem_size ) )
out_kernel_stop_time.record()

total_stop_time = time.time()

# Copy image to host
IMG_out = OUT_device.get()

# Debug
if(DEBUG):
    BOX_out = BOX_device.get()
    NORM_out = NORM_device.get()
    f = open('debug_gpu.txt', 'w')
    set_printoptions(threshold='nan')
    print >>f,'IMG'
    print >>f, str(IMG).replace('[',' ').replace(']', ' ')
    print >>f,'OUTPUT'
    print >>f, str(IMG_out).replace('[',' ').replace(']', ' ')
    print >>f,'BOX'
    print >>f, str(BOX_out).replace('[',' ').replace(']', ' ')
    print >>f,'NORM'
    print >>f, str(NORM_out).replace('[',' ').replace(']', ' ')
    f.close()

# Print results & save
imsave('{}_smoothed_gpu.png'.format(file_name), IMG_out, cmap=cm.gray, vmin=0, vmax=1)
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

