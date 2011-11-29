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

def parallel_smooth(filename, rank, size, comm):
    
    # Read image
    file_name          = filename
    IMG_rgb = imread(file_name)
    IMG     = array( IMG_rgb[:,:,0] )
    
    # Get image data
    Lx             = np.int32( IMG.shape[0] )
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
    nBx     = int( Ly/TPBx )        # Num of thread blocks
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
        __global__ void normalizeFilter( float* next_im, float* curr_im, int Lx, int Ly )
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
    m_array = gpuarray.to_gpu(curr_im)
    sum = gpuarray.sum(m_array).get()
        
    kernel_start_time = cu.Event()
    kernel_stop_time = cu.Event()
        
    kernel_start_time.record()

    ## TO DO ##   
    # I am not really sure if this is how you would run multiple kernels
    # Need to ensure that the input/output variables are correct to pass to 
    # the next kernel
    #########
    
    # Run the CUDA kernel with the appropriate inputs and outputs
    ## TO DO ##   
    # This kernel will utilize the IMG and modify the RAD and NORM
    #########
    smoothing_kernel(next_im_device, curr_im_device, Lx, Ly, block=( TPBx, TPBy,1 ),  grid=( nBx, nBy ), shared=( smem_size ) )
    # Run the CUDA kernel with the appropriate inputs and outputs
    ## TO DO ##   
    # This kernel will utilize the NORM and modify the IMG
    #########
    normalize_kernel(next_im_device, curr_im_device, Lx, Ly, block=( TPBx, TPBy,1 ),  grid=( nBx, nBy ), shared=( smem_size ) )
    # Run the CUDA kernel with the appropriate inputs and outputs
    ## TO DO ##   
    # This kernel will utilize the RAD and IMG and modify the OUT
    #########
    out_kernel(next_im_device, curr_im_device, Lx, Ly, block=( TPBx, TPBy,1 ),  grid=( nBx, nBy ), shared=( smem_size ) )
    
    kernel_stop_time.record()
    kernel_stop_time.synchronize()
        
    # Copy image to host
    curr_im = curr_im_device.get()
    
    total_stop_time = time.time()
    imsave('{}_smoothed.png'.format(filename), curr_im, cmap=cm.gray, vmin=0, vmax=1)
    
    # Print results & save
    total_time  = (total_stop_time - total_start_time)
    setup_time  = (setup_stop_time - setup_start_time)
    ker_time    = (kernel_start_time.time_till(kernel_stop_time) * 1e-3)
    
    results = [total_time, setup_time, ker_time]
    
    ### SEND
    comm.send(results, dest = 0)

########################
#  BEGINING OF MPI CODE
########################

# Initialize MPI constants
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print "Hello from process %d of %d" % (rank,size)

filenames = ['114_ccd7.jpg', '321_ccd7.jpg', '7417_ccd3.jpg', '11759_ccd3.jpg']

# Send filenames as data
parallel_smooth(filenames[rank], rank, size, comm)


# Print rank, mean, variance
if rank == 0:
    for k in range(0, size):
        results = comm.recv(source = k) # Receive data from processes
        print "***Rank %d***" % k
        #print results
        print "Total Time: %f"      % results[0]
        print "Setup Time: %f"      % results[1]
        print "Kernel Time: %f"     % results[2]
        print "\n"


########################
#  END OF MPI CODE
########################

