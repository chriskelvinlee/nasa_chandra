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

def parallel_sharpen(filename, num_iterations, rank, size, comm):
    
    # Read image
    file_name          = filename
    original_image_rgb = imread(file_name)
    original_image     = array( original_image_rgb[:,:,0] )

    # Get image data
    height             = np.int32( original_image.shape[0] )
    width              = np.int32( original_image.shape[1] )
    num_pixels         = width * height
    
    # Sharpening Filter Dependencies
    NUM_ITERATIONS     = num_iterations
    EPSILON            = np.float64( 0.07 / 16.0 )

    # Execution configuration
    TPBx	= int( 32 )                # Sweet spot num of threads per block
    TPBy    = int( 32 )                # 32*32 = 1024 max
    nBx     = int( width/TPBx )        # Num of thread blocks
    nBy     = int( height/TPBy ) 

    size=[width, height]
    
    kernel_source = \
    """
    __global__ void sharpeningFilter( float* next_im, float* curr_im, int height, int width, double EPSILON )
    {
        // Indexing
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int gtid = i + j * width;

        // Compute Sharpening
    	if ( i > 0 && i < width-1 && j > 0 && j < height-1 )
    	{
    	    next_im[gtid] = curr_im[gtid] + EPSILON * (
    		    -1*curr_im[gtid - width-1] + -2*curr_im[gtid - width] + -1*curr_im[gtid - width+1] + 
    		    -2*curr_im[gtid-1] + 12*curr_im[gtid] + -2*curr_im[gtid+1] + 
    			-1*curr_im[gtid + width-1] + -2*curr_im[gtid + width] + -1*curr_im[gtid + width+1]);
    	}
    }
    """
    # Initialize kernel
    P3A_kernel = nvcc.SourceModule(kernel_source).get_function("sharpeningFilter")

    total_start_time = time.time()

    # Allocate memory and constants
    smem_size   = int(TPBx*TPBy*4)
    curr_im     = array( original_image )
    next_im     = array( original_image )

    
    benchmark = []

    for iter in range(0, NUM_ITERATIONS):

        # Copy image to device
        curr_im_device = gpuarray.to_gpu(curr_im)
        next_im_device = gpuarray.to_gpu(next_im)
    
        # Compute the image's pixel mean/variance on host
        sum = 0
        for i in range(0, height):
        	for j in range(0, width):
        		sum += curr_im[i,j]
        mean = sum / num_pixels
    
        variance_start_time = time.time()
    
        variance_sum = 0
        for i in range(0, height):
        	for j in range(0, width):
        		variance_sum += ( curr_im[i,j] - mean )**2
        variance = variance_sum / num_pixels

        variance_stop_time = time.time()
    
        # Append the mean/variance
        benchmark.append([mean, variance])

        # Run the CUDA kernel with the appropriate inputs and outputs
        P3A_kernel(next_im_device, curr_im_device, height, width, EPSILON, block=( TPBx, TPBy,1 ),  grid=( nBx,nBy ) )	

        # Copy image to host
        curr_im = curr_im_device.get()
        next_im = next_im_device.get()
    
        # Swap references to the images
        curr_im, next_im = next_im, curr_im
    
        # Save image
        # imsave('iter{0:02d}.png'.format(iter), curr_im, cmap=cm.gray, vmin=0, vmax=1)

    total_stop_time = time.time()
    #imsave('Harvard_Sharpened_GPU_MPI_{}'.format(filename), curr_im, cmap=cm.gray, vmin=0, vmax=1)
  
    # Print results & save
    total_time  = (total_stop_time - total_start_time)
    setup_time  = (setup_stop_time - setup_start_time)
    #var_time    = (variance_start_time.time_till(variance_stop_time) * 1e-3)
    #ker_time    = (kernel_start_time.time_till(kernel_stop_time) * 1e-3)

    results = [size, benchmark, [total_time, setup_time]]

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

filenames = ['Harvard_Small.png', 'Harvard_Medium.png', 'Harvard_Large.png', 'Harvard_Huge.png']
NUM_ITERATIONS = 20

# Send filenames as data
parallel_sharpen(filenames[rank], NUM_ITERATIONS, rank, size, comm)


# Print rank, mean, variance
if rank == 0:
    for k in range(0, size):
        results = comm.recv(source = k) # Receive data from processes
        print "***Rank %d***" % k
        #print results
        print "Processing %d x %d image" % (results[0][0], results[0][1])
        for iter in range(0, NUM_ITERATIONS):
             print "Iteration %d:  Mean = %f,  Var = %f" % (iter, results[1][iter][0], results[1][iter][1])
        print "Total Time: %f"      % results[2][0]
        print "Setup Time: %f"      % results[2][1]
        #print "Variance Time: %f"   % results[2][2]
        #print "Kernel Time: %f"     % results[2][3]
        print "\n"
    

########################
#  END OF MPI CODE
########################

