import numpy as np
from pylab import *
import pycuda.autoinit
import pycuda.driver as cu
import pycuda.compiler as nvcc
from pycuda import gpuarray
import time

###
#### CALCULATION ERROR BOOLEAN
###
calculate_error = True

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

# Read image
file_name          = "Harvard_Small.png"
original_image_rgb = imread(file_name)
original_image     = array( original_image_rgb[:,:,0] )

# Get image data
height     = np.int32( original_image.shape[0] )
width      = np.int32( original_image.shape[1] )
num_pixels = width * height

# Sharpening Filter Dependencies
NUM_ITERATIONS  = 20
EPSILON         = np.float64( 0.07 / 16.0 )

# Execution configuration
TPBx	= int( 32 )                # Sweet spot num of threads per block
TPBy    = int( 32 ) 
nBx     = int( width/TPBx )        # Sweet spot num of thread blocks
nBy     = int( height/TPBy ) 

print "Processing %d x %d image" % (width, height)

total_start_time = time.time()
#setup_start_time = time.time()

# Allocate memory and constants
curr_im = array( original_image )
next_im = array( original_image )

for iter in range(0, NUM_ITERATIONS):

    # Copy image to device
    curr_im_device = gpuarray.to_gpu(curr_im)
    next_im_device = gpuarray.to_gpu(next_im)
    
    #setup_stop_time = time.time()
    
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
    
    # Print the mean/variance
    print "Iteration %d:  Mean = %f,  Variance = %f" % (iter, mean, variance)
    
    kernel_start_time = cu.Event()
    kernel_stop_time = cu.Event()

    kernel_start_time.record()

    # Run the CUDA kernel with the appropriate inputs and outputs
    P3A_kernel(next_im_device, curr_im_device, height, width, EPSILON, block=( TPBx, TPBy,1 ),  grid=( nBx,nBy ) )	

    kernel_stop_time.record()
    kernel_stop_time.synchronize()

    # Copy image to host
    curr_im = curr_im_device.get()
    next_im = next_im_device.get()
    
    # Swap references to the images
    curr_im, next_im = next_im, curr_im
    
    # Save image
    # imsave('iter{0:02d}.png'.format(iter), curr_im, cmap=cm.gray, vmin=0, vmax=1)

total_stop_time = time.time()



###
####  CALCULATE ERROR BY AVERAGE PIXEL RELATIVE ERROR
###
if calculate_error == True:
    # Save parallel version
    imsave('Harvard_Sharpened_GPU_A.png'.format(NUM_ITERATIONS), curr_im, cmap=cm.gray, vmin=0, vmax=1)
    
    # Read in parallel version for proper benchmark
    parallel_file   = "Harvard_Sharpened_GPU_A.png"     
    parallel_im_rgb = imread(parallel_file)                  
    parallel_im     = array( parallel_im_rgb[:,:,0] )   
    
    # Read in serial version, make sure precomputed dimension are same
    serial_file   = "Harvard_Sharpened_CPU_serial.png"  # Precomputed serial x20 iterations
    serial_im_rgb = imread(serial_file)                 # Must be same dimensions     
    serial_im     = array( serial_im_rgb[:,:,0] )       # Compare with parallel_im

    # Iterate through all pixels, find rel_error
    rel_error = 0.0
    for i in range( 1,height-1 ):
        for j in range( 1,width-1 ):
            if not (serial_im[i,j] == 0):               # Prevent pixel = 0 -> rel_error = inf 
                rel_error += (abs( parallel_im[i,j] - serial_im[i,j] ) / abs( serial_im[i,j] ))
    
    # Avgerage rel_error
    rel_error = rel_error/num_pixels                    # According to forum 1e-6 for mean     
    print "Relative Pixel Error  = %e" % rel_error           
    if rel_error > 1e-3:                                # Use 1e-3 for rel_error for entire image
    	print "***LARGE ERROR - POSSIBLE FAILURE!***"

  
# Print results & save
print "Total Time: %f"      % (total_stop_time - total_start_time)
#print "Setup Time: %f"      % (setup_stop_time - setup_start_time)
print "Variance Time: %f"   % (variance_stop_time - variance_start_time)
print "Kernel Time: %f"     % (kernel_start_time.time_till(kernel_stop_time) * 1e-3)
