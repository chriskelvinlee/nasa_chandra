from pylab import *
import time

# Read image
file_name          = "Harvard_Small.png"
original_image_rgb = imread(file_name)
# BW images have R=G=B. Extract the R-value
original_image     = array( original_image_rgb[:,:,0] )

# Get image data
height     = int32( original_image.shape[0] )
width      = int32( original_image.shape[1] )
num_pixels = width * height

print "Processing %d x %d image" % (width, height)

total_start_time = time.time()
setup_start_time = time.time()

# Allocate memory and constants
curr_im         = array( original_image )
next_im         = array( original_image )

setup_stop_time = time.time()
    
NUM_ITERATIONS  = 20
EPSILON         = 0.07 / 16.0

for iter in range(0, NUM_ITERATIONS):
	
    # Save the current image. Clamp the values between 0.0 and 1.0
    #imsave('iter{0:02d}.png'.format(iter), curr_im, cmap=cm.gray, vmin=0, vmax=1)

    # Compute the image's pixel mean and variance
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

    # Print the variance
    print "Iteration %d:  Mean = %f,  Variance = %f" % (iter, mean, variance)

    kernel_start_time = time.time()
    
    # Compute Sharpening
    for i in range(1, height-1):
    	for j in range(1, width-1):
		
    		next_im[i,j] = curr_im[i,j] + EPSILON * (
    			   -1*curr_im[i-1,j-1] + -2*curr_im[i-1,j] + -1*curr_im[i-1,j+1]
    			 + -2*curr_im[i  ,j-1] + 12*curr_im[i  ,j] + -2*curr_im[i  ,j+1]
    			 + -1*curr_im[i+1,j-1] + -2*curr_im[i+1,j] + -1*curr_im[i+1,j+1])

    kernel_stop_time = time.time()
	
    # Swap references to the images
    curr_im, next_im = next_im, curr_im

total_stop_time = time.time()

# Save the current image. Clamp the values between 0.0 and 1.0
imsave('Harvard_Sharpened_CPU_serial.png'.format(NUM_ITERATIONS), curr_im, cmap=cm.gray, vmin=0, vmax=1)

# Print results & save
print "Total Time: %f"      % (total_stop_time - total_start_time)
print "Setup Time: %f"      % (setup_stop_time - setup_start_time)
print "Variance Time: %f"   % (variance_stop_time - variance_start_time)
print "Kernel Time: %f"     % (kernel_stop_time - kernel_start_time)