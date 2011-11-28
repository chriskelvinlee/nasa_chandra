from pylab import *
import numpy as np
import time

# Update img directory to reflect github
file_name = 'input_small/114_ccd7.jpg'
original_image_rgb = imread(file_name)

# Image is black and white so R=B=G
IMG = array( original_image_rgb[:,:,0])

# Get image data
Lx = int32( IMG.shape[0])
Ly = int32( IMG.shape[1])

print "Processing %d x %d image" % (width, height)

total_start_time = time.time()
setup_start_time = time.time()

# Allocate memory
RAD = np.zeros((Lx, Ly), dtype=np.float64)
TOTAL = np.zeros((Lx, Ly), dtype=np.float64)
NORM = np.zeros((Lx, Ly), dtype=np.float64)
OUT = np.zeros((Lx, Ly), dtype=np.float64)

# Parameters
Threshold = 30.0
MaxRad = 30.0

# Array to hold updated values
ww = np.ones((Lx, Ly), dtype=np.float64)

setup_stop_time = time.time()
kernel_start_time = time.time()

# Begin smoothing kernel
for xx in range(Lx):
    for yy in range(Ly):
        qq = 0.0        ##
        sum = 0.0       ##
        ksum = 0.0      ##
        ss = qq         ##

        # Continue until parameters
        while (sum < Threshold) and (qq < MaxRad):
            ss = qq
            sum = 0.0
            ksum = 0.0
            
            # Updated for loops for python
            for ii in xrange( int(-ss), int(ss+1) ):
                for jj in xrange( int(-ss), int(ss+1) ):
                    sum += IMG[xx + ii][yy + jj] * ww[ii + ss][jj + ss]
                    ksum +=(ww[ii + ss][jj + ss])
            qq += 1
            
        # 
        RAD[xx][yy] = ss
        TOTAL[xx][yy] = sum
        
        # Missing mm parameter??
        for ii in xrange( int(-ss), int(ss+1) ):
            for jj in xrange( int(-ss), int(ss+1) ):
                NORM[xx+mm][yy+nn] += (ww[ii+ss][jj+ss])/ksum
#---------------------------------------------------------------

#
for xx in range(Lx):
    for yy in range(Ly):
        IMG[xx][yy] /= NORM[xx][yy]

#---------------------------------------------------------------

#
for xx in range(Lx):
    for yy in range(Ly):
        ss = RAD[xx][yy]
        sum = 0.0
        ksum = 0.0

        #
        for ii in xrange( int(-ss), int(ss+1) ):
            for jj in xrange( int(-ss), int(ss+1) ):
                sum += (IMG[xx+ii][yy+jj]*ww[ii+ss][jj+ss])
                ksum += ww[ii+ss][jj+ss]
        OUT[xx][yy] = sum / ksum

kernel_stop_time = time.time()
total_stop_time = time.time()
#---------------------------------------------------------------

# Save the current image. NOT DONE
imsave('input_small/114_ccd7{0}{1}.jpg'.format('_sharpen', '1'), OUT, cmap=cm.gray, vmin=0, vmax=1)

# Print results & save
print "Total Time: %f"      % (total_stop_time - total_start_time)
print "Setup Time: %f"      % (setup_stop_time - setup_start_time)
print "Kernel Time: %f"     % (kernel_stop_time - kernel_start_time)
