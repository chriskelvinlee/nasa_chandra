from pylab import *
import os
import numpy as np
import time

total_start_time = time.time()
setup_start_time = time.time()

# Update img directory to reflect github
file_name = 'input_small/321_ccd7.jpg'
original_image_rgb = imread(file_name)

# Image is black and white so R=B=G
IMG = array( original_image_rgb[:,:,0])

# Get image data
Lx = int32( IMG.shape[0])
Ly = int32( IMG.shape[1])

print "Processing file %s (%d x %d image)" % (file_name, Lx, Ly)

# Set Parameters
Threshold = 30
MaxRad = 3.0

#create a weighted gaussian sum
gx, gy = mgrid[-MaxRad:MaxRad+1, -MaxRad:MaxRad+1]
ww = exp(-(gx**2/float(MaxRad)+gy**2/float(MaxRad)))

# Allocate memory
NORM = np.zeros((Lx, Ly), dtype=np.float64)
OUT = np.zeros((Lx, Ly), dtype=np.float64)

setup_stop_time = time.time()
kernel_start_time = time.time()

# Begin smoothing kernel
for xx in range(Lx):
    for yy in range(Ly):
        sum = 0.0       # value of the sum
        ksum = 0.0      # value of the kernal sum

        #check for boundary condition
        #else skip to where qq = boundary
        if((xx + MaxRad < Lx) and (yy + MaxRad < Ly)):
            for ii in xrange( int(-MaxRad), int(MaxRad+1) ):
                for jj in xrange( int(-MaxRad), int(MaxRad+1) ):
                    sum += IMG[xx + ii][yy + jj] * ww[ii + MaxRad][jj + MaxRad]
                    ksum += ww[ii + MaxRad][jj + MaxRad]

        # Determine the normalization for each box
        # Norm can't be determined from the above loop because it relies on the
        # total ksum value, if placed above the incorrect ksum value will be
        # divided.
        for ii in xrange( int(-MaxRad), int(MaxRad+1) ):
            for jj in xrange( int(-MaxRad), int(MaxRad+1) ):
                if((xx + MaxRad < Lx) and (yy + MaxRad < Ly)):
                    NORM[xx+ii][yy+jj] += (ww[ii + MaxRad][jj + MaxRad] / ksum)
#---------------------------------------------------------------

# Normalize the image
for xx in range(Lx):
    for yy in range(Ly):
        IMG[xx][yy] /= NORM[xx][yy]
        
#---------------------------------------------------------------

# Output file
for xx in range(Lx):
    for yy in range(Ly):
        sum = 0.0
        ksum = 0.0
        
        #
        for ii in xrange( int(-MaxRad), int(MaxRad+1) ):
            for jj in xrange( int(-MaxRad), int(MaxRad+1) ):
                if((xx + MaxRad < Lx) and (yy + MaxRad < Ly)):
                    sum += (IMG[xx+ii][yy+jj] * ww[ii + MaxRad][jj + MaxRad])
                    ksum += ww[ii + MaxRad][jj + MaxRad]
        #check for divide by zero
        if(ksum != 0):
            OUT[xx][yy] = sum / ksum
        else:
            OUT[xx][yy] = 0



kernel_stop_time = time.time()
total_stop_time = time.time()
#---------------------------------------------------------------

# Save the current image.
imsave('{}_serial_smoothed.png'.format(os.path.splitext(file_name)[0]), OUT, cmap=cm.gray, vmin=0, vmax=1)

# Print results & save
print "Total Time: %f"      % (total_stop_time - total_start_time)
print "Setup Time: %f"      % (setup_stop_time - setup_start_time)
print "Kernel Time: %f"     % (kernel_stop_time - kernel_start_time)
