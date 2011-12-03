from pylab import *
import os
import sys
import numpy as np
import time

total_start_time = time.time()
setup_start_time = time.time()

#Get the input filename from the command line
try:
    file_name = sys.argv[1]; MaxRad = float(sys.argv[2]); Threshold = float(sys.argv[3])
except:
    print "Usage:",sys.argv[0], "infile maxrad threshold"; sys.exit(1)

original_image_rgb = imread(file_name)

# Image is black and white so R=B=G
IMG = array( original_image_rgb[:,:,0])

# Get image data
Lx = int32( IMG.shape[0])
Ly = int32( IMG.shape[1])

print "Processing file %s (%d x %d image)" % (file_name, Lx, Ly)

# Allocate memory
# size of the box needed to reach the threshold value or maxrad value
BOX = np.zeros((Lx, Ly), dtype=np.float64) 
# normalized array
NORM = np.zeros((Lx, Ly), dtype=np.float64)
# output array
OUT = np.zeros((Lx, Ly), dtype=np.float64)

setup_stop_time = time.time()
kernel_start_time = time.time()

# Convolve the image with the gaussian kernel
for xx in range(Lx):
    for yy in range(Ly):
        qq = 1.0        # var to increase size of box
        sum = 0.0       # value of the sum
        ksum = 0.0      # value of the kernal sum
        ss = qq         # size of the box

        # Continue until parameters are met
        while (sum < Threshold) and (qq < MaxRad):
            ss = qq
            # ARE THE FOLLOWING TWO LINES NECESSARY?
            # SUM = O WILL JUST NULL THE ABOVE THRESHOLD REQUIREMENT
            sum = 0.0
            ksum = 0.0
                
            #check for boundary condition
            if((xx + ss < Lx) and (yy + ss < Ly)):
                #create a weighted gaussian sum
                gx, gy = mgrid[-ss:ss+1, -ss:ss+1]
                ww = exp(-(gx**2/float(ss)+gy**2/float(ss)))
                #loop over the box to determine the values
                for ii in xrange( int(-ss), int(ss+1) ):
                    for jj in xrange( int(-ss), int(ss+1) ):
                        sum += IMG[xx + ii][yy + jj] * ww[ii + ss][jj + ss]
                        ksum += ww[ii + ss][jj + ss]
            else:
                break
            qq += 1
        
        # save the size of the box determined to reach the threshold value
        BOX[xx][yy] = ss

        # Determine the normalization for each box
        # Norm can't be determined from the above loop because it relies on the
        # total ksum value, if placed above the incorrect ksum value will be
        # divided.
        if((xx + ss < Lx) and (yy + ss < Ly)):
            #create a weighted gaussian sum
            gx, gy = mgrid[-ss:ss+1, -ss:ss+1]
            ww = exp(-(gx**2/float(ss)+gy**2/float(ss)))
            for ii in xrange( int(-ss), int(ss+1) ):
                for jj in xrange( int(-ss), int(ss+1) ):
                    if(ksum != 0):
                        NORM[xx+ii][yy+jj] += (ww[ii + ss][jj + ss] / ksum)
#---------------------------------------------------------------

# Normalize the image
for xx in range(Lx):
    for yy in range(Ly):
        if(NORM[xx][yy] != 0):
            IMG[xx][yy] /= NORM[xx][yy]
        
#---------------------------------------------------------------

# Output file will be smoothed with the normalized IMG
for xx in range(Lx):
    for yy in range(Ly):
        ss = BOX[xx][yy]
        sum = 0.0
        ksum = 0.0
        
        if((xx + ss < Lx) and (yy + ss < Ly)):
            #create a weighted gaussian sum for the specific BOX size
            gx, gy = mgrid[-ss:ss+1, -ss:ss+1]
            ww = exp(-(gx**2/float(ss)+gy**2/float(ss)))
            for ii in xrange( int(-ss), int(ss+1) ):
                for jj in xrange( int(-ss), int(ss+1) ):
                    sum += (IMG[xx+ii][yy+jj] * ww[ii + ss][jj + ss])
                    ksum += ww[ii + ss][jj + ss]
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
