from pylab import *
import os
import numpy as np
import time

#Debug value, if 1 print out debug text file
DEBUG = 1

# Update img directory to reflect github
file_name = 'extrap_data/11759_ccd3/11759_32x32.png'
original_image_rgb = imread(file_name)

# Image is black and white so R=B=G
IMG = array( original_image_rgb[:,:,0])

# Get image data
Lx = int32( IMG.shape[0])
Ly = int32( IMG.shape[1])

print "Processing %d x %d image" % (Lx, Ly)

total_start_time = time.time()
setup_start_time = time.time()

# Allocate memory
# size of the box needed to reach the threshold value or maxrad value
BOX = np.zeros((Lx, Ly), dtype=np.float32) 
# normalized array
NORM = np.zeros((Lx, Ly), dtype=np.float32)
# output array
OUT = np.zeros((Lx, Ly), dtype=np.float32)

# Set Parameters
Threshold = np.int32(1)
MaxRad = np.int32(10)

setup_stop_time = time.time()
kernel_start_time = time.time()

# Begin smoothing kernel
for xx in range(Lx):
    for yy in range(Ly):
        qq = 1.0        # size of box
        sum = 0.0       # value of the sum
        ksum = 0.0      # value of the kernal sum
        ss = qq         # size of the box around source pixel
        
        # Continue until parameters
        while (sum < Threshold) and (qq < MaxRad):
            ss = qq
            sum = 0.0
            ksum = 0.0
            
            #check for boundary condition
            for ii in xrange( int(-ss), int(ss+1) ):
                for jj in xrange( int(-ss), int(ss+1) ):
                    if((xx + ss < Lx) and (xx - ss >= 0) and (yy + ss < Ly) and (yy - ss >=0)):                        
                        sum += IMG[xx + ii][yy + jj]
                        ksum += 1.0
            qq += 1
            
        # save the size of the box determined to reach the threshold value
        BOX[xx][yy] = ss

        # Determine the normalization for each box
        # Norm can't be determined from the above loop because it relies on the
        # total ksum value, if placed above the incorrect ksum value will be
        # divided.
        for ii in xrange( int(-ss), int(ss+1) ):
            for jj in xrange( int(-ss), int(ss+1) ):
                if((xx + ss < Lx) and (xx - ss >= 0) and (yy + ss < Ly) and (yy - ss >=0)):      
                    if(ksum != 0):
                        NORM[xx+ii][yy+jj] += 1.0 / ksum
#---------------------------------------------------------------

# Normalize the image
for xx in range(Lx):
    for yy in range(Ly):
        if(NORM[xx][yy] != 0):
            IMG[xx][yy] /= NORM[xx][yy]
        
#---------------------------------------------------------------

# Output file
for xx in range(Lx):
    for yy in range(Ly):
        ss = BOX[xx][yy]
        sum = 0.0
        ksum = 0.0

        #resmooth with normalized IMG
        for ii in xrange( int(-ss), int(ss+1) ):
            for jj in xrange( int(-ss), int(ss+1) ):
                if((xx + ss < Lx) and (xx - ss >= 0) and (yy + ss < Ly) and (yy - ss >=0)):      
                    sum += IMG[xx+ii][yy+jj]
                    ksum += 1.0
        #check for divide by zero
        if(ksum != 0):
            OUT[xx][yy] = sum / ksum


kernel_stop_time = time.time()
total_stop_time = time.time()
#---------------------------------------------------------------

# Save the current image.
imsave('{}_smoothed_serial.png'.format(os.path.splitext(file_name)[0]), OUT, cmap=cm.gray, vmin=0, vmax=1)

# Debug
if(DEBUG):
    f = open('debug_serial.txt', 'w')
    set_printoptions(threshold='nan')
    print >>f,'IMG'
    print >>f, str(IMG).replace('[',' ').replace(']', ' ')
    print >>f,'OUTPUT'
    print >>f, str(OUT).replace('[',' ').replace(']', ' ')
    print >>f,'BOX'
    print >>f, str(BOX).replace('[',' ').replace(']', ' ')
    print >>f,'NORM'
    print >>f, str(NORM).replace('[',' ').replace(']', ' ')
    f.close()

# Print results & save
print "Total Time: %f"      % (total_stop_time - total_start_time)
print "Setup Time: %f"      % (setup_stop_time - setup_start_time)
print "Kernel Time: %f"     % (kernel_stop_time - kernel_start_time)
