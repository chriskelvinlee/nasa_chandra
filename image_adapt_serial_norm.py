from pylab import *
import os
import numpy as np
import time

# Update img directory to reflect github
file_name = 'png_input/114_ccd7_small.png'
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
RAD = np.zeros((Lx, Ly), dtype=np.float64)
NORM = np.zeros((Lx, Ly), dtype=np.float64)
OUT = np.zeros((Lx, Ly), dtype=np.float64)

# Set Parameters
Threshold = 10
MaxRad = 2

setup_stop_time = time.time()
kernel_start_time = time.time()

# Begin smoothing kernel
for xx in range(Lx):
    for yy in range(Ly):
        qq = 0.0        # size of box
        sum = 0.0       # value of the sum
        ksum = 0.0      # value of the kernal sum
        ss = qq         # size of the box around source pixel
        
        # Continue until parameters
        while (sum < Threshold) and (qq < MaxRad):
            ss = qq
            sum = 0.0
            ksum = 0.0
            
            #check for boundary condition
            #else skip to where qq = boundary
            ## I BELIEVE THIS THE LINE BELOW MAY BE INCORRECT. IT ONLY CHECKS THE
            ## RIGHT-END AND BOTTOM-END OF THE IMAGE. IT FORGETS TO CHECK THE
            ## LEFT-END AND TOP-END IS TEH BOX GRID EXCEEDS BOUDND LIMITS. MAY
            ## BE WRONG FOR LINE 73 AS WELL.
            ## LOOK AT C VERSION IN GPU:
            ## if ( i > 0 && i < Ly && j > 0 && j < Lx )
            if((xx + ss < Lx) and (yy + ss < Ly)):
                # Updated for loops for python
                for ii in xrange( int(-ss), int(ss+1) ):
                    for jj in xrange( int(-ss), int(ss+1) ):
                        sum += IMG[xx + ii][yy + jj]
                        ksum += 1.0
            else:
                break;
            qq += 1
            
        # set the size of the radius for the determined box
        RAD[xx][yy] = ss
        
        # Determine the normalization for each box
        # Norm can't be determined from the above loop because it relies on the
        # total ksum value, if placed above the incorrect ksum value will be
        # divided.
        for ii in xrange( int(-ss), int(ss+1) ):
            for jj in xrange( int(-ss), int(ss+1) ):
                if((xx + ss < Lx) and (yy + ss < Ly)):
                    NORM[xx+ii][yy+jj] += 1.0 / ksum
#---------------------------------------------------------------

# Normalize the image
for xx in range(Lx):
    for yy in range(Ly):
        IMG[xx][yy] /= NORM[xx][yy]
        
#---------------------------------------------------------------

# Output file
for xx in range(Lx):
    for yy in range(Ly):
        ss = RAD[xx][yy]
        sum = 0.0
        ksum = 0.0

        #
        for ii in xrange( int(-ss), int(ss+1) ):
            for jj in xrange( int(-ss), int(ss+1) ):
                if((xx + ss < Lx) and (yy + ss < Ly)):
                    sum += IMG[xx+ii][yy+jj]
                    ksum += 1.0
        #check for divide by zero
        if(ksum != 0):
            OUT[xx][yy] = sum / ksum
        else:
            OUT[xx][yy] = 0


kernel_stop_time = time.time()
total_stop_time = time.time()
#---------------------------------------------------------------

# Save the current image.
imsave('{}_serial_smoothed_serial.png'.format(os.path.splitext(file_name)[0]), OUT, cmap=cm.gray, vmin=0, vmax=1)

# Debug
f = open('debug.txt', 'w')
set_printoptions(threshold='nan')
print >>f,'IMG'
print >>f, str(IMG).replace('[',' ').replace(']', ' ')
print >>f,'BOX'
print >>f, str(RAD).replace('[',' ').replace(']', ' ')
print >>f,'NORM'
print >>f, str(NORM).replace('[',' ').replace(']', ' ')
f.close()

# Print results & save
print "Total Time: %f"      % (total_stop_time - total_start_time)
print "Setup Time: %f"      % (setup_stop_time - setup_start_time)
print "Kernel Time: %f"     % (kernel_stop_time - kernel_start_time)
