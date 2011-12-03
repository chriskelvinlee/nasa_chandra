#!/usr/bin/env python
# encoding: utf-8
"""
error_check.py

Created by Christopher K. Lee on 2011-12-03.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""
import os
from pylab import *



# Read in serial version, make sure precomputed dimension are same
serial_file   = "extrap_data/11759_ccd3/11759_32x32_smoothed_serial.png"  # Precomputed serial x20 iterations
serial_im_rgb = imread(serial_file)                 # Must be same dimensions     
serial_im     = array( serial_im_rgb[:,:,0] )       # Compare with parallel_im
    
# Read in parallel version for proper benchmark
parallel_file   = "extrap_data/11759_ccd3/11759_32x32_smoothed_gpu.png"     
parallel_im_rgb = imread(parallel_file)                  
parallel_im     = array( parallel_im_rgb[:,:,0] )   

# Find width and height
width = int32( serial_im_rgb.shape[0])
height = int32( serial_im_rgb.shape[1])
num_pixels = width*height

# Iterate through all pixels, find rel_error
rel_error = 0.0
for i in range( 1,height-1 ):
    for j in range( 1,width-1 ):
        if not (serial_im[i,j] == 0):               # Prevent pixel = 0 -> rel_error = inf 
            rel_error += (abs( parallel_im[i,j] - serial_im[i,j] ) / abs( serial_im[i,j] ))

# Avgerage rel_error
rel_error = rel_error/num_pixels                   # According to forum 1e-6 for mean/variance     

# Debug
f = open('rel_error.txt', 'w')
set_printoptions(threshold='nan')
print >>f,'Serial'
print >>f, str(serial_file)
print >>f,'Parallel'
print >>f, str(parallel_file)
print >>f, "Relative Pixel Error  = %e" % rel_error  
if rel_error > 1e-3:            # Use 1e-3 for rel_error for entire image
    print >>f, "***LARGE ERROR - POSSIBLE FAILURE!***"
f.close()

          