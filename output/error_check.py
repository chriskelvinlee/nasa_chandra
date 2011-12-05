#!/usr/bin/env python
# encoding: utf-8
"""
error_check.py

Created by Christopher K. Lee on 2011-12-03.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""
import os
from pylab import *


output_serial = ['extrap_data/output/serial/11759_ccd3_32x32_smoothed_serial.png',
                'extrap_data/output/serial/11759_ccd3_64x64_smoothed_serial.png',
                'extrap_data/output/serial/11759_ccd3_128x128_smoothed_serial.png',
                'extrap_data/output/serial/11759_ccd3_256x256_smoothed_serial.png',
                'extrap_data/output/serial/11759_ccd3_512x512_smoothed_serial.png']
                
output_parallel = ['extrap_data/output/parallel/11759_ccd3/11759_32x32_smoothed_gpu.png',
                'extrap_data/output/parallel/11759_ccd3/11759_64x64_smoothed_gpu.png',
                'extrap_data/output/parallel/11759_ccd3/11759_128x128_smoothed_gpu.png',
                'extrap_data/output/parallel/11759_ccd3/11759_256x256_smoothed_gpu.png',
                'extrap_data/output/parallel/11759_ccd3/11759_512x512_smoothed_gpu.png']
                
f = open('rel_error.txt', 'w')

for k in xrange(0,5):

    # Read in serial version, make sure precomputed dimension are same
    serial_file   = output_serial[k]
    serial_im_rgb = imread(serial_file)                 # Must be same dimensions     
    serial_im     = array( serial_im_rgb[:,:,0] )       # Compare with parallel_im
    
    # Read in parallel version for proper benchmark
    parallel_file   = output_parallel[k]
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
    rel_error = rel_error/num_pixels/100                   # According to forum 1e-6 for mean/variance     

    # Debug

    set_printoptions(threshold='nan')
    print >>f, "Error check for %d x %d image" % (width, height)
    #print >>f,'Serial'
    #print >>f, str(serial_file)
    #print >>f,'Parallel'
    #print >>f, str(parallel_file)
    print >>f, "Relative Pixel Error  = %e" % rel_error  
    if rel_error > 1e-4:            # Use 1e-4 for rel_error for entire image
        print >>f, "***LARGE ERROR - POSSIBLE FAILURE!***"
    print >>f, "\n"
f.close()

          