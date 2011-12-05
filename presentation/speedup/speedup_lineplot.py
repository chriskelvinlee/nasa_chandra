#!/usr/bin/env python

from pylab import *
import numpy as np
import matplotlib.pyplot as plt

N = 9
ind = arange(N) + 0.3  # the x locations for the groups
width = 0.2  

fig = plt.figure()
ax = fig.add_subplot(111)

plot([ 0.3,  1.3,  2.3,  3.3,  4.3,  5.3,  6.3,  7.3,  8.3],
        [2.00, 2.64, 3.29, 3.91, 4.55, 5.20, 5.35, 5.36, 5.38], 'go-',  label='line 2')

plot([ 0.3,  1.3,  2.3,  3.3,  4.3,  5.3,  6.3,  7.3,  8.3],
        [2.41, 3.12, 3.32, 3.95,4.67, 5.14, 5.30 ,5.31 ,5.34], 'bo-',  label='line 2')      


ax.set_xticks(ind)
ax.set_xticklabels( ('32', '64', '128', '256', '512', '1024', '2048', '4096', '8192') )
ax.set_ylabel('Speedup (log)')
ax.set_xlabel('Image Size (pixel x pixel)')
ax.set_title('Speedup of Adaptive Smoothing (CPU/CUDA)')
ax.legend( ('11759_ccd3', '11759'), 'center right' )
vals = arange(1, 7, 1)
yticks(vals, ['%1.1f' % val for val in vals])


plt.savefig('speedup_lineplot.png')