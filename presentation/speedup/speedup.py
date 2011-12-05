#!/usr/bin/env python

from pylab import *

time_serial = [5.379504, 
               23.474275,
               99.412917, 
               382.463874, 
               1544.600379,
               6176.98505292,
               24709.00347468, 
               98837.07716172, 
               395349.37190988]

time_parallel = [0.053993,
                 0.054018,
                 0.050647,
                 0.047445,
                 0.043356,
                 0.039274,
                 0.110894,
                 0.432924,
                 1.652598]

speed_up = [0,0,0,0,0,0,0,0,0]

pos = arange(9) + .5

index = 0
for x in time_parallel: 
    speed_up[index] = time_serial[index]/x
    index += 1 

print speed_up

figure(1)
bar(pos, speed_up, align='center', log='true')
xticks(pos, ('32', '64', '128', '256', '512', '1024', '2048', '4096', '8192'))

title('Speed Up vs Image Size')
grid(True)
xlabel('Image Size (pixel x pixel)')
ylabel('Speed Up (log scale)')


show()
