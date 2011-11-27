from pylab import *
import numpy as np

#Read Image
file_name = "114/114_ccd7.jpg"
original_image_rgb = imread(file_name)

# Image is black and white so R=B=G
IMG = array( original_image_rgb[:,:,0])

Lx = int32( IMG.shape[0])
Ly = int32( IMG.shape[1])

RAD = np.zeros((Lx, Ly), dtype=np.float64)
TOTAL = np.zeros((Lx, Ly), dtype=np.float64)
NORM = np.zeros((Lx, Ly), dtype=np.float64)
OUT = np.zeros((Lx, Ly), dtype=np.float64)

Threshold = 30.0
MaxRad = 30.0

ww = np.ones((Lx, Ly), dtype=np.float64)

for xx in range(Lx):
    for yy in range(Ly):
        qq = 0.0
        sum = 0.0
        ksum = 0.0
        ss = qq

        while (sum < Threshold) and (qq < MaxRad):
            ss = qq
            sum = 0.0
            ksum = 0.0
            
            for (ii = -ss; ii <= ss; ii++):
                for (jj = -ss; jj <= ss; jj++):
                    sum += IMG[xx + ii][yy + jj] * ww[ii + ss][jj + ss]
                    ksum +=(ww[ii + ss][jj + ss])
            qq += 1
        
        RAD[xx][yy] = ss
        TOTAL[xx][yy] = sum

        for(ii = -ss; ii <= ss; ii++):
            for(jj = -ss; jj <= ss; jj++):
                NORM[xx+mm][yy+nn] += (ww[ii+ss][jj+ss])/ksum
#---------------------------------------------------------------

for xx in range(Lx):
    for yy in range(Ly):
        IMG[xx][yy] /= NORM[xx][yy]

#---------------------------------------------------------------
for xx in range(Lx):
    for yy in range(Ly):
        ss = RAD[xx][yy]
        sum = 0.0
        ksum = 0.0

        for (ii = -ss; ii <= ss; ii++):
            for(jj=-ss; jj <= ss; jj++):
                sum += (IMG[xx+ii][yy+jj]*ww[ii+ss][jj+ss])
                ksum += ww[ii+ss][jj+ss]
        OUT[xx][yy = sum / ksum
#---------------------------------------------------------------
print "Processing %d x %d image" % (width, height)

