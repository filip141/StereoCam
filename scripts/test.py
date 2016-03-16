# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

cap = cv2.VideoCapture(2)
cap1 = cv2.VideoCapture(1)

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

while(1):

    # Take each frame
    _, frame = cap.read()
    _, frame1 = cap1.read()

    rows,cols,dim = frame.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
    frame = cv2.warpAffine(frame,M,(cols,rows))
    frame1 = cv2.warpAffine(frame1,M,(cols,rows))

    imgL = frame  # downscale images for faster processing
    imgR = frame1

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM(minDisparity = min_disp,
        numDisparities = num_disp,
        SADWindowSize = window_size,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        disp12MaxDiff = 1,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        fullDP = False
    )

    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    cv2.imshow('left', imgL)
    cv2.imshow('right', imgR)
    cv2.imshow('disparity', (disp-min_disp)/num_disp)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
###NEW CODE
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
