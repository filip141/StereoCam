import cv2
import numpy as np
from matplotlib import pyplot as plt

win_size = 0
prefilcap = 0
uniq = 0
sws = 0
sr = 0
dmd = -1

def set_win_size(arg):
    global win_size
    win_size = arg

def set_min_prefilter(arg):
    global prefilcap
    prefilcap = arg

def set_uniq(arg):
    global uniq
    uniq = arg

def set_sws(arg):
    global sws
    sws = arg

def set_sr(arg):
    global sr
    sr = arg

def set_dmd(arg):
    global dmd
    dmd = arg



# Load previously saved data
with np.load('disortion_params.npz') as X:
    left_maps = X["left_maps"]
    right_maps = X["right_maps"]

cap1 = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(1)
cap1.set(3, 320)
cap1.set(4, 240)
cap2.set(3, 320)
cap2.set(4, 240)


cam_sources = [0,1]

cv2.imshow('disparity',np.zeros((100,100)))
cv2.createTrackbar('win_size', 'disparity', 3, 11, set_win_size)
cv2.createTrackbar('prefilter', 'disparity', 0, 500, set_min_prefilter)
cv2.createTrackbar('unique', 'disparity', 0, 30, set_uniq)
cv2.createTrackbar('sws', 'disparity', 0, 200, set_sws)
cv2.createTrackbar('sr', 'disparity', 0, 5, set_sr)
cv2.createTrackbar('disp12', 'disparity', -1, 5, set_dmd)

while(1):

    # Take each frame
    _, frame_l = cap1.read()
    _, frame_r = cap2.read()

    ## Frame sizes
    rows,cols,dim = frame_l.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
    frame_l = cv2.warpAffine(frame_l,M,(cols,rows))
    frame_r = cv2.warpAffine(frame_r,M,(cols,rows))

    frames = [frame_l, frame_r]
    maps = [left_maps, right_maps]
    rect_frames = [0,0]
    for src in cam_sources:
        map1_conv = np.array([ [ [ coef for coef in y ] for y in x] for x in maps[src][0]])
        map2_conv = np.array([ [ y for y in x] for x in maps[src][1]])
        rect_frames[src] = cv2.remap(frames[src], map1_conv, map2_conv, cv2.INTER_LANCZOS4)

    # Undistorted images
    frame_l = rect_frames[0]
    frame_r = rect_frames[1]

    # disparity range is tuned for 'aloe' image pair
    window_size = win_size
    min_disp = 0
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        preFilterCap = prefilcap,
        numDisparities = num_disp,
        blockSize = window_size,
        uniquenessRatio = uniq,
        speckleWindowSize = sws,
        speckleRange = sr*16,
        disp12MaxDiff = dmd,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        # fullDP = False
    )
    frame_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
    frame_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
    disp = stereo.compute(frame_l, frame_r).astype(np.float32) / 16.0

    cv2.imshow('frame_left',frame_l)
    cv2.imshow('frame_right',frame_r)
    cv2.imshow('disparity',(disp-min_disp)/num_disp)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()