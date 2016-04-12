import cv2
import numpy as np
from matplotlib import pyplot as plt

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)

# Load previously saved data
with np.load('disortion_params.npz') as X:
    left_maps = X["left_maps"]
    right_maps = X["right_maps"]

with np.load('stereo_tune.npz') as X:
    window_size = X["window_size"]
    min_disp = 0
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        preFilterCap = X["pre_filter"],
        numDisparities = num_disp,
        blockSize = window_size,
        uniquenessRatio = X["unique"],
        speckleWindowSize = X["speckle_win"],
        speckleRange = X["speckle_range"]*16,
        disp12MaxDiff = X["disp12"],
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        # fullDP = False
    )

cap1 = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(1)
cap1.set(3, 320)
cap1.set(4, 240)
cap2.set(3, 320)
cap2.set(4, 240)


cam_sources = [0,1]



while(1):


    for i in range(0,5):
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

    frame_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
    frame_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
    frame_l[frame_l == 0] = cv2.mean(frame_l)[0]
    frame_r[frame_r == 0] = cv2.mean(frame_r)[0]

    # fig, ax = plt.subplots()
    # im = ax.imshow(frame_l, interpolation='none')
    # ax.format_coord = Formatter(im)
    # plt.show()

    disp = stereo.compute(frame_l, frame_r).astype(np.float32) / 16.0

    ret,thresh1 = cv2.threshold((disp-min_disp)/num_disp,0.35,1.0,cv2.THRESH_BINARY)
    mask = cv2.erode(thresh1, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2).astype(np.uint8)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    areas = [cv2.contourArea(c) for c in cnts]
    max_index = np.argmax(areas)
    cnt=cnts[max_index]

    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(rect_frames[0],(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("disparity",(disp-min_disp)/num_disp)
    cv2.imshow('tr',thresh1)
    cv2.imshow('frame_left',rect_frames[0])
    cv2.imshow('frame_right',rect_frames[1])

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()