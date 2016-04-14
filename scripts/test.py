import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema


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
    Q = X["Q"]

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

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)
cap1.set(3, 320)
cap1.set(4, 240)
cap2.set(3, 320)
cap2.set(4, 240)


cam_sources = [0,1]



while(1):


    for i in range(0,5):
        _, frame_l = cap1.read()
        _, frame_r = cap2.read()

    # Frame sizes
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
    disp = stereo.compute(frame_l, frame_r).astype(np.float32) / 16.0

    oldnum = 0
    point_list =[]
    for step in range(99, 0, -1):
        new_tresh = step/100.00
        ret, thresh1 = cv2.threshold((disp-min_disp) / num_disp, new_tresh, 1.0, cv2.THRESH_BINARY)
        num = cv2.countNonZero(thresh1)
        point_list.append((num - oldnum, new_tresh))
        oldnum = num
    pixels_dif = np.array([x[0] for x in point_list])
    pixels_dif[pixels_dif < 0.45*max(pixels_dif)] = 0
    peak_max_ind = argrelextrema(pixels_dif, np.greater)[0].tolist()
    peaks_max_val = [point_list[i][0] for i in peak_max_ind]
    mask_list = []
    for peak_ind, val in zip(peak_max_ind, peaks_max_val):
        search_left_ind = peak_ind
        search_right_ind = peak_ind
        while pixels_dif[search_left_ind] > 0.10*val:
            search_left_ind -= 1
        while pixels_dif[search_right_ind] > 0.10*val:
            search_right_ind += 1
        print point_list[search_left_ind][1],point_list[search_right_ind][1], point_list[peak_ind][1]
        _, tresh_temp = cv2.threshold((disp-min_disp) / num_disp, point_list[search_right_ind][1],
                                      point_list[search_left_ind][1], cv2.THRESH_BINARY
                                      )
        mask_list.append(tresh_temp)

    # plt.plot(np.array([x[1] for x in point_list]), np.array([x[0] for x in point_list]))
    # plt.show()


    # fig, ax = plt.subplots()
    # im = ax.imshow(mask, interpolation='none')
    # ax.format_coord = Formatter(im)
    # plt.show()
    # cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
    #                         cv2.CHAIN_APPROX_SIMPLE)[-2]
    # areas = [cv2.contourArea(c) for c in cnts]
    # max_index = np.argmax(areas)
    # cnt=cnts[max_index]
    #
    # x,y,w,h = cv2.boundingRect(cnt)
    # cv2.rectangle(rect_frames[0],(x,y),(x+w,y+h),(0,255,0),2)
    # c = max(cnts, key=cv2.contourArea)
    # M = cv2.moments(c)
    # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    # cv2.circle(rect_frames[0], center, 5, (0, 0, 255), -1)
    # print Q[2,3]*(1/Q[3,2])/disp[center]*1.0
    try:
        cv2.imshow('tr1',mask_list[0])
    except:
        pass
    try:
        cv2.imshow('tr2',cv2.bitwise_and(mask_list[1], cv2.bitwise_not(mask_list[0])))
    except:
        pass
    cv2.imshow("disparity",(disp-min_disp)/num_disp)
    cv2.imshow('frame_left',rect_frames[0])
    cv2.imshow('frame_right',rect_frames[1])

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()