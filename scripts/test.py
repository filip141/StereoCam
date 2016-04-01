import cv2
import numpy as np
from matplotlib import pyplot as plt

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

# Load previously saved data
with np.load('disortion_params.npz') as X:
    mtx_l, dist_l, mtx_r, dist_r = [X[i] for i in ('mtx_l','dist_l', 'mtx_r','dist_r')]

cap1 = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(3)

## Remove translation
dist_l = np.array(dist_l[0])
dist_r = np.array(dist_r[0])
dist_r[2] = 0; dist_r[3] = 0
dist_l[2] = 0; dist_l[3] = 0
dist_l[-1] = 0; dist_l[-1] = 0
dist_r[-1] = 0; dist_r[-1] = 0

while(1):

    # Take each frame
    _, frame_l = cap1.read()
    _, frame_r = cap2.read()

    ## Frame sizes
    rows,cols,dim = frame_l.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
    frame_l = cv2.warpAffine(frame_l,M,(cols,rows))
    frame_r = cv2.warpAffine(frame_r,M,(cols,rows))

    # find new camera
    newcameramtx_l, roi_l=cv2.getOptimalNewCameraMatrix(mtx_l,dist_l,(cols,rows),1,(cols,rows))
    newcameramtx_r, roi_r=cv2.getOptimalNewCameraMatrix(mtx_r,dist_r,(cols,rows),1,(cols,rows))

    frame_l = cv2.undistort(frame_l, mtx_l, dist_l, None, newcameramtx_l)
    frame_r = cv2.undistort(frame_r, mtx_r, dist_r, None, newcameramtx_r)

    # crop the image
    # x_l,y_l,w_l,h_l = roi_l
    # x_r,y_r,w_r,h_r = roi_r
    # frame_l = frame_l[y_l:y_l+h_l, x_l:x_l+w_l]
    # frame_r = frame_r[y_r:y_r+h_r, x_r:x_r+w_r]

    # sift = cv2.SIFT()
    # kp1, des1 = sift.detectAndCompute(frame_l,None)
    # kp2, des2 = sift.detectAndCompute(frame_r,None)
    #
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)
    #
    #
    # flann = cv2.FlannBasedMatcher(index_params,search_params)
    # matches = flann.knnMatch(des1,des2,k=2)
    #
    # good = []
    # pts1 = []
    # pts2 = []
    #
    # # ratio test as per Lowe's paper
    # for i,(m,n) in enumerate(matches):
    #     if m.distance < 0.8*n.distance:
    #         good.append(m)
    #         pts2.append(kp2[m.trainIdx].pt)
    #         pts1.append(kp1[m.queryIdx].pt)
    #
    # pts1 = np.int32(pts1)
    # pts2 = np.int32(pts2)
    # F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    #
    # # We select only inlier points
    # pts1 = pts1[mask.ravel()==1]
    # pts2 = pts2[mask.ravel()==1]
    #
    # # Find epilines corresponding to points in right image (second image) and
    # # drawing its lines on left image
    # lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    # lines1 = lines1.reshape(-1,3)
    # img5,img6 = drawlines(frame_l,frame_r,lines1,pts1,pts2)
    #
    # # Find epilines corresponding to points in left image (first image) and
    # # drawing its lines on right image
    # lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    # lines2 = lines2.reshape(-1,3)
    # img3,img4 = drawlines(frame_r,frame_l,lines2,pts2,pts1)

    # disparity range is tuned for 'aloe' image pair
    window_size = 11
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
    frame_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
    frame_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
    disp = stereo.compute(frame_l, frame_r).astype(np.float32) / 16.0

    cv2.imshow('frame_left',frame_l)
    cv2.imshow('frame_right',frame_r)
    cv2.imshow('frame_right',(disp-min_disp)/num_disp)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()