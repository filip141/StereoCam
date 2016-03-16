#!/usr/bin/python
import cv2
import numpy as np
import argparse


class CamCalibrate(object):

    def __init__(self, height, width, scale,
                 lcamera, rcamera):
        try:
            lc, rc = [int(lcamera[-1]), int(rcamera[-1])]
        except ValueError:
            print '\nConversion Error, probably invalid device ID\n'
            raise

         ## Stereo Camera object declaration
        self.cam_r = cv2.VideoCapture(rc)
        self.cam_l = cv2.VideoCapture(lc)

        try:
            ## Cheestboard size
            self.cheesboard_size = (int(width), int(height))
            self.cheesboard_scale = float(scale)
        except ValueError:
            print '\nConversion Error, wrong height and width arg!\n'
            raise

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        ## Image and Object points buffers
        self.objpoints = []
        self.imgpoints_r = []   ## Right Camera
        self.imgpoints_l = []   ## Left Camera
        self.img_counter = 0

    ## Calibrate camera
    def calibrate(self):

        ##
        undisorted = False
        cheesboard_found = False
        possib_cho = ['N', 'Y']
        user_choice = ""

        # object points, depend on cheesboard size
        objp = np.zeros((np.prod(self.cheesboard_size),3)
                        , np.float32)
        objp[:,:2] = np.mgrid[0:self.cheesboard_size[0],0:self.cheesboard_size[1]]\
            .T.reshape(-1,2)
        objp = objp*self.cheesboard_scale
        raw_input("Press any key to start calibration...")

        while True:

            ## Read single frame from camera
            for i in range(0,5):
                _, frame_l = self.cam_l.read()
                _, frame_r = self.cam_r.read()

            ## Frame sizes
            rows,cols,dim = frame_l.shape

            M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
            frame_l = cv2.warpAffine(frame_l,M,(cols,rows))
            frame_r = cv2.warpAffine(frame_r,M,(cols,rows))

            ## Convert to grayscale
            gray_l = cv2.cvtColor(frame_l,cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l,
                                                     self.cheesboard_size, None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r,
                                                     self.cheesboard_size, None)

            ## If cheesboard corners found
            if ret_l and ret_r and not undisorted:

                print "Left Camera: Cheesboard Found"
                print "Right Camera: Cheesboard Found"

                self.objpoints.append(objp)

                cv2.cornerSubPix(gray_l, corners_l,(5, 5),(-1,-1), self.criteria)
                cv2.cornerSubPix(gray_r, corners_r,(5, 5),(-1,-1), self.criteria)

                ## Add image points to buffer
                self.imgpoints_l.append(corners_l)
                self.imgpoints_r.append(corners_r)

                ## Draw and display the corners
                cv2.drawChessboardCorners(frame_l, self.cheesboard_size, corners_l, ret_l)
                cv2.drawChessboardCorners(frame_r, self.cheesboard_size, corners_r, ret_r)

                ## Put text on image
                cv2.putText(frame_l,"Found!!!", (100,rows/2), cv2.FONT_HERSHEY_SIMPLEX,
                            4,(255,255,255),5, 1)
                cv2.putText(frame_r,"Found!!!", (100,rows/2), cv2.FONT_HERSHEY_SIMPLEX,
                            4,(255,255,255),5, 1)

                self.img_counter = self.img_counter + 1
                cheesboard_found = True

            cv2.imshow('img_left',frame_l)
            cv2.imshow('img_right',frame_r)
            k = cv2.waitKey(5) & 0xFF

            ## Founded?, What next
            if cheesboard_found:
                while user_choice not in possib_cho:
                    user_choice =  raw_input("Found: %d , continue (Y/N) : " % (self.img_counter,))
                cheesboard_found = False

            ## If user chose No
            if user_choice == possib_cho[0]:
                print "Using calculated parameters to remove disortion..."

                ## Calculating disortion params
                ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(self.objpoints,
                                                                             self.imgpoints_l,
                                                                             gray_l.shape[::-1],
                                                                             None,None)
                ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(self.objpoints,
                                                                             self.imgpoints_r,
                                                                             gray_r.shape[::-1],
                                                                             None,None)
                newcameramtx_l, roi_l=cv2.getOptimalNewCameraMatrix(mtx_l,dist_l,(cols,rows),1,(cols,rows))
                newcameramtx_r, roi_r=cv2.getOptimalNewCameraMatrix(mtx_r,dist_r,(cols,rows),1,(cols,rows))

                # undistort
                frame_l = cv2.undistort(frame_l, mtx_l, dist_l, None, newcameramtx_l)
                frame_r = cv2.undistort(frame_r, mtx_r, dist_r, None, newcameramtx_r)

                # crop the image
                x_l,y_l,w_l,h_l = roi_l
                x_r,y_r,w_r,h_r = roi_r
                frame_l = frame_l[y_l:y_l+h_l, x_l:x_l+w_l]
                frame_r = frame_r[y_r:y_r+h_r, x_r:x_r+w_r]
                undisorted = True
                user_choice = ""

            elif user_choice == possib_cho[1]:
                user_choice = ""
            if k == 27:
                break

        cv2.destroyAllWindows()




def main():

    ## Script description
    description = 'Script from StereoCam package to calibrate stereo camera\n' \
                  'It removes physical effect called distortion, this step is necessary\n' \
                  'in cheap cameras.\n'

    ## Set command line arguments
    parser = argparse.ArgumentParser(description)

    ## Cheesboard size parameters
    parser.add_argument('-ch', '--height', dest='height', action='store', default="6")
    parser.add_argument('-cw', '--width', dest='width', action='store', default="9")
    parser.add_argument('-s', '--scale', dest='scale', action='store', default="2.3")

    ## Camera parameters
    parser.add_argument('-lc', '--leftcamera', dest='lcamera', action='store', default="/dev/video1")
    parser.add_argument('-rc', '--rightcamera', dest='rcamera', action='store', default="/dev/video2")

    args = parser.parse_args()

    ## Calibrate camera
    cc = CamCalibrate(args.height, args.width, args.scale,
                      args.lcamera, args.rcamera)
    cc.calibrate()

if __name__ == '__main__':
    main()