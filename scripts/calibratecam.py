#!/usr/bin/python
import cv2
import numpy as np
import argparse


class CamCalibrate(object):
    '''
        Python class contains methods to calibrate stereo
         camera by taking pictures of chessboard shape in different positions
    '''

    def __init__(self, height, width, scale,
                 lcamera, rcamera, lowres):
        try:
            lc, rc = [int(lcamera[-1]), int(rcamera[-1])]
        except ValueError:
            print '\nConversion Error, probably invalid device ID\n'
            raise
        try:
            # Cheeseboard size
            self.chessboard_size = (int(width), int(height))
            self.chessboard_scale = float(scale)
        except ValueError:
            print '\nConversion Error, wrong height and width arg!\n'
            raise

        # list of owned cameras
        self.cam_sources = [0,1]
        self.lowres = lowres

        # Stereo Camera object declaration
        self.cam_r = cv2.VideoCapture(rc)
        self.cam_l = cv2.VideoCapture(lc)

        # For lack off usb bandwidth
        if lowres:
            self.cam_r.set(3, 320)
            self.cam_r.set(4, 240)
            self.cam_l.set(3, 320)
            self.cam_l.set(4, 240)

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Image and Object points buffers
        self.objpoints = []
        self.imgpoints_r = []
        self.imgpoints_l = []
        self.img_counter = 0

    # Check CV version
    @staticmethod
    def is_opencv3():
        if cv2.__version__.startswith("3."):
            return True
        else:
            return False

    # Calibrate camera
    def calibrate(self):

        ##
        undisorted = False
        chessboard_found = False
        possib_cho = ['N', 'Y']
        user_choice = ""

        # object points, depend on chessboard size
        objp = np.zeros((np.prod(self.chessboard_size), 3)
                        , np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]]\
            .T.reshape(-1, 2)
        objp *= self.chessboard_scale
        raw_input("Press any key to start calibration...")

        while True:

            # Read single frame from camera
            for i in range(0,5):
                _, frame_l = self.cam_l.read()
                _, frame_r = self.cam_r.read()

            # Frame sizes
            rows, cols, dim = frame_l.shape

            M = cv2.getRotationMatrix2D((cols/2, rows/2), -90, 1)
            frame_l = cv2.warpAffine(frame_l, M, (cols, rows))
            frame_r = cv2.warpAffine(frame_r, M, (cols, rows))

            # Convert to grayscale
            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l,
                                                         self.chessboard_size, None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r,
                                                         self.chessboard_size, None)

            # If chessboard corners found
            if ret_l and ret_r and not undisorted:

                print "Left Camera: Chessboard Found"
                print "Right Camera: Chessboard Found"

                self.objpoints.append(objp)

                cv2.cornerSubPix(gray_l, corners_l, (5, 5), (-1, -1), self.criteria)
                cv2.cornerSubPix(gray_r, corners_r, (5, 5), (-1, -1), self.criteria)

                # Add image points to buffer
                self.imgpoints_l.append(corners_l)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                cv2.drawChessboardCorners(frame_l, self.chessboard_size, corners_l, ret_l)
                cv2.drawChessboardCorners(frame_r, self.chessboard_size, corners_r, ret_r)

                if self.lowres:

                    # Put text on image
                    cv2.putText(frame_l, "Found!!!", (100, rows / 2), cv2.FONT_HERSHEY_SIMPLEX,
                                2, (255, 255, 255), 5, 1)
                    cv2.putText(frame_r, "Found!!!", (100, rows / 2), cv2.FONT_HERSHEY_SIMPLEX,
                                2, (255, 255, 255), 5, 1)
                else:
                    # Put text on image
                    cv2.putText(frame_l, "Found!!!", (100, rows / 2), cv2.FONT_HERSHEY_SIMPLEX,
                                4, (255 , 255, 255), 5, 1)
                    cv2.putText(frame_r, "Found!!!", (100, rows / 2), cv2.FONT_HERSHEY_SIMPLEX,
                                4, (255, 255, 255), 5, 1)

                self.img_counter += 1
                chessboard_found = True

            # Remap rectified frames
            elif undisorted:
                frames = [frame_l, frame_r]
                maps = [left_maps, right_maps]
                rect_frames = [0,0]
                for src in self.cam_sources:
                        rect_frames[src] = cv2.remap(frames[src], maps[src][0], maps[src][1], cv2.INTER_LANCZOS4)

                # Undistorted images
                frame_l = rect_frames[0]
                frame_r = rect_frames[1]

            # show image
            cv2.imshow('img_left', frame_l)
            cv2.imshow('img_right', frame_r)
            k = cv2.waitKey(5) & 0xFF

            # Founded?, What next
            if chessboard_found:
                while user_choice not in possib_cho:
                    user_choice = raw_input("Found: %d , continue (Y/N) : " % (self.img_counter,))
                chessboard_found = False

                # If user chose No
                if user_choice == possib_cho[0]:
                    print "Using calculated parameters to remove disortion..."

                    if not self.is_opencv3():
                        # Calculating distortion params for stereo camera
                        ret_val, camera_mat_l, dist_l, camera_mat_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
                                self.objpoints, self.imgpoints_l,
                                self.imgpoints_r, gray_l.shape[::-1]
                        )

                        # distortion params
                        dist = [dist_l, dist_r]

                        # Remove translation
                        for src in self.cam_sources:
                            dist[src][0][-1] = 0.0
                    else:
                        # Calculating distortion params for each camera
                        ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(self.objpoints,
                                                                                     self.imgpoints_l,
                                                                                     gray_l.shape[::-1],
                                                                                     None,None)
                        ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(self.objpoints,
                                                                                     self.imgpoints_r,
                                                                                     gray_r.shape[::-1],
                                                                                    None,None)

                        print "Calibration Error for left, right camera : " + str(ret_l) + ", " + str(ret_r)

                        # distortion params
                        dist = [dist_l, dist_r]

                        # Remove translation
                        for src in self.cam_sources:
                            dist[src][0][-1] = 0.0

                        # find new camera and remove translation params
                        camera_mat_l, roi_l = cv2.getOptimalNewCameraMatrix(mtx_l, dist_l, (cols, rows), 1, (cols, rows))
                        camera_mat_r, roi_r = cv2.getOptimalNewCameraMatrix(mtx_r, dist_r, (cols, rows), 1, (cols, rows))

                        stereorms, mtx_l, dist_l, mtx_r,\
                        dist_r, R, T, E, F = cv2.stereoCalibrate(
                                objectPoints=self.objpoints, imagePoints1=self.imgpoints_l,
                                imagePoints2=self.imgpoints_r, cameraMatrix1=camera_mat_l,
                                distCoeffs1=dist_l, cameraMatrix2=camera_mat_r,
                                distCoeffs2=dist_r, imageSize=gray_l.shape[::-1],
                                flags=(cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_RATIONAL_MODEL)
                        )
                        print "Calibration Error for both cameras : " + str(stereorms)

                        # Crop option
                        rectify_scale = 0

                        # Rectification
                        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                                camera_mat_l, dist_l,
                                camera_mat_r, dist_r,
                                gray_l.shape[::-1], R, T, alpha=rectify_scale
                        )
                        left_maps = cv2.initUndistortRectifyMap(camera_mat_l, dist_l, R1, P1, gray_l.shape[::-1], cv2.CV_16SC2)
                        right_maps = cv2.initUndistortRectifyMap(camera_mat_r, dist_r, R2, P2, gray_l.shape[::-1], cv2.CV_16SC2)

                    # Save result
                    np.savez(
                            "../data/disortion_params.npz", camera_mat_l=camera_mat_l,
                            camera_mat_r=camera_mat_r, dist_l=dist_l,
                            dist_r=dist_r, R=R, T=T, E=E, Q=Q,
                            F=F, left_maps=left_maps, right_maps=right_maps
                    )

                    undisorted = True
                user_choice = ""

            if k == 27:
                break

        cv2.destroyAllWindows()


def main():

    # Script description
    description = 'Script from StereoCam package to calibrate stereo camera\n' \
                  'It removes physical effect called distortion, this step is necessary\n' \
                  'in cheap cameras.\n'

    # Set command line arguments
    parser = argparse.ArgumentParser(description)

    # Chessboard size parameters
    parser.add_argument('-ch', '--height', dest='height', action='store', default="6")
    parser.add_argument('-cw', '--width', dest='width', action='store', default="9")
    parser.add_argument('-s', '--scale', dest='scale', action='store', default="2.3")
    parser.add_argument('--lowres', dest='lowres', action="store_true", default=True)

    # Camera parameters
    parser.add_argument('-lc', '--leftcamera', dest='lcamera', action='store', default="/dev/video2")
    parser.add_argument('-rc', '--rightcamera', dest='rcamera', action='store', default="/dev/video1")

    args = parser.parse_args()

    # Calibrate camera
    cc = CamCalibrate(args.height, args.width, args.scale,
                      args.lcamera, args.rcamera, args.lowres)
    cc.calibrate()

if __name__ == '__main__':
    main()
