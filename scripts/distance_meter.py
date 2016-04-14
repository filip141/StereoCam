import cv2
import argparse
import numpy as np
from scipy.signal import argrelextrema


class DistanceMeter(object):
    '''
    Algorithm using disparity map approximate distance information
    to specified nearest object
    '''

    CAMERA_WIDTH_PARAM = 3
    CAMERA_HEIGHT_PARAM = 4

    def __init__(self, lcamera, rcamera, low_res):
        try:
            lc, rc = [int(lcamera[-1]), int(rcamera[-1])]
        except ValueError:
            print '\nConversion Error, probably invalid device ID\n'
            raise

        # Stereo Camera object declaration
        self.cam_r = cv2.VideoCapture(rc)
        self.cam_l = cv2.VideoCapture(lc)

        # For lack off usb bandwidth
        if low_res:
            self.cam_r.set(DistanceMeter.CAMERA_WIDTH_PARAM, 320)
            self.cam_r.set(DistanceMeter.CAMERA_HEIGHT_PARAM, 240)
            self.cam_l.set(DistanceMeter.CAMERA_WIDTH_PARAM, 320)
            self.cam_l.set(DistanceMeter.CAMERA_HEIGHT_PARAM, 240)

        # Initialize stereo
        self.cam_sources = [0,1]
        self.stereo = self.init_stereo()
        self.left_maps, self.right_maps, self.qmat = self.load_params()

    # Load calibration parameters
    def load_params(self):
        # Load previously saved data
        with np.load('disortion_params.npz') as X:
            left_maps = X["left_maps"]
            right_maps = X["right_maps"]
            Q = X["Q"]
        return left_maps, right_maps, Q

    # Initialize stereoSGBM algoritm
    def init_stereo(self):
        with np.load('stereo_tune.npz') as X:
            window_size = X["window_size"]
            min_disp = 0
            num_disp = 112 - min_disp
            stereo = cv2.StereoSGBM_create(
                    minDisparity=min_disp,
                    preFilterCap=X["pre_filter"],
                    numDisparities=num_disp,
                    blockSize=window_size,
                    uniquenessRatio=X["unique"],
                    speckleWindowSize=X["speckle_win"],
                    speckleRange=X["speckle_range"] * 16,
                    disp12MaxDiff=X["disp12"],
                    P1=8 * 3 * window_size**2,
                    P2=32 * 3 * window_size**2,
            )
        return stereo

    # Read frames from video device
    def read_frames(self):
        for i in range(0, 5):
            _, frame_l = self.cam_l.read()
            _, frame_r = self.cam_r.read()
        return frame_l, frame_r

    # Rotate stereo frame 90 degree
    def rotate90(self, stereoframe):

        frame_l = stereoframe[0]
        frame_r = stereoframe[1]

        # Frame sizes
        rows, cols, dim = frame_l.shape

        rot_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
        frame_l = cv2.warpAffine(frame_l, rot_mat, (cols, rows))
        frame_r = cv2.warpAffine(frame_r, rot_mat, (cols, rows))
        return frame_l, frame_r

    # Rectify stereo frame
    def rectify_frames(self, stereoframe):
        maps = [self.left_maps, self.right_maps]
        rect_frames = [0, 0]
        for src in self.cam_sources:
            map1_conv = np.array([[[coef for coef in y] for y in x] for x in maps[src][0]])
            map2_conv = np.array([[y for y in x] for x in maps[src][1]])
            rect_frames[src] = cv2.remap(stereoframe[src], map1_conv, map2_conv, cv2.INTER_LANCZOS4)
        return rect_frames

    # Compute stereo disparity map
    def computer_stereo(self, stereoframe):
        frame_l = cv2.cvtColor(stereoframe[0], cv2.COLOR_BGR2GRAY)
        frame_r = cv2.cvtColor(stereoframe[1], cv2.COLOR_BGR2GRAY)
        frame_l[frame_l == 0] = cv2.mean(frame_l)[0]
        frame_r[frame_r == 0] = cv2.mean(frame_r)[0]
        disp = self.stereo.compute(frame_l, frame_r).astype(np.float32) / 16.0
        return disp

    # Method to find object mask, is possible to find n objects
    def find_object(self, disparity_map, nobj):
        old_num = 0
        num_disp = 112
        point_list = []
        mask_list = []

        # Diff calculation
        for step in range(99, 0, -1):
            new_tresh = step / 100.00
            ret, disp_tresh = cv2.threshold(disparity_map / num_disp, new_tresh, 1.0, cv2.THRESH_BINARY)
            num = cv2.countNonZero(disp_tresh)
            point_list.append((num - old_num, new_tresh))
            old_num = num
        pixels_dif = np.array([x[0] for x in point_list])
        pixels_dif[pixels_dif < 0.45*max(pixels_dif)] = 0
        peak_max_ind = argrelextrema(pixels_dif, np.greater)[0].tolist()
        peaks_max_val = [point_list[i][0] for i in peak_max_ind]

        # Check nobj argument
        if nobj > len(peak_max_ind):
            print "nobject argument out of range, casted to list length"
            nobj = len(peak_max_ind)

        peak_max_ind = peak_max_ind[:nobj]
        peaks_max_val = peaks_max_val[:nobj]

        # Find peak size
        for peak_ind, val in zip(peak_max_ind, peaks_max_val):
            search_left_ind = peak_ind
            search_right_ind = peak_ind
            while pixels_dif[search_left_ind] > 0.10*val:
                search_left_ind -= 1
            while pixels_dif[search_right_ind] > 0.10*val:
                search_right_ind += 1
            _, tresh_temp = cv2.threshold(disparity_map / num_disp, point_list[search_right_ind][1],
                                          point_list[search_left_ind][1], cv2.THRESH_BINARY
                                          )
            mask_list.append(tresh_temp)
        return mask_list

    # Measure distance to object
    def measure_dist(self):
        # Read and rotate
        stereoframe = self.read_frames()
        stereoframe = self.rotate90(stereoframe)

        # Rectify and compute disparity map
        stereoframe = self.rectify_frames(stereoframe)
        disp = self.computer_stereo(stereoframe)
        mask_list = self.find_object(disp, 1)
        cv2.imshow("mask", mask_list[0])

    # Never ending loop
    def camera_loop(self):
        while True:
            self.measure_dist()
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()


def main():
    # Script description
    description = 'Script from StereoCam package to get distance from image\n' \
                  'Algorithm using disparity map approximate distance information\n' \
                  'to specified nearest object \n'

    # Set command line arguments
    parser = argparse.ArgumentParser(description)

    # Camera parameters
    parser.add_argument('-lc', '--leftcamera', dest='lcamera', action='store', default="/dev/video1")
    parser.add_argument('-rc', '--rightcamera', dest='rcamera', action='store', default="/dev/video2")
    parser.add_argument('--lowres', dest='lowres', action="store_true", default=True)

    args = parser.parse_args()

    # Calibrate camera
    dm = DistanceMeter(args.lcamera, args.rcamera, args.lowres)
    dm.camera_loop()


if __name__ == '__main__':
    main()
