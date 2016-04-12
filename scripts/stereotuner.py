import cv2
import argparse
import threading
import numpy as np

CAMERA_WIDTH_PARAM = 3
CAMERA_HEIGHT_PARAM = 4

class StereoSGBMTuner(threading.Thread):
    '''
    StereoTuner for stereoSGBM OpenCV method
    '''

    def __init__(self, lcamera, rcamera, low_res):
        threading.Thread.__init__(self)
        try:
            lc, rc = [int(lcamera[-1]), int(rcamera[-1])]
        except ValueError:
            print '\nConversion Error, probably invalid device ID\n'
            raise

        # list of owned cameras
        self.cam_sources = [0,1]
        self.low_res = low_res
        self.params_loaded = False

        # trackbar params
        self.window_size = 0
        self.pre_filter = 0
        self.unique = 0
        self.speckle_win = 0
        self.speckle_range = 0
        self.disp12 = -1

        # Stereo Camera object declaration
        self.cam_r = cv2.VideoCapture(rc)
        self.cam_l = cv2.VideoCapture(lc)

        # For lack off usb bandwidth
        if low_res:
            self.cam_r.set(CAMERA_WIDTH_PARAM, 320)
            self.cam_r.set(CAMERA_HEIGHT_PARAM, 240)
            self.cam_l.set(CAMERA_WIDTH_PARAM, 320)
            self.cam_l.set(CAMERA_HEIGHT_PARAM, 240)

        self.init_trackbar()
        self.start()

    def run(self):
        aval = ['Y', 'N']
        user_str = ''
        # Show message after loading params
        while not self.params_loaded:
            pass

        # Show again when typed wrong cmd
        while user_str not in aval:
            user_str = raw_input("Save ? ( Y/N )")
        if user_str == aval[0]:
            print "Parameters Saved ! Congratz"
            self.save_params()

    # Save parameters
    def save_params(self):
        # Save result
        np.savez(
            "stereo_tune.npz", window_size=self.window_size, pre_filter=self.pre_filter, unique=self.unique,
            speckle_win=self.speckle_win, speckle_range=self.speckle_range, disp12=self.disp12
        )

    # Load calibration parameters
    def load_params(self):
        # Load previously saved data
        with np.load('disortion_params.npz') as X:
            left_maps = X["left_maps"]
            right_maps = X["right_maps"]

        return left_maps, right_maps

    # Initialize trackbar to tune params
    def init_trackbar(self):
        cv2.imshow('disparity',np.zeros((100,100)))
        cv2.createTrackbar('blockSize', 'disparity', 3, 11, self.set_window_size)
        cv2.createTrackbar('preFilterCap', 'disparity', 0, 500, self.set_pre_filter)
        cv2.createTrackbar('uniquenessRatio', 'disparity', 0, 30, self.set_unique)
        cv2.createTrackbar('speckleWindowSize', 'disparity', 0, 200, self.set_speckle_win)
        cv2.createTrackbar('speckleRange', 'disparity', 0, 5, self.set_speckle_range)
        cv2.createTrackbar('disp12MaxDiff', 'disparity', -1, 5, self.set_disp12_max_diff)

    def rotate90(self, stereoframe):

        frame_l = stereoframe[0]
        frame_r = stereoframe[1]

        # Frame sizes
        rows, cols, dim = frame_l.shape

        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
        frame_l = cv2.warpAffine(frame_l, M, (cols, rows))
        frame_r = cv2.warpAffine(frame_r, M, (cols, rows))
        return frame_l, frame_r

    def computer_disparity(self, frames):
        # disparity range is tuned image pair
        window_size = self.window_size
        min_disp = 0
        num_disp = 112-min_disp
        stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
            preFilterCap = self.pre_filter,
            numDisparities = num_disp,
            blockSize = window_size,
            uniquenessRatio = self.unique,
            speckleWindowSize = self.speckle_win,
            speckleRange = self.speckle_range*16,
            disp12MaxDiff = self.disp12,
            P1 = 8*3*window_size**2,
            P2 = 32*3*window_size**2,
            # fullDP = False
        )
        frame_l = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        frame_r = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
        disp = stereo.compute(frame_l, frame_r).astype(np.float32) / 16.0
        return (disp-min_disp)/num_disp

    def start_tune(self):

        print "Loading params"
        # Load calibration parameters
        maps = self.load_params()
        self.params_loaded = True
        while True:

            # Read single frame from camera
            for i in range(0,5):
                _, frame_l = self.cam_l.read()
                _, frame_r = self.cam_r.read()

            # Rotate 90 degrees
            frames = self.rotate90((frame_l, frame_r))

            # Convert type and remap
            rect_frames = [0,0]
            for src in self.cam_sources:
                map1_conv = np.array([[[coef for coef in y] for y in x] for x in maps[src][0]])
                map2_conv = np.array([[y for y in x] for x in maps[src][1]])
                rect_frames[src] = cv2.remap(frames[src], map1_conv, map2_conv, cv2.INTER_LANCZOS4)

            # Compute disparity from pair of images
            disp = self.computer_disparity(rect_frames)
            frame_l = frames[0]
            frame_r = frames[1]

            cv2.imshow('frame_left', frame_l)
            cv2.imshow('frame_right', frame_r)
            cv2.imshow('disparity', disp)
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()

    # Set window size parameter
    def set_window_size(self, arg):
        self.window_size = arg

    # Set pre filter parameter
    def set_pre_filter(self, arg):
        self.pre_filter = arg

    # Set unique ratio
    def set_unique(self, arg):
        self.unique = arg

    # Set speckle window parameter
    def set_speckle_win(self, arg):
        self.speckle_win = arg

    # Set speckle range parameter
    def set_speckle_range(self, arg):
        self.speckle_range = arg

    # Set disp12MaxDiff parameter
    def set_disp12_max_diff(self, arg):
        self.disp12 = arg

def main():

    # Script description
    description = 'Script from StereoCam package to tune stereo camera\n' \
                  'Using this script you can find parameters to tune your stereo camera\n' \
                  'Good luck ! :D \n'

    # Set command line arguments
    parser = argparse.ArgumentParser(description)

    # Camera parameters
    parser.add_argument('-lc', '--leftcamera', dest='lcamera', action='store', default="/dev/video0")
    parser.add_argument('-rc', '--rightcamera', dest='rcamera', action='store', default="/dev/video1")
    parser.add_argument('--lowres', dest='lowres', action="store_true", default=True)

    args = parser.parse_args()

    # Calibrate camera
    ct = StereoSGBMTuner(args.lcamera, args.rcamera, args.lowres)
    ct.start_tune()



if __name__ == '__main__':
    main()