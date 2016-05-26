import cv2
import sys
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
        self.cam_sources = [0, 1]
        self.stereo = self.init_stereo()
        self.left_maps, self.right_maps, self.qmat = self.load_params()

    def __del__(self):
        self.cam_l.release()
        self.cam_r.release()

    # Load calibration parameters
    @staticmethod
    def load_params():
        # Load previously saved data
        with np.load('../data/disortion_params.npz') as X:
            left_maps_1 = X["left_maps_1"]
            right_maps_1 = X["right_maps_1"]
            left_maps_2 = X["left_maps_2"]
            right_maps_2 = X["right_maps_2"]
            q_matrix = X["Q"]
        return (left_maps_1, left_maps_2), (right_maps_1, right_maps_2), q_matrix

    # Initialize stereoSGBM algoritm
    @staticmethod
    def init_stereo():
        with np.load('../data/stereo_tune.npz') as X:
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
                    P1=8 * 3 * window_size ** 2,
                    P2=32 * 3 * window_size ** 2,
            )
        return stereo

    # Read frames from video device
    def read_frames(self):
        for i in range(0, 5):
            _, frame_l = self.cam_l.read()
            _, frame_r = self.cam_r.read()
        return frame_l, frame_r

    # Rotate stereo frame 90 degree
    @staticmethod
    def rotate90(stereoframe):

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

    @staticmethod
    def disp_diff(disparity_map):
        old_num = 0
        num_disp = 112
        point_list = []
        disp_norm = disparity_map / num_disp

        # Diff calculation
        for step in range(99, 0, -1):
            new_tresh = step / 100.00
            ret, disp_tresh = cv2.threshold(disp_norm, new_tresh, 1.0, cv2.THRESH_BINARY)
            num = cv2.countNonZero(disp_tresh)
            point_list.append((num - old_num, new_tresh))
            old_num = num
        return disp_norm, point_list

    @staticmethod
    def remove_prev_mask(mask_list):
        obj_found = len(mask_list)
        for obj_num in range(0, obj_found):
            for obj_prev in range(obj_num - 1, -1, -1):
                    bit_nor = cv2.bitwise_not(mask_list[obj_prev])
                    mask_list[obj_num] = cv2.bitwise_and(mask_list[obj_num], bit_nor)
        return mask_list

    # Find mask common value estimate
    @staticmethod
    def find_disp_mean(cnt_points, disparity_map):
        disp_map = disparity_map
        mask = np.zeros(disparity_map.shape, np.uint8)
        cv2.drawContours(mask, [cnt_points], contourIdx=-1, color=(255, 255, 255), thickness=cv2.FILLED)
        return cv2.mean(disp_map, mask=mask)[0]

    def find_object(self, object_mask, stereoframe, disparity_map, hor_param):
        # Find contour and convert to uint8
        object_mask = (object_mask * hor_param).astype(np.uint8)
        contour_points = cv2.findContours(object_mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)[-2]
        if contour_points:
            # Calculate moments
            c = max(contour_points, key=cv2.contourArea)
            obj_moments = cv2.moments(c)
            center = (
                int(obj_moments["m10"] / obj_moments["m00"]),
                int(obj_moments["m01"] / obj_moments["m00"])
            )
            if center < disparity_map.shape:
                areas = [cv2.contourArea(c) for c in contour_points]
                max_index = np.argmax(areas)
                cnt = contour_points[max_index]
                disp_comm = self.find_disp_mean(cnt, disparity_map)
                x, y, w, h = cv2.boundingRect(cnt)
                # Show rectangle
                cv2.rectangle(stereoframe[0], (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw on image
                cv2.circle(stereoframe[0], center, 5, (0, 0, 255), -1)
                obj_distance = int(self.qmat[2, 3] * (1 / self.qmat[3, 2]) / disp_comm * 1.0)
                cv2.putText(stereoframe[0], str(obj_distance), (x + w - 30, y + h), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1, 1)
                return stereoframe, obj_distance, center
        return stereoframe, None, None

    # Method to find object mask, is possible to find n objects
    def find_mask(self, disparity_map, nobj):

        mask_list = []

        # Find maxima's
        disp_norm, point_list = self.disp_diff(disparity_map)
        pixels_dif = np.array([x[0] for x in point_list])
        pixels_dif[pixels_dif < 0.0325 * disparity_map.size] = 0
        peak_max_ind = argrelextrema(pixels_dif, np.greater)[0].tolist()
        peaks_max_val = [point_list[i][0] for i in peak_max_ind]

        # Check nobj argument
        if nobj > len(peak_max_ind):
            nobj = len(peak_max_ind)

        peak_max_ind = peak_max_ind[:nobj]
        peaks_max_val = peaks_max_val[:nobj]

        for peak_ind, val in zip(peak_max_ind, peaks_max_val):
            search_left_ind = peak_ind
            search_right_ind = peak_ind
            # Find peak edges
            while pixels_dif[search_left_ind] > 0.10 * val:
                if search_left_ind - 1 < 0:
                    break
                search_left_ind -= 1
            while pixels_dif[search_right_ind] > 0.10 * val:
                if search_right_ind + 1 > len(pixels_dif) - 1:
                    break
                search_right_ind += 1
            # Extract mask
            mask = cv2.inRange(disp_norm, point_list[search_right_ind][1], point_list[search_left_ind][1])
            tresh_mask = cv2.bitwise_and(disp_norm, disp_norm, mask=mask)
            mask_list.append(tresh_mask)
        mask_list = self.remove_prev_mask(mask_list)
        return mask_list

    # Measure distance to object
    def measure_dist(self, nobj, hor_param):
        object_params = []
        # Read and rotate
        stereoframe = self.read_frames()
        if (stereoframe[0] is None) or (stereoframe[1] is None):
            raise ValueError("Probably wrong camera interface")
        stereoframe = self.rotate90(stereoframe)

        # Rectify and compute disparity map
        stereoframe = self.rectify_frames(stereoframe)
        disp = self.computer_stereo(stereoframe)
        mask_list = self.find_mask(disp, nobj)
        for object_mask in mask_list:
            stereoframe, obj_dist, obj_center = self.find_object(object_mask, stereoframe, disp, hor_param)
            if obj_dist:
                object_params.append((obj_dist, obj_center))
        return stereoframe, object_params

    # Never ending loop
    def camera_loop(self, nobj, hor_param):
        while True:
            stereoframe, _ = self.measure_dist(nobj, hor_param)
            cv2.imshow("left", stereoframe[0])
            cv2.imshow("right", stereoframe[1])
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
    parser.add_argument('-no', '--nobjects', dest='nobj', action='store', default="4")
    parser.add_argument('-hp', '--horizont', dest='hor_param', action='store', default="5.1")
    parser.add_argument('--lowres', dest='lowres', action="store_true", default=True)

    args = parser.parse_args()
    try:
        nobj_param = int(args.nobj)
        hor_param = float(args.hor_param)
    except Exception:
        print "--nobjects and horizont parameters should be integer"
        raise

    # Distance camera
    dm = DistanceMeter(args.lcamera, args.rcamera, args.lowres)
    dm.camera_loop(nobj_param, hor_param)


if __name__ == '__main__':
    main()
