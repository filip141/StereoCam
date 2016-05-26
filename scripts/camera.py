import cv2
from abc import ABCMeta, abstractmethod
from distance_meter import DistanceMeter


class VideoCamera(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_frame(self):
        pass


class DistanceCamera(object):
    def __init__(self, cam_1, cam_2):
        self.video = DistanceMeter(cam_1, cam_2, True)

    def __del__(self):
        del self.video

    def get_frame(self, n_obj, h_param):
        try:
            stereoframe, objects = self.video.measure_dist(n_obj, h_param)
        except ValueError:
            objects = []
            cam_img = cv2.imread("../static/img/there-is-no-connected-camera-mac.jpg", 0)
            cam_img = cv2.resize(cam_img, (320, 240))
            stereoframe = (cam_img, cam_img)
        ret, jpeg = cv2.imencode('.jpg', stereoframe[0])
        return jpeg.tobytes(), objects
