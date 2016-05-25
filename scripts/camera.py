import cv2
from abc import ABCMeta, abstractmethod
from distance_meter import DistanceMeter


class VideoCamera(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_frame(self):
        pass


class DistanceCamera(object):
    def __init__(self):
        self.video = DistanceMeter("/dev/video2", "/dev/video3", True)

    def get_frame(self):
        stereoframe, objects = self.video.measure_dist(4, 5.1)
        ret, jpeg = cv2.imencode('.jpg', stereoframe[0])
        return jpeg.tobytes(), objects
