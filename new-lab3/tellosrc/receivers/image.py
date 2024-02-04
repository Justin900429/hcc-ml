from threading import Lock

import cv2
import numpy as np

from tellosrc.base import ResourceThread


class ImageReceiver(ResourceThread):
    def __init__(self, ip="192.168.10.1", port=11111):
        super().__init__()
        self.ip = ip
        self.port = port

        self.lock = Lock()
        self.id = None
        self.img = None

    def run(self):
        cap = cv2.VideoCapture("udp://%s:%d" % (self.ip, self.port))
        while not self.stopped():
            success, frame = cap.read()
            if success:
                with self.lock:
                    self.img = frame
                    if self.id is None:
                        self.id = 0
                    else:
                        self.id += 1
        cap.release()

    def get_result(self):
        with self.lock:
            if self.id is None:
                return self.id, (None,)
            else:
                return self.id, (np.copy(self.img),)
