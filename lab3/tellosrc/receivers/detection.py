import json
from threading import Lock

import cv2
import numpy as np
import requests
import torch
from tellosrc.base import ResourceThread
from tellosrc.receivers.image import ImageReceiver
from utils.plots import plot_one_box


class DetectionReceiver(ResourceThread):
    def __init__(self, image_receiver: ImageReceiver, img_size, conf_threshold, nms_threshold, url):
        super().__init__()
        self.image_receiver = image_receiver
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.url = url
        self.headers = {"content-type": "image/jpeg"}

        self.result_lock = Lock()
        self.id = None
        self.img = None
        self.bboxes = None
        self.labels = None
        self.scores = None
        self.names = None

    def run(self):
        while not self.stopped():
            id, (raw,) = self.image_receiver.get_result()
            if id is None:
                continue

            _, img_encoded = cv2.imencode(".jpg", raw)  # encode image
            params = {
                "img_size": self.img_size,
                "conf_threshold": self.conf_threshold,
                "nms_threshold": self.nms_threshold,
            }
            r = requests.post(  # send request
                self.url, data=img_encoded.tobytes(), headers=self.headers, params=params
            )
            r = json.loads(r.text)  # decode response
            bboxes = torch.tensor([pred["bbox"] for pred in r["predictions"]])
            scores = torch.tensor([pred["score"] for pred in r["predictions"]])
            labels = torch.tensor([pred["label"] for pred in r["predictions"]])
            colors = torch.tensor([pred["color"] for pred in r["predictions"]])
            names = [pred["name"] for pred in r["predictions"]]
            # draw detections
            for bbox, score, color, name in zip(bboxes, scores, colors, names):
                plot_one_box(
                    bbox,
                    raw,
                    label=name,
                    color=color.tolist(),
                    line_thickness=1,
                )

            # save detections and image
            with self.result_lock:
                self.id = id
                self.img = raw
                self.bboxes = bboxes.numpy()
                self.scores = scores.numpy()
                self.labels = labels.numpy()
                self.names = names

    def get_result(self):
        with self.result_lock:
            if self.id is None:
                return self.id, (None, None, None, None, None)
            else:
                return self.id, (
                    np.copy(self.img),
                    np.copy(self.bboxes),
                    np.copy(self.scores),
                    np.copy(self.labels),
                    self.names[:],
                )
