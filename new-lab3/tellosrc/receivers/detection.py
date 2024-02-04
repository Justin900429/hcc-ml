import json
from threading import Lock

import cv2
import torch
import numpy as np
import requests
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from yolov3.transforms import preprocess
from yolov3.utils import draw_bbox, draw_text

from tellosrc.base import ResourceThread
from tellosrc.receivers.image import ImageReceiver


class DetectionReceiver(ResourceThread):
    def __init__(self,
                 image_receiver: ImageReceiver,
                 img_size,
                 conf_threshold,
                 nms_threshold,
                 url):
        super().__init__()
        self.image_receiver = image_receiver
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.url = url
        self.headers = {'content-type': 'image/jpeg'}

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

            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)  # BGR to RGB
            raw = Image.fromarray(raw).convert('RGB')   # convert to Pillow Image
            raw_size = torch.tensor([raw.size[1], raw.size[0]]).long()
            preprocess.update_img_size(self.img_size)
            img, _ = preprocess(raw)
            img = np.asarray(to_pil_image(img))         # convert to cv2 image
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB to BGR

            _, img_encoded = cv2.imencode('.jpg', img)  # encode image
            params = {
                'img_size': self.img_size,
                'conf_threshold': self.conf_threshold,
                'nms_threshold': self.nms_threshold,
            }
            r = requests.post(                          # send request
                self.url,
                data=img_encoded.tobytes(),
                headers=self.headers,
                params=params)
            r = json.loads(r.text)                      # decode response
            bboxes = torch.tensor([pred['bbox'] for pred in r['predictions']])
            scores = torch.tensor([pred['score'] for pred in r['predictions']])
            labels = torch.tensor([pred['label'] for pred in r['predictions']])
            colors = torch.tensor([pred['color'] for pred in r['predictions']])
            names = [pred['name'] for pred in r['predictions']]
            if len(bboxes) > 0:
                bboxes = preprocess.revert(bboxes, raw_size, self.img_size)

            # draw detections
            for bbox, score, color, name in zip(bboxes, scores, colors, names):
                draw_bbox(raw, bbox, name, color)
                draw_text(raw, bbox, name, color)
            raw = cv2.cvtColor(np.asarray(raw), cv2.COLOR_RGB2BGR)

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
                return self.id, (
                    None, None, None, None, None
                )
            else:
                return self.id, (
                    np.copy(self.img),
                    np.copy(self.bboxes),
                    np.copy(self.scores),
                    np.copy(self.labels),
                    self.names[:],
                )
