import socket
import json
from threading import Thread, Event, Lock
from time import sleep

import cv2
import torch
import numpy as np
import requests
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from yolov3.transforms import preprocess
from yolov3.models import YOLOs
from yolov3.models.utils import parse_weights
from yolov3.utils import nms, draw_bbox, draw_text


class StoppableThread(Thread):
    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self.stop_event = Event()
        self.stopped_event = Event()

    def stop(self):
        self.stop_event.set()
        counter = 0
        while self.is_alive():
            print("Wait for [%s] to stop: %d" % (
                self.__class__.__name__, counter))
            sleep(1)
            counter += 1
        print("[%s] is stopped" % self.__class__.__name__)

    def stopped(self):
        return self.stop_event.is_set()


class StateReceiver(StoppableThread):
    def __init__(self, ip="0.0.0.0", port=8890):
        super().__init__()
        self.ip = ip
        self.port = port
        self.state_lock = Lock()
        self.state = dict()

    def run(self):
        sck = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sck.bind((self.ip, self.port))
        while not self.stopped():
            response, _ = sck.recvfrom(1024)
            for item in response.decode().strip().split(';'):
                item = item.strip()
                if len(item) == 0:
                    continue
                try:
                    key, value = item.split(':')
                    if key == 'mpry':
                        value = tuple(map(float, value.split(',')))
                    else:
                        value = float(value)
                    with self.state_lock:
                        self.state[key] = value
                except Exception:
                    print("Invalid state item: '%s'" % item)
        sck.close()

    def get_state(self):
        with self.state_lock:
            return self.state.copy()


class ImageReceiver(StoppableThread):
    def __init__(self, ip="192.168.10.1", port=11111):
        super().__init__()
        self.ip = ip
        self.port = port

        self.lock = Lock()
        self.id = None
        self.img = None

    def run(self):
        counter = 0
        cap = cv2.VideoCapture("udp://%s:%d" % (self.ip, self.port))
        while not self.stopped():
            success, frame = cap.read()
            if success:
                with self.lock:
                    self.id = counter
                    self.img = frame
                    counter += 1
        cap.release()

    def get_img(self):
        with self.lock:
            return self.id, self.img


class LocalDetectionReceiver(StoppableThread):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    label2name = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    label2color = torch.randint(
        0, 256, size=(len(label2name), 3),
        generator=torch.Generator().manual_seed(1))

    def __init__(self,
                 image_receiver: ImageReceiver,
                 img_size,
                 conf_threshold,
                 nms_threshold,
                 model,
                 n_classes,
                 weights):
        super().__init__()
        self.image_receiver = image_receiver
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self.model = YOLOs[model](n_classes).to(self.device)
        if weights.endswith('.pt'):
            ckpt = torch.load(weights)
            self.model.load_state_dict(ckpt['model'])
        else:
            parse_weights(self.model, weights)
        self.model.eval()

        self.result_lock = Lock()
        self.id = None
        self.img = None
        self.bboxes = None
        self.labels = None
        self.scores = None
        self.names = None

    def run(self):
        while not self.stopped():
            id, raw = self.image_receiver.get_img()
            if id is None:
                continue
            # BGR to RGB
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            # convert to Pillow Image
            raw = Image.fromarray(raw).convert('RGB')
            raw_size = torch.tensor([raw.size[1], raw.size[0]]).long()
            img, _ = preprocess(raw)
            img = img.unsqueeze(0)

            # start to detect
            with torch.no_grad():
                img = img.to(self.device)
                raw_size = raw_size.to(self.device)
                bboxes = self.model(img)
                bboxes = nms(
                    bboxes, self.conf_threshold, self.nms_threshold)[0]

            # draw bboxes
            bboxes, scores, labels = torch.split(bboxes, [4, 1, 1], dim=1)
            bboxes = preprocess.revert(bboxes, raw_size, self.img_size)
            scores = scores.view(-1)
            labels = labels.view(-1).long()
            names = []
            for bbox, label in zip(bboxes, labels):
                name = self.label2name[label]
                color = self.label2color[label]
                draw_bbox(raw, bbox, name, color)
                draw_text(raw, bbox, name, color)
                names.append(name)
            raw = cv2.cvtColor(np.asarray(raw), cv2.COLOR_RGB2BGR)

            # save detections and image
            with self.result_lock:
                self.id = id
                self.img = raw
                self.bboxes = bboxes
                self.scores = scores
                self.labels = labels
                self.names = names

    def get_result(self):
        with self.result_lock:
            return (
                self.id,
                self.img,
                self.bboxes,
                self.scores,
                self.labels,
                self.names
            )


class RemoteDetectionReceiver(StoppableThread):
    def __init__(self,
                 image_receiver: ImageReceiver,
                 img_size,
                 conf_threshold,
                 nms_threshold,
                 url='http://140.113.160.149:11080/api/yolov3'):
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
            id, raw = self.image_receiver.get_img()
            if id is None:
                continue
            # BGR to RGB
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            # convert to Pillow Image
            raw = Image.fromarray(raw).convert('RGB')
            raw_size = torch.tensor([raw.size[1], raw.size[0]]).long()
            preprocess.update_img_size(self.img_size)
            img, _ = preprocess(raw)
            # convert to cv2 image
            img = np.asarray(to_pil_image(img))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # encode image as jpeg
            _, img_encoded = cv2.imencode('.jpg', img)
            # url params
            params = {
                'img_size': self.img_size,
                'conf_threshold': self.conf_threshold,
                'nms_threshold': self.nms_threshold,
            }
            # send http request with image and receive response
            r = requests.post(
                self.url,
                data=img_encoded.tobytes(),
                headers=self.headers,
                params=params)
            # decode response
            r = json.loads(r.text)
            bboxes = torch.tensor([pred['bbox'] for pred in r['predictions']])
            scores = torch.tensor([pred['score'] for pred in r['predictions']])
            labels = torch.tensor([pred['label'] for pred in r['predictions']])
            colors = torch.tensor([pred['color'] for pred in r['predictions']])
            names = [pred['name'] for pred in r['predictions']]
            if len(bboxes) > 0:
                bboxes = preprocess.revert(bboxes, raw_size, self.img_size)
            for bbox, score, color, name in zip(bboxes, scores, colors, names):
                draw_bbox(raw, bbox, name, color)
                draw_text(raw, bbox, name, color)
                print('+ Label: %s, Conf: %.5f' % (name, score))
            raw = cv2.cvtColor(np.asarray(raw), cv2.COLOR_RGB2BGR)

            # save detections and image
            with self.result_lock:
                self.id = id
                self.img = raw
                self.bboxes = bboxes
                self.scores = scores
                self.labels = labels
                self.names = names

    def get_result(self):
        with self.result_lock:
            return (
                self.id,
                self.img,
                self.bboxes,
                self.scores,
                self.labels,
                self.names
            )
