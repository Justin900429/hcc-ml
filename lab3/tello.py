import argparse
import socket
import time

import cv2

from yolov3.models import YOLOs
from receivers import (
    ImageReceiver,
    LocalDetectionReceiver,
    RemoteDetectionReceiver,
    StoppableThread,
    StateReceiver,
)


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# server port
parser.add_argument('--port', type=int, default=None, help='server port')
# yolo
parser.add_argument('--model', default='yolov3', choices=YOLOs.keys(),
                    help='model name')
parser.add_argument('--weights', default="./weights/yolov3.weights", type=str,
                    help='path to weights file')
parser.add_argument('--n_classes', default=80, type=int,
                    help='nunmber of classes')
# demo
parser.add_argument('--img_size', type=int, default=416,
                    help='evaluation image size')
parser.add_argument('--conf_threshold', type=float, default=0.5,
                    help='confidence threshold')
parser.add_argument('--nms_threshold', type=float, default=0.45,
                    help='nms threshold')
args = parser.parse_args()


class MovingPolicy(StoppableThread):
    def __init__(self, detection_receiver, state_receiver, transmitter):
        super().__init__()
        self.detection_receiver = detection_receiver
        self.state_receiver = state_receiver
        self.transmitter = transmitter

    def run(self):
        # res = transmitter.send('takeoff')
        # print('[takeoff]: %s' % res)
        prev_id = None
        while not self.stopped():
            id, _, bboxes, scores, labels, names = \
                self.detection_receiver.get_result()
            state = self.state_receiver.get_state()
            if (id is None) or (id == prev_id) or (state is None):
                continue
            print('-' * 80)
            print("Battery: %d%%" % state['bat'])
            print("X Speed: %.1f" % state['vgx'])
            print("Y Speed: %.1f" % state['vgy'])
            print("Z Speed: %.1f" % state['vgz'])
            for bbox, score, label, name in zip(bboxes, scores, labels, names):
                # center (x, y) and box size (w, h)
                x, y, w, h = bbox
                print(", ".join([
                    'Label: %d' % int(label),
                    'Name: %s' % name,
                    'Conf: %.5f' % score,
                    'center: (%.1f, %.1f)' % (x, y),
                    'size: (%.1f, %.1f)' % (w, h)]))
            prev_id = id
            # ---------------------------
            # Add your moving policy here
            # ---------------------------
        # res = transmitter.send('land')
        # print('[land]: %s' % res)


class CommandTransmitter:
    def __init__(self, ip='192.168.10.1', port=8889, local_port=56789):
        self.ip = ip
        self.port = port
        self.sck = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sck.bind(('0.0.0.0', local_port))

    def send(self, command, timeout=1.0):
        self.sck.sendto(command.encode(), (self.ip, self.port))
        self.sck.settimeout(timeout)
        try:
            response, _ = self.sck.recvfrom(1024)
            if len(response) == 0:
                return None
            return response.decode()
        except socket.timeout:
            return None


if __name__ == '__main__':
    transmitter = CommandTransmitter()

    # Send start command and enable stream
    retry_counter = 1
    while True:
        try:
            res = transmitter.send('command')
            if res is None:
                raise RuntimeError("[command]: No response")
            if res.strip() != "ok":
                raise RuntimeError('[command]: %s ' % res)
            res = transmitter.send('streamon')
            if res is None:
                raise RuntimeError("[streamon]: No response")
            if res.strip() != "ok":
                raise RuntimeError('[streamon]: %s ' % res)
            break
        except RuntimeError as e:
            print(str(e))
            print('Retry... %d' % retry_counter)
            retry_counter += 1
            time.sleep(0.5)

    # Receive State from Tello.
    state_receiver = StateReceiver()
    state_receiver.start()

    # Receive Image from Tello.
    image_receiver = ImageReceiver()
    image_receiver.start()

    # Detect objects in received image.
    detection_receiver = LocalDetectionReceiver(
        image_receiver,
        args.img_size, args.conf_threshold, args.nms_threshold,
        args.model, args.n_classes, args.weights)
    # The other option is to use remote server as detection backend.
    # detection_receiver = RemoteDetectionReceiver(
    #     image_receiver,
    #     args.img_size, args.conf_threshold, args.nms_threshold,
    #     url='http://140.113.160.149:%d/api/yolov3' % args.port)
    detection_receiver.start()

    # The moving policy is executed in another thread.
    moving_policy = MovingPolicy(
        detection_receiver, state_receiver, transmitter)
    moving_policy.start()

    # Main thread is used to update the `cv2.imshow` window.
    try:
        prev_id = None
        while True:
            # id, img, _, _, _, _ = detection_receiver.get_result()
            id, img = image_receiver.get_img()
            if id is not None and id != prev_id:
                cv2.imshow('Detection', img)
                cv2.waitKey(1)
                prev_id = id
    except KeyboardInterrupt as e:
        # Catch `ctrl+c` event
        cv2.destroyAllWindows()     # Close `cv2.imshow` window.
        # Stop all threads.
        state_receiver.stop()
        image_receiver.stop()
        detection_receiver.stop()
        moving_policy.stop()
        raise e
