import argparse
import json

import torch
import cv2
import requests
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from yolov3.transforms import preprocess
from yolov3.utils import draw_bbox, draw_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--port', type=int, default=8888, help='server port')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--img', type=str, default='./data/street.jpg')
    parser.add_argument('--img_size', type=int, default=416)
    args = parser.parse_args()

    raw = Image.open(args.img)
    raw_size = torch.tensor([raw.size[1], raw.size[0]]).long()
    preprocess.update_img_size(args.img_size)
    img, _ = preprocess(raw)
    # convert to cv2 image
    img = np.asarray(to_pil_image(img))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', img)
    # url params
    params = {
        'img_size': args.img_size,
        'conf_threshold': 0.5,
        'nms_threshold': 0.45,
    }
    # send http request with image and receive response
    r = requests.post(
        f"http://{args.host}:{args.port}/api/yolov3",
        data=img_encoded.tobytes(),
        headers={'content-type': 'image/jpeg'},
        params=params)
    # decode response
    r = json.loads(r.text)
    bboxes = torch.tensor([pred['bbox'] for pred in r['predictions']])
    scores = torch.tensor([pred['score'] for pred in r['predictions']])
    labels = torch.tensor([pred['label'] for pred in r['predictions']])
    colors = torch.tensor([pred['color'] for pred in r['predictions']])
    names = [pred['name'] for pred in r['predictions']]
    if len(bboxes) > 0:
        bboxes = preprocess.revert(bboxes, raw_size, args.img_size)
    for bbox, score, color, name in zip(bboxes, scores, colors, names):
        draw_bbox(raw, bbox, name, color)
        draw_text(raw, bbox, name, color)
        print('+ Label: %s, Conf: %.5f' % (name, score))
    raw = cv2.cvtColor(np.asarray(raw), cv2.COLOR_RGB2BGR)
    cv2.imwrite('testserver.png', raw)
    print('Saved testserver.png')
