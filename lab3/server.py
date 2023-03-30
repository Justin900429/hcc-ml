import argparse
import json

import cv2
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, Response

from yolov3.transforms import preprocess
from yolov3.models import YOLOs
from yolov3.models.utils import parse_weights
from yolov3.utils import nms


if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# server
parser.add_argument('--port', type=int, default=8888, help='server port')
# yolo
parser.add_argument('--model', choices=YOLOs.keys(), default='yolov3',
                    help='model name')
parser.add_argument('--weights', type=str, default='./weights/yolov3.weights',
                    help='path to weights file')
parser.add_argument('--n_classes', default=80, type=int,
                    help='nunmber of classes')
parser.add_argument('--img_size', type=int, default=416,
                    help='evaluation image size')
parser.add_argument('--conf_threshold', type=float, default=0.5,
                    help='confidence threshold')
parser.add_argument('--nms_threshold', type=float, default=0.45,
                    help='nms threshold')
args = parser.parse_args()


# define label names
label2name = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']

# define label colors
label2color = torch.randint(
    0, 256, size=(len(label2name), 3),
    generator=torch.Generator().manual_seed(1))

# Initiate model
model = YOLOs[args.model](args.n_classes).to(device)
if args.weights.endswith('.pt'):
    ckpt = torch.load(args.weights)
    model.load_state_dict(ckpt['model'])
else:
    parse_weights(model, args.weights)
model.eval()

# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/api/yolov3', methods=['POST'])
def test():
    r = request
    # get parsed contents of query string
    conf_threshold = float(
        r.args.get('conf_threshold', str(args.conf_threshold)))
    nms_threshold = float(
        r.args.get('nms_threshold', str(args.nms_threshold)))
    img_size = int(
        r.args.get('img_size', str(args.img_size)))
    # convert string of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # convert to pillow image
    img = Image.fromarray(img)
    orig_size = torch.tensor([img.size[1], img.size[0]]).long()
    preprocess.update_img_size(img_size)
    img, _ = preprocess(img)
    img = img.unsqueeze(0)
    # make prediction
    with torch.no_grad():
        img, orig_size = img.to(device), orig_size.to(device)
        bboxes = model(img)
        bboxes = nms(bboxes, conf_threshold, nms_threshold)[0]
    # build a response dict to send back to client
    response = {
        "predictions": []
    }
    print("-" * 80)
    if len(bboxes):
        bboxes, scores, labels = torch.split(bboxes, [4, 1, 1], dim=1)
        bboxes = preprocess.revert(bboxes, orig_size, img_size)
        for bbox, score, label in zip(bboxes, scores, labels):
            name = label2name[int(label)]
            color = label2color[int(label)]
            response['predictions'].append({
                "bbox": [
                    bbox[0].item(), bbox[1].item(),
                    bbox[2].item(), bbox[3].item()],
                "score": score.item(),
                "label": int(label),
                "color": color.tolist(),
                "name": name,
            })
            print('+ Label: %s, Conf: %.5f' % (name, score))
    else:
        print("No Objects Deteted!!")

    return Response(
        response=json.dumps(response), status=200, mimetype="application/json")


# start flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=args.port)
