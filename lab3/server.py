import argparse
import json

import cv2
import numpy as np
import torch
from flask import Flask, Response, request
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # server
    parser.add_argument("--port", type=int, default=8888, help="server port")
    # yolo
    parser.add_argument(
        "--weights", type=str, default="./weights/yolov7-w6.pt", help="path to weights file"
    )
    parser.add_argument("--img_size", type=int, default=640, help="evaluation image size")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.45, help="nms threshold")
    return parser.parse_args()


# define label names
label2name = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

args = parse_args()

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# define label colors
label2color = torch.randint(
    0, 256, size=(len(label2name), 3), generator=torch.Generator().manual_seed(1)
)

# Initiate model
model = attempt_load(args.weights, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(args.img_size, s=stride)

# Initialize the Flask application
app = Flask(__name__)


def preprocess(img, img_size, stride):
    img = letterbox(img, img_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img) / 255.0
    img = torch.from_numpy(img).unsqueeze(0)
    return img.float()


# route http posts to this method
@app.route("/api/yolov7", methods=["POST"])
def test():
    r = request
    # get parsed contents of query string
    conf_threshold = float(r.args.get("conf_threshold", str(args.conf_threshold)))
    nms_threshold = float(r.args.get("nms_threshold", str(args.nms_threshold)))
    img_size = int(r.args.get("img_size", str(imgsz)))

    # convert string of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    im0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = preprocess(im0, img_size, stride)

    # make prediction
    with torch.no_grad():
        img = img.to(device)
        pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_threshold, nms_threshold, agnostic=True)[0]

    # build a response dict to send back to client
    response = {"predictions": []}
    print("-" * 80)

    if len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()
        for *xyxy, conf, cls in reversed(pred):
            name = label2name[int(cls)]
            color = label2color[int(cls)]
            response["predictions"].append(
                {
                    "bbox": [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()],
                    "score": conf.item(),
                    "label": int(cls),
                    "color": color.tolist(),
                    "name": name,
                }
            )
    else:
        print("No Objects Deteted!!")

    return Response(response=json.dumps(response), status=200, mimetype="application/json")


# start flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port)
