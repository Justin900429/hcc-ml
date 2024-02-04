import argparse
import json

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from utils.plots import plot_one_box

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--port", type=int, default=8888, help="server port")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--img", type=str, default="./data/street.jpg")
    parser.add_argument("--img_size", type=int, default=640)
    args = parser.parse_args()

    raw = Image.open(args.img)
    raw_size = torch.tensor([raw.size[1], raw.size[0]]).long()
    img = np.asarray(raw)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # encode image as jpeg
    _, img_encoded = cv2.imencode(".jpg", img)
    # url params
    params = {
        "img_size": args.img_size,
        "conf_threshold": 0.5,
        "nms_threshold": 0.45,
    }
    # send http request with image and receive response
    r = requests.post(
        f"http://{args.host}:{args.port}/api/yolov7",
        data=img_encoded.tobytes(),
        headers={"content-type": "image/jpeg"},
        params=params,
    )
    # decode response
    r = json.loads(r.text)
    bboxes = torch.tensor([pred["bbox"] for pred in r["predictions"]])
    scores = torch.tensor([pred["score"] for pred in r["predictions"]])
    labels = torch.tensor([pred["label"] for pred in r["predictions"]])
    colors = torch.tensor([pred["color"] for pred in r["predictions"]])
    names = [pred["name"] for pred in r["predictions"]]
    for bbox, score, color, name in zip(bboxes, scores, colors, names):
        plot_one_box(
            bbox,
            img,
            label=name,
            color=color.tolist(),
            line_thickness=1,
        )
        print("+ Label: %s, Conf: %.5f" % (name, score))
    cv2.imwrite("testserver.png", img)
    print("Saved testserver.png")
