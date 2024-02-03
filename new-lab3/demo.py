import argparse

import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from numpy import random
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

COLORS = [
    [4, 170, 69],
    [49, 115, 180],
    [92, 189, 54],
    [19, 80, 98],
    [18, 192, 175],
    [190, 232, 107],
    [15, 240, 14],
    [130, 208, 72],
    [206, 154, 121],
    [11, 69, 243],
    [126, 112, 207],
    [177, 44, 193],
    [163, 222, 179],
    [235, 222, 149],
    [156, 93, 138],
    [38, 47, 31],
    [70, 210, 100],
    [92, 96, 118],
    [149, 81, 163],
    [233, 201, 134],
    [24, 108, 193],
    [0, 118, 237],
    [34, 156, 144],
    [187, 106, 2],
    [117, 156, 197],
    [190, 12, 49],
    [65, 57, 126],
    [216, 30, 211],
    [155, 96, 91],
    [210, 49, 70],
    [202, 66, 197],
    [244, 63, 248],
    [93, 150, 196],
    [200, 63, 150],
    [198, 112, 69],
    [184, 85, 69],
    [56, 225, 175],
    [116, 235, 69],
    [180, 167, 94],
    [46, 202, 78],
    [20, 81, 249],
    [198, 43, 122],
    [254, 60, 18],
    [217, 93, 167],
    [154, 236, 143],
    [241, 134, 209],
    [246, 43, 160],
    [183, 110, 4],
    [81, 38, 227],
    [83, 30, 215],
    [11, 125, 221],
    [240, 242, 36],
    [232, 230, 132],
    [252, 195, 251],
    [183, 85, 214],
    [39, 205, 155],
    [61, 246, 12],
    [31, 122, 135],
    [125, 28, 191],
    [100, 30, 219],
    [174, 187, 216],
    [81, 155, 254],
    [115, 163, 234],
    [6, 203, 61],
    [52, 86, 78],
    [230, 82, 201],
    [125, 224, 153],
    [24, 9, 130],
    [160, 132, 160],
    [60, 89, 160],
    [219, 63, 4],
    [227, 188, 156],
    [17, 222, 6],
    [158, 252, 105],
    [36, 42, 252],
    [65, 42, 194],
    [163, 197, 65],
    [20, 32, 154],
    [152, 90, 103],
    [18, 69, 3],
]


def preprocess(img, img_size, stride):
    img0 = cv2.imread(img)
    img = letterbox(img0, img_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img) / 255.0
    img = torch.from_numpy(img).unsqueeze(0)
    return img.float(), img0


def detect():
    global COLORS
    source, imgsz, weights = opt.image, opt.img_size, opt.weights

    # Initialize
    device = select_device(opt.device)

    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    if len(names) >= len(COLORS):
        COLORS += [
            [random.randint(0, 255) for _ in range(3)] for _ in range(len(names) - len(COLORS))
        ]

    # Load image and run inference
    img, im0 = preprocess(source, imgsz, stride)
    img = img.to(device)
    with torch.no_grad():
        pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_threshold, opt.nms_threshold, agnostic=True)

    # Process detections
    for det in pred:  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f"{names[int(cls)]}"
                plot_one_box(
                    xyxy,
                    im0,
                    label=label,
                    color=COLORS[int(cls)],
                    line_thickness=1,
                )

    # Save results (image with detections)
    cv2.imwrite("demo.png", im0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="source")
    parser.add_argument("--weights", nargs="+", type=str, required=True, help="model.pt path(s)")
    parser.add_argument("--img_size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument(
        "--conf_threshold", type=float, default=0.55, help="object confidence threshold"
    )
    parser.add_argument("--nms_threshold", type=float, default=0.4, help="IOU threshold for NMS")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    opt = parser.parse_args()
    detect()
