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


def preprocess(img, img_size, stride):
    img0 = cv2.imread(img)
    img = letterbox(img0, img_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img) / 255.0
    img = torch.from_numpy(img).unsqueeze(0)
    return img.float(), img0


def detect():
    source, imgsz, weights = opt.image, opt.img_size, opt.weights

    # Initialize
    device = select_device(opt.device)

    # Load model
    model = attempt_load(weights, map_location=device)
    model.eval()
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Load image and run inference
    img, im0 = preprocess(source, imgsz, stride)
    img = img.to(device)
    with torch.no_grad():
        pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(
        pred,
        opt.conf_threshold,
        opt.nms_threshold,
    )

    # Process detections
    for det in pred:  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f"{names[int(cls)]} {conf:.2f}"
                plot_one_box(
                    xyxy,
                    im0,
                    label=label,
                    color=colors[int(cls)],
                    line_thickness=1,
                )

    # Save results (image with detections)
    cv2.imwrite("demo.png", im0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="inference/images", help="source")
    parser.add_argument(
        "--weights", nargs="+", type=str, default="yolov7.pt", help="model.pt path(s)"
    )
    parser.add_argument("--img_size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument(
        "--conf_threshold", type=float, default=0.45, help="object confidence threshold"
    )
    parser.add_argument("--nms_threshold", type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    opt = parser.parse_args()
    detect()
