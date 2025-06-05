import random

import cv2
import tyro
from ultralytics import YOLO

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


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def main(
    image: str,
    weights: str = "yolo11s.pt",
    img_size: int = 640,
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.7,
    device: str = "cuda",
):
    model = YOLO(weights)

    results = model.predict(
        image,
        conf=conf_threshold,
        iou=nms_threshold,
        imgsz=img_size,
        device=device,
        save=False,
    )[0]

    ori_img = results.orig_img
    boxes = results.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        plot_one_box(
            box.xyxy[0],
            ori_img,
            label=f"{model.names[int(box.cls)]} {box.conf.item():.2f}",
            color=COLORS[int(box.cls)],
        )
    cv2.imwrite("demo.png", ori_img)


if __name__ == "__main__":
    tyro.cli(main)
