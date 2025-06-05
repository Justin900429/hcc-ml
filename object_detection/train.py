import tyro
from ultralytics import YOLO


def main(
    model_path: str = "yolo11s.pt",
    data_path: str = "assets/custom_dataset.yaml",
    project_path: str = "runs",
    batch_size: int = 16,
    epochs: int = 100,
    imgsz: int = 640,
    device: str = "cuda",
    overwrite: bool = False,
):
    model = YOLO(model_path)
    model.train(
        project=project_path,
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch_size,
        exist_ok=overwrite,
    )


if __name__ == "__main__":
    tyro.cli(main)
