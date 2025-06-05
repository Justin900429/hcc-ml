# 🎯 Object Detection Tutorial

> [!CAUTION]
> This tutorial focuses on model training. For data labeling instructions, please check lab3.

## 🛠️ Setup Environment

```shell
python3.12 -m venv obj
source obj/bin/activate
pip install -r requirements.txt
```

## 📦 Data Preparation

> [!WARNING]
> For simplicity, we provide pre-labeled data in this tutorial. In your final project, you will need to prepare and label your own dataset.

1. Download the dataset:

    ```shell
    wget https://github.com/Justin900429/hcc-ml/releases/download/dataset/HCC_catdog.zip
    ```

2. Unzip the data:

    ```shell
    # Replace {NAME_OF_ZIP} with your zip file name
    unzip -q {NAME_OF_ZIP}.zip -d dataset

    # Example
    unzip -q HCC_catdog.zip -d dataset
    ```

Your folder structure should look like this:

```plaintext
dataset
| - images  # training images
| - labels  # labels for images (matching image names)
| - classes.txt  # class names
| ...  # other files
```

## ⚙️ Configuration

Edit `assets/custom_dataset.yaml`:

```yaml
path: ./dataset  # keep as is
train: images    # keep as is
val: images      # keep as is

# Update these labels to match your classes
names:
  0: cat
  1: dog
```

> [!TIP]
> Keep the first three lines unchanged. Just make sure your label names match the order in `dataset/classes.txt`.

## 🚀 Training

Start training with:

```shell
python train.py
```

> [!NOTE] Advanced Note
> Want to experiment? Try adjusting `epochs` or changing the pretrained model in `model_path`.

After training, find your model weights in `runs/train/weights`:

* `best.pt` - Best model (highest mAP)
* `last.pt` - Latest model

> [!TIP]
> Use `best.pt` for predictions. Curious about mAP? Check out its meaning!

> [!IMPORTANT]
> Running `python train.py` multiple times creates new folders (`runs/train2`, `runs/train3`, etc.). To overwrite previous results, add the `--overwrite` flag.

## 🔮 Making Predictions

Test your model:

```shell
python predict.py \
    --image {PATH_TO_IMAGE} \
    --weights {PATH_TO_WEIGHT}

# Example
python predict.py \
    --image dataset/images/4d8f0341-c1.jpg \
    --weights runs/train/weights/best.pt
```

> [!TIP]
> You can adjust `conf_threshold` and `nms_threshold` as covered in lab3.
