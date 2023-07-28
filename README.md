# HemeYolo-dev/README

# Introduction

The HemeYolo package includes various scripts for cell object detection on bone marrow aspirate patches. These scripts cover data preparation, model training, conformal calibration, and more.

Before getting started, it is essential to download and install the HemeYolo-dev package. Here are the necessary steps:

```
git clone <https://github.com/neoyinzhanghan/HemeYolo-dev>
```

```
cd HemeYolo-dev
```

```
pip install .
```

After completing these steps, you can proceed with the pipeline for cell object detection.

---

# Table of content

- Data Preparation
    - Cropping and label creation
    - Splitting dataset
    - Data augmentation
- Training YOLOv8
- Deploying checkpoints
- Visualization
- Conformal procedure

---

# Data Preparation

Recommend that you create a separate conda environment for each stage of the pipeline.

```bash
conda create --name HemeYolo_data
```

```bash
conda activate HemeYolo_data
```

```bash
pip install -r ./HemeYolo_data/requirements.txt
```

Note that the `pip` install assumes that HemeYolo-dev is your working directory.

---

## **Cropping and label creation**

The first step in data preparation is to crop the labeled regions to the desired pixel size. The labels are assumed to be centroid labels, where a dot is placed where a desired cell is located. However, YOLO labeled data requires bounding boxes as labels. The current package supports two methods of bounding box creation: fixed box size and Laplacian Beam Search (LBS). Fixed box size creates a square box centered at the centroid label using a user-specified radius. Laplacian Beam Search looks for a box size that is adaptive to the actual cell size based on Laplacian boundary detection.

To run the script, first ensure that you have the right package requirements. This assumes that HemeYolo-dev is your working directory.

```
pip install -r ./HemeYolo_data/requirements.txt
```

Then run this script to perform the cropping and labeling of your data.

```
python ./HemeYolo_data/crop_and_label.py \
--images_dir /path/to/a/folder/of/region/images \
--label_csv /path/to/a/csv/file/containing/labels/for/each/images \
--output_dir /path/to/where/you/want/to/save/cropped/images/and/labels \
--crop_width 512 \
--crop_height 512 \
--box_method LBS

```

The `crop_width` is the desired width of your cropped regions, and the `crop_height` is the desired height of your cropped regions. The `box_method` should be set to LBS if you want to use the LBS procedure for adaptive boxing. It should be an integer if you want to use fixed box size with that particular radius.

---

## Splitting Dataset

To split the dataset into three chunks for training, validation, and test, use the `train_valid_test_split.py` script. First, ensure that you have the right package requirements by running:

```
pip install -r ./HemeYolo_data/requirements.txt
```

Then run the following command, replacing the file paths and ratios with your own:

```
python ./HemeYolo_data/train_valid_test_split.py \
--data_dir /path/to/unsplit/data \
--output_dir /path/to/save/split/data \
--train_ratio 0.5 \
--valid_ratio 0.25 \
--test_ratio 0.25

```

Note that the three ratios must add up to 1, otherwise you will get an error.

A `custom.yaml` file will be created in the output directory to inform the YOLO trainer where to find the training, validation, and testing data. The file has the following format:

```
path: (dataset directory path)
train: (complete path to dataset train folder)
test: (complete path to dataset test folder)
val: (complete path to dataset validation folder)

nc: 1

names: ['WBC']

```

This file is short and concise, providing essential information for the YOLO trainer.

---

## Data Augmentation

After obtaining the training data, the next step is to apply data augmentation to the training images. Currently, only offline data augmentation is supported. To apply data augmentation, save the augmented data and then train the model with it using the `augment_train_data.py` script.

To use the `augment_train_data.py` script, first ensure that you have the right package requirements by running:

```
pip install -r ./HemeYolo_data/requirements.txt

```

Then, run the following command, replacing the file paths with your own:

```
python ./HemeYolo_data/augment_train_data.py \
--data_dir /path/to/training/data 
```

This line of text describes how the `augment_train_data.py` script modifies the data directory in place by adding new augmented regions to the data and updating the labels accordingly.

---

# Training YOLO

To train the YOLO model, it is recommended that you create a separate conda environment for training. First, create a new environment and activate it:

```bash
conda create --name HemeYolo_training
```

```bash
conda activate HemeYolo_training
```

Next, install the required packages for training using the `requirements.txt` file:

```bash
pip install -r ./HemeYolo_YOLOv8/requirements.txt
```

The `custom.yaml` file generated from the previous scripts will automatically include the training, validation, and testing data directories. To use this file, move it to the `HemeYolo-dev/HemeYolo_YOLOv8` folder, and set it as your working directory:

```bash
mv /path/to/your/custom.yaml /path/to/HemeYolo-dev/HemeYolo_YOLOv8
```

```bash
cd /path/to/HemeYolo-dev/HemeYolo_YOLOv8
```

Finally, start the training by running the script below:

```bash
yolo task=detect mode=train model=yolov8n.pt data=custom.yaml epochs=300 imgsz=512
```

The `model` parameter specifies which version of YOLO to use, the `custom.yaml` file specifies the data directory to be used for training, the `epochs` parameter specifies the maximum number of training epochs, and `imgsz` parameter specifies the size of the square regions.

---

# Deploying Checkpoints

## Getting YOLOv8 Annotations

Use the same conda environment as the training script. Have a folder of regions to be annotated, and run the following script. The training script from earlier will generate two checkpoint files in `./detect/train/weights/` — `[last.pt](http://last.pt)` and `best.pt`. The former is the checkpoint from the last epoch of training and the best is the checkpoint with the best performance across all epochs of training. This command assumes that HemeYolo-dev/HemeYolo_YOLOv8 is your working directory.

```bash
python ./get_YOLOv8_annotations.py \
--data_folder /path/to/folder/of/regions/to/be/annotated \
--output_folder /path/to/folder/to/save/output/labels \
--chkpt_path /path/to/the/model/checkpoint \
--conf_thres 0
```

When the YOLOv8 model runs, it produces 300 bounding boxes in absolute pixel coordinates and each box is associated with a confidence score between 0 and 1. The `conf_thres` parameter determines which boxes to keep based on their confidence score. However, the confidence score distribution can be arbitrary, so it is recommended to set `conf_thres` to 0 and perform a conformal calibration on the data to find the confidence score threshold with statistical error control guarantee.

---

## Custom Deployment

If you want to deploy the checkpoint yourself and write your own scripts, you just need the following checkpoint call in your Python script.

```python
from ultralytics import YOLO
import pandas as pd

model = YOLO("/path/to/checkpoint")

image_path = "path/to/region/image/to/apply/the/model"

conf_thres = 0.3 # Your desired confidence threshold

# The model applies to a list of image paths
result = model([image_path], conf=conf_thres)[0]

boxes = result.boxes.data  # This grabs the output annotations

# This for-loop is for converting the output annotations into a nicely organized
#     pandas dataframe with the columns TL_x, TL_y, BR_x, BR_y, confidence, class
#     TL_x means the x coordinate of the top-left corner of the bounding box
#     in absolute pixel coordinates, and BR_y, for instance, stands for the y
#     coordinate of the bottom-right corner.

df = pd.DataFrame(columns=['TL_x', 'TL_y', 'BR_x', 'BR_y', 'confidence', 'class'])

l1 = len(boxes)

for i in range(l1):
    box = boxes[i]
    
    TL_x, TL_y = int(box[0]), int(box[1])
    BR_x, BR_y = int(box[2]), int(box[3])
    conf = float(box[4])
    cls = int(box[5])
    
    # use pd.concat instead of append to avoid deprecation
    df = pd.concat([df, pd.DataFrame([[TL_x, TL_y, BR_x, BR_y, conf, cls]], columns=['TL_x', 'TL_y', 'BR_x', 'BR_y', 'confidence', 'class'])])
```

---

# Visualization

## Annotating Regions

Once you have the bounding box labels, you can generate annotated regions with the bounding boxes drawn. This is done by the following script. This script assumes that HemeYolo-dev is your working directory.

```bash
python ./HemeYolo_visual/annotate_regions.py \
--input_dir /path/to/folder/of/region/images \
--labels_dir /path/to/folder/of/box/labels \
--output_dir /path/to/save/the/annotated/images \
--label_type top-left-bottom-right \
--image_ext .jpg \
--conf_thres 0.35
```

The `label_type` parameter ensures that the label file can be read correctly. Two inputs are viable for the `label_type` parameter. `top-left-bottom-right` is used if the labels are generated by a YOLO model (in absolute pixel coordinate). The YOLO outputs specify the top-left and bottom-right corners of each bounding box with the corresponding confidence scores. In this case, the `conf_thres` parameter must be specified to decide which bounding boxes are to be drawn (those with confidence scores higher than `conf_thres`). `center-width-height` is used if the labels are the human-annotated centroid labels with bounding box radius (in relative coordinates). In this case, the `conf_thres` parameter does not need to be specified. If specified, it will have no effect on the program, and a `UserWarning` will be raised. Make sure that your `image_ext` is specified correctly. The usual assumption is that you save things using the `.jpg` format. Note that if you are trying to get annotated regions using YOLO outputs, you should have already run the conformal calibration script to find the `conf_thres`.

---

## Extract Cells

Instead of annotating regions, you can also call extract the bounding boxes of the regions and save them as pictures. This is done by the following script. This script assumes that HemeYolo-dev is your working directory.

```bash
python ./HemeYolo_visual/extract_cells.py \
--input_dir /path/to/folder/of/region/images \
--labels_dir /path/to/folder/of/box/labels \
--output_dir /path/to/save/the/annotated/images \
--label_type top-left-bottom-right \
--image_ext .jpg \
--conf_thres 0.35
```

The parameters are the same as `annotate_regions.py`.

---

# Conformal Procedure

Recommend that you create a new conda environment for this. This script assumes that HemeYolo-dev is your working directory.

```bash
conda create --name HemeYolo_thresholding
```

```bash
conda activate HemeYolo_thresholding
```

```bash
pip install -r HemeYolo_thresholding/requirements.txt
```

After obtaining the unthresholded bounding boxes from a YOLO checkpoint on the validation dataset (when `conf_thres==0`), you can perform a conformal procedure to obtain a conf_thres with a FNR guarantee. 

See the following Notion page for detailed descriptions of the conformal procedure.

[Conformal Prediction Procedure for HemeYolo Cell-detection FNR Control](https://foam-salamander-68d.notion.site/Conformal-Prediction-Procedure-for-HemeYolo-Cell-detection-FNR-Control-226430d2906d446f996f9b68521c2bce?pvs=4)

For caliberating, use this script:

```bash
python ./HemeYolo_thresholding/fnr_control.py \
--label_dir /path/to/folder/of/human/labels/calibration/set \
--annotation_dir /path/to/folder/of/YOLO/annotations/calibration/set \
--mode 1 \
--max_p_value 0.05 \
--alpha 0.05 \
--min_iou 0.5 
```

Once you have obtained a `conf_thres` from the calibration set, you can test its FNR performance on the test set using the following script:

```bash
python ./HemeYolo_thresholding/fnr_control.py \
--label_dir /path/to/folder/of/human/labels/test/set \
--annotation_dir /path/to/folder/of/YOLO/annotations/test/set \
--mode 0 \
--conf_thres 0.35
```
