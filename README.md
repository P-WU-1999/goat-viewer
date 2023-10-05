# goat-viewer

<p align="center">
  <img src="https://github.com/P-WU-1999/goat-viewer/blob/main/repo_images/Concept.png">
</p>

How to Steer at Goats for Accurate Facial Recognition: Goat individual visual recognition based on VGGNet. Including 
preprocessing and recognition functions.

The project is tested on the [Billah's Goat Image Dataset](https://data.mendeley.com/datasets/4skwhnrscr/2). It achieved an accuracy rate of
94% in this dataset.

## Structure

<p align="center">
  <img src="https://github.com/P-WU-1999/goat-viewer/blob/main/repo_images/Methodology.png">
</p>

## Instructions

### Preprocessing

Preprocessing is based on the [YOLOv8](https://github.com/ultralytics/ultralytics) algorithm. We used it's “Detect” and 
"Segment" models. Instructions can be found on the [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/). 
We also provided two trained YOLOv8 models in [models/YOLO](https://github.com/P-WU-1999/goat-viewer/tree/main/models/YOLO).
You may replace them with your own models by overwrite them.

If you are using your own model, you should modify the num_goats parameter in [training/train.py](https://github.com/P-WU-1999/goat-viewer/blob/main/training/train.py) and [recognition/recognition.py](https://github.com/P-WU-1999/goat-viewer/tree/main/datasets/recognition).

By doing preprocess you should follow steps below:

1. Put your datasets in the [datasets/source](https://github.com/P-WU-1999/goat-viewer/tree/main/datasets/source) folder.
Different goats should be kept in folders corresponding to the goat name.
2. Run [preprocess/preprocessor.py](https://github.com/P-WU-1999/goat-viewer/blob/main/preprocess/preprocessor.py).
3. Check the preprocessed data in [datasets/preprocessout](https://github.com/P-WU-1999/goat-viewer/tree/main/datasets/preprocessout) folder.

### Prepare training dataset

Training set and validation set are required for training. We provided methods to generate them automatically:

1. Run [training/label.py](https://github.com/P-WU-1999/goat-viewer/blob/main/training/label.py).
2. Run [training/separate.py](https://github.com/P-WU-1999/goat-viewer/blob/main/training/separate.py). This will separate the preprocessed dataset as 7:3 for training and validation.
3. Check your training set at [datasets/train](https://github.com/P-WU-1999/goat-viewer/tree/main/datasets/train) and validation set at [datasets/valid](https://github.com/P-WU-1999/goat-viewer/tree/main/datasets/valid)

### Training

Run [training/train.py](https://github.com/P-WU-1999/goat-viewer/blob/main/training/train.py). This will start the training loop.
Tensorboard logs will be output at training/logs folder (It will be automatically create). The best models with the highest accuracy will be saved in the training folder.

### Recognition

1. [Preprocess](https://github.com/P-WU-1999/goat-viewer#preprocessing) the new dataset.
2. Put the preprocessed dataset into [datasets/recognition](https://github.com/P-WU-1999/goat-viewer/tree/main/datasets/recognition) folder.
3. Put the trained model in recognition folder and rename it as model.pth.
4. Run [recognition/recognition.py](https://github.com/P-WU-1999/goat-viewer/blob/main/recognition/recognition.py).

## Our Results

<p align="center">
  <img src="https://github.com/P-WU-1999/goat-viewer/blob/main/repo_images/ncm.png">
</p>

## Citation

If you use either the paper or code in your paper or project, please kindly star this repo and cite our work.



