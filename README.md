# Bears and Hippos Classification

This project is aimed at solving a binary classification problem: determining whether an image contains a hippo or a bear. The images used for training were collected from [images.cv](https://images.cv/) and [universe.roboflow.com](https://universe.roboflow.com/), and the model is evaluated using a custom test set consisting of personal photographs taken in various locations, including Teriberka, Kaliningrad Zoo, Ho Chi Minh City Zoo, Tanzania, and one photo with yellow hippopotamus from Krasnodar.

## Problem Description

- **Task**: Binary image classification
- **Classes**: Bear, Hippo
- **Data**: 
  - **Training and validation sets**: Images from images.cv and universe.roboflow.com.
  - **Test set**: Personal photos taken in various locations.
  
## Models

Two models are used for the classification task:

1. **Simple CNN Model**: A custom convolutional neural network model designed for the task.
2. **ResNet Model**: A pre-trained ResNet model (ResNet-50) fine-tuned for the task.

## Dataset

The dataset consists of images of bears and hippos, which are organized into directories:

- `data/train/hippo/` — Images of hippos.
- `data/train/bear/` — Images of bears.
- `data/val/hippo/` — Validation set images of hippos.
- `data/val/bear/` — Validation set images of bears.
- `data/test/` — Test set consisting of personal photos.

## Installation

Clone the repository:

```bash
git clone https://github.com/avsakharov/bears-and-hippos-classification.git
cd bears-and-hippos-classification
```

## Git Large File Storage (LFS)
As the project contains large model files (e.g., resnet_model.pt, simple_cnn_model.pt), they are managed using Git LFS. To download the models from Git LFS, please make sure you have [Git LFS](https://git-lfs.github.com/) installed on your machine.
After cloning the repository, you can fetch the model files by running:

```bash
git lfs pull
```

If you don't want to download the pre-trained models and would prefer to train them yourself, you can do so by using the `Bears and Hippos Classification.ipynb` notebook.
In this notebook, simply uncomment the corresponding cells to train the models from scratch.

## Model Inference

If you have new images of bears or hippos, add them to the `data/my_photos` folder and run the `Bears and Hippos Classification.ipynb` file to get predictions.

