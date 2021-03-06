# Semantic Segmentation Example
This repository shows how to perform semantic segmentation with pytorch and tensorflow.
It uses the dataset of the [Kaggle Data Science Bowl 2018](https://data.broadinstitute.org/bbbc/BBBC038/).

## Usage

1) Install normal packages: `pip install --user -r requirements.txt`

2) Install [Tensorflow](https://www.tensorflow.org/install/) or [Pytorch](https://pytorch.org/)
3) Download and prepare data: `python data.py`
4) Start jupyter notebook and use `segment_tf` or `segment_pytorch`

### Different Data Set
In order to use the notebooks with different data create the folders `images` and `masks` and put the images and labels inside with the following naming convention:
- image: `images/{idx}.png`
- mask: `masks/{idx}_mask.png`
