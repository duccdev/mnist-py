# `mnist-py`

Neural network made from scratch that trains on the MNIST dataset to recognize handwritten digits.

# Docs

## Prerequisites

You must install everything inside `requirements.txt` by running `pip3 install -r requirements.txt`

## Training

Grab a dataset of shape `(n, 28, 28)` and put it inside the `datasets` folder or use the default MNIST dataset that comes with this repository  
Run `python3 train.py` and follow the instructions

## Using it

Train a model first or use the MNIST model that comes with this repository and then run `python3 main.py`  
When using your own images, you can use [https://pixilart.com/](https://pixilart.com/) to draw the images  
You can use any resolution since it automatically upscales/downscales but it is preferred to use 28x28 for extra precision
