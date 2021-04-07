# Complexity estimation

From Nagle and Lavie 2020, Predicting human complexity perception of real-world scenes.

This repository provides the weights file, ground truth from the experiment, and code to check the complexity model and generate predictions for new images.

# Requirements

The model has over 18 million parameters, so a good GPU is required; we used a Titan X (12 GB VRAM) for development.
Python 3
TensorFlow
Keras
You will need to download the PASCAL VOC image dataset, decompress it and configure its location in the notebook.

Git LFS is used to store the .h5 weights file.