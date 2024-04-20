# CIFAR-10-Image-Classification-using-PyTorch
An image classification project implementing a custom neural network architecture using PyTorch to accurately classify images from the CIFAR-10 dataset 

## Overview
This project involves the implementation of a custom neural network architecture to classify images from the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 testing images.

## Dataset
The CIFAR-10 dataset used in this project is loaded into PyTorch DataLoaders to facilitate efficient training and testing procedures. The dataset can be explored more at [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

## Model Architecture
The neural network architecture is designed with multiple convolutional layers arranged in blocks (B1, B2, ..., BK) followed by an output block that generates logits for classification. Each block processes the input image and outputs a transformed image, which is passed to the next block or the output block.

### Intermediate Blocks
- Each intermediate block comprises multiple independent convolutional layers.
- Each convolutional layer applies a set of transformations and combines their outputs weighted by a computed vector.

### Output Block
- The final block computes a logits vector from the last intermediate block's output, which is used to classify the image.

## Training and Evaluation
- The model is trained using cross-entropy loss.
- A batch size and various hyperparameters were set to optimize training.
- Training involves calculating loss and accuracy after each epoch, and testing is done to evaluate the model's performance.

## Results
- Training loss and accuracy plots are included to demonstrate the learning process.
- The model achieved a maximum accuracy of [88.51]% on the CIFAR-10 test set.

## Techniques Used
- Various techniques from the coursework were employed to enhance model performance, including [list techniques used].
- Hyperparameter tuning was crucial in improving the accuracy of the model.

## How to Run
To run this project, follow these steps:
1. Clone the repository:
   ```bash
   git clone [Repository URL]

