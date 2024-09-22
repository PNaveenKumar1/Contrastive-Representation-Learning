# Contrastive Representation Learning

## Overview

This project involves implementing machine learning models on the CIFAR-10 dataset. It is divided into two main problems:

1. **Softmax Regression (Multiclass Logistic Regression)**: Implementing binary logistic regression and extending it to handle multi-class classification using softmax.
2. **Contrastive Representation Learning**: Training an encoder to map images into a vector space, where similar images are close together and dissimilar ones are far apart.

The project is implemented using PyTorch and adheres to standard deep learning practices, including the use of gradient descent optimization, cross-entropy loss, and model fine-tuning. The CIFAR-10 dataset is used, which contains 60,000 32x32 color images from 10 mutually exclusive classes.

## Setup

To run the project, ensure you have the latest stable version of PyTorch installed. 

## Problems

### Problem 1: Softmax Regression

- **Goal**: Implement logistic regression for binary classification and extend it to softmax regression for multi-class classification.
- **Files**: Implementations are in `run.py`, `utils.py`, and the `LogisticRegression/` folder.
- **Cost Function**: Binary cross-entropy for logistic regression and multi-class cross-entropy for softmax regression.
- **Training Strategy**: Batch size of 256, stopping criteria based on 80% validation accuracy.
- **Results**: Achieved consistent accuracy and loss reduction using gradient clipping and L2 regularization. The training loss and accuracy plots are saved as `1.1a.png` and `1.1b.png`.

### Problem 2: Contrastive Representation Learning

- **Goal**: Train a neural network (encoder) to map images into a vector space. Then, fine-tune the encoder and classify images using a logistic regression classifier.
- **Files**: Implementations are in the `ContrastiveRepresentation/` folder.
- **Network Architecture**: Encoder models experimented with include VGG11 and VGG16, achieving an accuracy of up to 85.88%.
- **Cost Function**: Used the margin triplet loss function to maximize the similarity between positive image pairs and minimize the similarity with negative image pairs.
- **t-SNE Plot**: Visualized the learned vector space of the encoder using t-SNE, showing distinct class clusters (saved as `1.3.png`).
- **Fine-tuning**: Fine-tuned the encoder with additional layers and achieved significant performance improvements compared to the frozen model.

## Triplet Loss
L(a,p,n) = max{d(ai,pi) − d(ai,ni) + margin,0}

 d(xi,yi) = ∥xi − yi∥p
## How to Run

To run the code:

1. For Softmax regression:
    ```bash
    python run.py <22915> --mode softmax <hyperparameters>
    ```
2. For Contrastive Representation Learning:
    ```bash
    python run.py <22915> --mode cont_rep <hyperparameters>
    ```
Hyperparameters were mentioned in the Report
## Results

- **Softmax Regression**: Achieved over 80% accuracy on binary logistic regression and comparable results for multi-class classification.
- **Contrastive Representation Learning**: Achieved 85.88% accuracy using a fine-tuned VGG16 encoder with triplet margin loss.

Plots and results are included in the report.

## Report

The detailed report for this assignment is available in the repository as `Report_<22915>.pdf`.

