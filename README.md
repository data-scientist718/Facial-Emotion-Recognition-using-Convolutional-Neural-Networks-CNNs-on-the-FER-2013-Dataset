
# Facial Emotion Recognition using CNNs on FER-2013 Dataset

## Overview

This project implements a Convolutional Neural Network (CNN) to recognize and classify facial emotions from grayscale images. The model is trained on the FER-2013 dataset, which contains 35,887 48x48 pixel images labeled into seven emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

The goal is to develop an emotion classification system that can accurately classify facial expressions into one of these seven categories. This model can serve as the foundation for applications such as emotion detection in video surveillance, emotional AI, or human-computer interaction.

## Project Purpose

### Objective

Develop a CNN-based deep learning model for facial emotion recognition to classify images into seven different emotions using the FER-2013 dataset.

### Dataset

- **Dataset Name**: FER-2013 Dataset
- **Dataset Size**: 35,887 grayscale images, each 48x48 pixels
- **Emotion Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral
- **Source**: [Kaggle FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

## Key Tasks

1. **Data Preprocessing**: Load, clean, and preprocess the dataset images; one-hot encode emotion labels.
2. **Model Building**: Construct a CNN-based neural network for image classification.
3. **Model Training**: Train the CNN model on the FER-2013 dataset for 50 epochs.
4. **Evaluation**: Assess model performance using accuracy, loss metrics, and confusion matrix.
5. **Application**: Implement a prediction function to classify emotions from new images.

## Expected Outcome

The project aims to produce a CNN model that can classify facial emotions with a test accuracy of approximately 75-80%. The model should be able to generalize to new, unseen images and predict the most likely emotion.

## Project Structure

- `Facial_Emotion_Recognition_CNN.ipynb`: Jupyter notebook containing the full implementation of the project.
- `emotion_recognition_model.h5`: The saved trained CNN model.
- `fer2013.csv`: The dataset containing images and emotion labels (must be downloaded manually).
- `README.md`: This documentation file.

## Installation and Requirements

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt