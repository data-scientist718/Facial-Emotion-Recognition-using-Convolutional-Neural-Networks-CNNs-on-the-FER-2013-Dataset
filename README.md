
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
```

## Dependencies

To run this project, ensure you have the following dependencies installed:

- **TensorFlow** >= 2.4.0
- **Keras** >= 2.4.3
- **NumPy** >= 1.19.5
- **Pandas** >= 1.1.5
- **Matplotlib** >= 3.3.4
- **Seaborn** >= 0.11.1
- **Scikit-learn** >= 0.24.0

## Key Features

### 1. Data Preprocessing

- Load the FER-2013 dataset and preprocess the images for CNN input.
- Normalize the images and convert the emotion labels into one-hot encoded format.

### 2. Model Building

- A Convolutional Neural Network (CNN) was designed with the following components:
  - **Conv2D Layers**: Extract features from the input images.
  - **MaxPooling2D Layers**: Reduce the dimensionality of the features.
  - **Dense Layers**: Fully connected layers for final classification.
  - **Dropout Layer**: Included to prevent overfitting.
  - **Softmax Activation**: Used for multiclass classification.

### 3. Model Training

- The model is trained on the FER-2013 dataset using categorical cross-entropy as the loss function and Adam optimizer.
- The training process is tracked for accuracy and loss metrics over 50 epochs.

### 4. Model Evaluation

- The model's performance is evaluated on the test set, with accuracy and loss scores calculated.
- A confusion matrix is generated to visualize the model's performance across all emotion classes.
- Training and validation accuracy and loss are plotted to monitor model performance over time.

### 5. Prediction Function

- The `predict_emotion` function is provided, which takes a new image as input and returns the predicted emotion label.
- This function can be used for real-time predictions on unseen images.

## Usage Instructions

### 1. Training the Model

To train the CNN model from scratch, use the Jupyter notebook `Facial_Emotion_Recognition_CNN.ipynb`. After training, the model will be saved as `emotion_recognition_model.h5`.

### 2. Running Predictions

To run predictions, use the `predict_emotion` function within the notebook. This function takes an image as input and returns the predicted emotion.

Example usage:

```python
predicted_emotion = predict_emotion(model, sample_img)
print(f'Predicted emotion: {predicted_emotion}') 
```

## Results

- **Test Accuracy**: The model achieves an approximate test accuracy of 75-80% on the FER-2013 dataset.
- **Model Performance**: The model generalizes well to unseen images and can accurately classify facial expressions into one of the seven emotions.

## Future Enhancements

- **Data Augmentation**: Applying data augmentation techniques like random rotations, zooms, and shifts could improve the model's generalization performance on the test data.
- **Transfer Learning**: Using a pre-trained model like VGG16 or ResNet and fine-tuning it on the FER-2013 dataset could boost performance.
- **Real-time Application**: The current implementation could be extended to real-time emotion detection from live video streams using a webcam or other video input.

## Conclusion

This project demonstrates the use of CNNs for image classification tasks such as facial emotion recognition. The model has been trained and evaluated on the FER-2013 dataset and is capable of predicting facial emotions with reasonable accuracy. With further tuning and improvements, this model could be integrated into various real-world applications requiring emotion detection.
