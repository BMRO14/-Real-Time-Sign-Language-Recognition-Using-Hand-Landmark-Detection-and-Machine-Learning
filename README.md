# Real-Time Sign Language Recognition

A simple project that recognizes static hand signs for the 26 letters of the English alphabet using a webcam, hand landmarks, and machine learning.

## Overview

This project uses MediaPipe Hands to detect hand landmarks from live video.  
The landmark coordinates are then normalized and used to train a Random Forest classifier.  
The system can recognize letters A–Z in real time and display the prediction on the video stream.

## Features

- Real-time webcam-based recognition.
- Recognition of the 26 alphabet letters.
- Hand landmark detection with MediaPipe Hands.
- Machine learning classification with Random Forest.
- Custom dataset collected from labeled gesture images.

## How It Works

1. Capture images of hand gestures for each letter.
2. Detect 21 hand landmarks from each image.
3. Normalize the landmark coordinates.
4. Train a Random Forest model.
5. Use the trained model for live prediction from the webcam.

## Project Files

- `collect_imgs.py` – collects gesture images from the webcam.
- `create_dataset.py` – extracts hand landmarks and builds the dataset.
- `train_classifier.py` – trains the classifier and saves the model.
- `interference_classifier.py` – runs real-time recognition.

## Dataset

- 26 classes: A to Z.
- 200 images per class.
- 5200 total images.

## Alphabet Reference

![Alphabet gestures](images/alphabet.png)

## Results

The project achieved about 99.9% accuracy on the test set.

## Requirements

- Python
- OpenCV
- MediaPipe
- NumPy
- scikit-learn

## Author

Vlad-Andrei Paraschiv
