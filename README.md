# ASL Recognition Using Deep Learning
**American Sign Language (ASL) Alphabets Gesture Recognition**  
Performance Comparison of CNN, Transfer Learning, and Vision Transformers

---

## Summary

This project focuses on recognizing American Sign Language (ASL) alphabet gestures using deep learning models. It evaluates and compares the performance of three model types:

- Convolutional Neural Networks (CNN)
- Transfer Learning (MobileNetV2)
- Vision Transformers (ViT)

The system also includes a real-time ASL recognizer using webcam input.

---

## Dataset

Three public datasets were merged for this project:

- Kaggle ASL Alphabet Dataset
- Roboflow ASL Gestures
- Sign Language MNIST

Images were preprocessed and resized into three versions:

- 28x28 for lightweight CNN models
- 64x64 for enhanced training
- 224x224 for transfer learning and transformers

---

## Methodology

- Image preprocessing (grayscale, normalization, resizing)
- One-hot encoding of alphabet labels
- Model development and training:
  - Simple CNN models
  - Improved CNN with batch normalization and dropout
  - Transfer Learning using MobileNetV2 with and without fine-tuning
  - Vision Transformers (ViT)
- Evaluation using:
  - Accuracy
  - Precision, Recall, F1-Score
  - Confusion Matrix
- Real-time ASL alphabet recognition using OpenCV and webcam

---

## How to Use

1. Clone the repository:
   ```
   git clone https://github.com/hibakhurshid96/ASL-Recognition-using-deep-learning.git
   cd ASL-Recognition-using-deep-learning
   ```

2. Install dependencies:
   ```
   pip install opencv-python tensorflow transformers pillow
   ```

3. Run the real-time recognizer:
   ```
   python Asl-app.py
   ```

---

## Output

- Accuracy up to 98.61% using Vision Transformers
- Side-by-side comparison of models based on performance metrics
- Real-time recognition system capable of detecting alphabet gestures through webcam

---
