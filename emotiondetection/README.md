# Emotion Detection using YOLO 

## Overview

This project performs real-time facial emotion detection using a webcam. It has face detection with a custom-trained YOLO model for emotion classification. The system detects faces from the live video stream and predicts the corresponding emotion for each detected face.

## Features

* Real-time webcam emotion detection
* Face detection using OpenCV
* Emotion classification using a custom YOLO model
* Bounding boxes with emotion labels
* Lightweight and suitable for CPU execution
* Modular project structure

## Technologies Used

* Python
* OpenCV
* Ultralytics YOLO
* NumPy

## Project Structure

```
EmotionDetection/
│── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── best.pt
│
│── detect.py
│── requirements.txt
│── README.md
│── .gitignore
```

> The dataset is intentionally excluded from this repository because of its size.


## Supported Emotions

The model predicts the following emotions (depending on the dataset used during training):

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

## Requirements

Typical dependencies include:

* Python 3.10+
* OpenCV
* Ultralytics
* NumPy

Install everything using:

```bash
pip install -r requirements.txt
```

## Notes

* Ensure the webcam is connected before running the program.
* Good lighting improves detection accuracy.
* The trained model file (`best.pt`) is required.
* The dataset is not included in this repository.

## Future Improvements

* Emotion logging
* Multiple face tracking
* Emotion statistics dashboard
* Masked face emotion detection
* Edge device deployment
* GUI interface

## Author

**Jesmitha K**
