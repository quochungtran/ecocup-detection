# Object Detection of Ecocups using Deep Learning Techniques

This project aims to develop a deep learning model for object detection of Ecocups in images and videos. The project utilizes three different methods: sliding window, Mask R-CNN, and YOLO, and includes a Python implementation for each method.

## Methodology

- Sliding Window: This method involves scanning the image or video with a small window and classifying each window as containing an Ecocup or not. The implementation includes a Python script that performs sliding window object detection using a pre-trained deep learning model.

- Mask R-CNN: This method is a more advanced approach that involves instance segmentation. The implementation includes a Python script that performs object detection using a pre-trained Mask R-CNN model.

- YOLO: This method stands for You Only Look Once, and is a real-time object detection system. The implementation includes a Python script that fine-tunes a pre-trained YOLO model on the Ecocup dataset and performs object detection on new images and videos.

## Dataset

The dataset used in this project consists of images and videos containing Ecocups. The dataset was manually annotated to label the location of each Ecocup in the images and videos. The annotated dataset is available upon request.

## Requirements

The implementation of each method requires the following libraries:

- OpenCV
- NumPy
- TensorFlow (for Mask R-CNN and YOLO)

## Usage

To use the object detection implementation of a specific method, navigate to the corresponding folder and run the Python script. The script will take an input image or video and output the same file with the detected Ecocups highlighted.

## Credits

This project was developed by [Your Name] as part of [Project Name]. The Mask R-CNN implementation is based on the Matterport implementation (https://github.com/matterport/Mask_RCNN). The YOLO implementation is based on the Darknet implementation (https://github.com/pjreddie/darknet).
