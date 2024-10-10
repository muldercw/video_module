Video Processing & Monitoring Application
This is a real-time video processing and monitoring web application built using Streamlit, OpenCV, and Clarifai. The app allows users to process videos from different sources, such as URLs, webcams, or streaming URLs, and apply object detection models to the video frames.

The app offers:

Support for multiple video input types (Standard Video URLs, Webcam, and Streaming URLs).
The ability to select different object detection models.
A flexible interface for selecting the number of frames to skip.
Real-time display of processed frames in a grid.
JSON responses with prediction results for detected objects.
Stop and start functionality for video processing.
Features
Real-time Video Processing: Use any video source (URLs, webcam, or streams) to perform object detection in real-time.
Object Detection Models: Choose from various object detection models hosted on Clarifai.
Frame Skipping: Adjust the frame skip value to control the processing speed and performance.
Multiple Stream Support: Process and monitor multiple video streams simultaneously.
JSON Response Display: View the model's JSON predictions and detection results directly in the app.
Background Subtraction: Detect foreground objects using background subtraction with OpenCV's MOG2 algorithm.
