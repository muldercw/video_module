import streamlit as st
import cv2
import numpy as np
import threading
import subprocess
import os
import sys
from collections import deque
from clarifai.client.auth import create_stub
from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.client.model import Model
from clarifai.client.app import App
from clarifai.modules.css import ClarifaiStreamlitCSS
from google.protobuf import json_format

def check_ffmpeg_installed():
    """Check if FFmpeg is installed, install it if not."""
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False

def install_ffmpeg():
    """Install FFmpeg."""
    if sys.platform.startswith('linux'):
        st.write("Installing FFmpeg on Linux...")
        subprocess.run(['sudo', 'apt-get', 'update'], check=True)
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'ffmpeg'], check=True)
    elif sys.platform == 'darwin':  # macOS
        st.write("Installing FFmpeg on macOS...")
        subprocess.run(['brew', 'install', 'ffmpeg'], check=True)
    elif sys.platform == 'win32':
        st.write("FFmpeg is not available for Windows. Please install manually from https://ffmpeg.org/download.html")
        st.stop()
    else:
        st.write("Unsupported OS. Please install FFmpeg manually.")
        st.stop()

# Check for FFmpeg installation
if not check_ffmpeg_installed():
    install_ffmpeg()

# Continue with the rest of your script...
def list_models():
    app_obj = App(user_id=userDataObject.user_id, app_id=userDataObject.app_id)
    all_models = list(app_obj.list_models())
    usermodels = []
    for model in all_models:
        model_url = f"https://clarifai.com/{userDataObject.user_id}/{userDataObject.app_id}/models/{model.id}"
        _umod = {"Name": model.id, "URL": model_url, "type": "User"}
        usermodels.append(_umod)
    return usermodels + list_community_models()

def list_community_models():
    predefined_model_urls = [
        {"Name": "General-Image-Detection", "URL": "https://clarifai.com/clarifai/main/models/general-image-detection", "type": "Community"},
        {"Name": "Face Detection", "URL": "https://clarifai.com/clarifai/main/models/face-detection", "type": "Community"},
        {"Name": "Disable Detections", "URL": "xx", "type": "disabled"}
    ]
    return predefined_model_urls

def run_model_inference(frame, model_option):
    if model_option['type'] == "disabled":
        return frame, None
    _frame = frame.copy()
    frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
    model_url = model_option['URL']
    detector_model = Model(url=model_url)
    cv2.putText(_frame, model_option['Name'], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    prediction_response = detector_model.predict_by_bytes(frame_bytes, input_type="image")
    regions = prediction_response.outputs[0].data.regions
    for region in regions:
        top_row = round(region.region_info.bounding_box.top_row, 3)
        left_col = round(region.region_info.bounding_box.left_col, 3)
        bottom_row = round(region.region_info.bounding_box.bottom_row, 3)
        right_col = round(region.region_info.bounding_box.right_col, 3)

        for concept in region.data.concepts:
            name = concept.name
            value = round(concept.value, 4)
            cv2.putText(_frame, f"{name}:{value}", (int(left_col * frame.shape[1]), int(top_row * frame.shape[0]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(_frame, (int(left_col * frame.shape[1]), int(top_row * frame.shape[0])),
                                  (int(right_col * frame.shape[1]), int(bottom_row * frame.shape[0])), (0, 255, 0), 2)
    return _frame, prediction_response

st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)

# Authentication and Clarifai setup
auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()

st.title("Video Processing & Monitoring")

# Collapsible JSON results display
json_responses = []

def display_json_responses():
    if st.checkbox("Show JSON Results", value=False):
        st.subheader("Model Predictions (JSON Responses)")
        for idx, response in enumerate(json_responses):
            st.json(response)

# Section for playing and processing video frames
st.subheader("Video Frame Processing")
video_option = st.radio("Choose Video Input:", ("Multiple Video URLs", "Webcam", "RTMP Stream", "UDP Stream"), horizontal=True)

if video_option == "Webcam":
    # Option to capture video from webcam
    webcam_input = st.camera_input("Capture a frame from your webcam:")

    if webcam_input:
        # Read the uploaded image
        frame = cv2.imdecode(np.frombuffer(webcam_input.read(), np.uint8), cv2.IMREAD_COLOR)

        # Process the frame (if needed)
        frame = cv2.putText(frame, "Webcam Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Convert the frame from BGR to RGB (for displaying in Streamlit)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(rgb_frame, caption="Processed Webcam Frame")

elif video_option in ["RTMP Stream", "UDP Stream"]:
    # Input for streaming URL
    stream_url = st.text_input("Enter the streaming URL:")

    if st.button("Start Stream"):
        if stream_url:
            # Start processing the RTMP or UDP stream
            frame_placeholder = st.empty()
            stop_event = threading.Event()

            # Function to process the stream using FFmpeg
            def process_stream(stream_url, stop_event):
                command = [
                    'ffmpeg',
                    '-i', stream_url,
                    '-f', 'rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-an', '-sn', '-vcodec', 'rawvideo', '-y', '-'
                ]
                pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)

                while True:
                    if stop_event.is_set():
                        break

                    raw_frame = pipe.stdout.read(640 * 480 * 3)  # Adjust based on resolution
                    if not raw_frame:
                        break

                    frame = np.frombuffer(raw_frame, np.uint8).reshape((480, 640, 3))  # Adjust based on resolution

                    # Run inference on the frame (you can choose a default model for streaming)
                    model_option = {"Name": "General-Image-Detection", "URL": "https://clarifai.com/clarifai/main/models/general-image-detection", "type": "Community"}
                    processed_frame, prediction_response = run_model_inference(frame, model_option)

                    if prediction_response:
                        # Append prediction results to JSON responses
                        json_responses.append(json_format.MessageToJson(prediction_response))

                    # Convert the frame from BGR to RGB (for displaying in Streamlit)
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                    # Display the processed frame
                    frame_placeholder.image(rgb_frame, caption="Streaming Frame")

                pipe.terminate()

            # Start the streaming thread
            thread = threading.Thread(target=process_stream, args=(stream_url, stop_event))
            thread.start()

            # Stop processing button
            if st.button("Stop Stream"):
                stop_event.set()
                thread.join()

else:
    # Input for multiple video URLs with prepopulated example URLs
    video_urls = st.text_area("Enter video URLs (one per line):",
                               value="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4\nhttp://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4")

    # Slider for frame skip selection
    frame_skip = st.slider("Select how many frames to skip:", min_value=1, max_value=20, value=2)

    # Obtain models from list_models()
    available_models = list_models()

    # Create a model selector for each video URL
    url_list = [url.strip() for url in video_urls.split('\n') if url.strip()]
    model_options = []
    for idx, url in enumerate(url_list):
        model_names = [model["Name"] for model in available_models]
        selected_model_name = st.selectbox(f"Select a model for Video {idx+1}", model_names, key=f"model_{idx}")
        selected_model = next(model for model in available_models if model["Name"] == selected_model_name)
        model_options.append(selected_model)

    if st.button("Process Videos"):
        video_buffers = [deque(maxlen=20) for _ in range(len(url_list))]  # Buffer for storing recent frames
        threads = []
        stop_event = threading.Event()

        # Function to process each video
        def process_video(video_url, index, model_option, stop_event):
            video_capture = cv2.VideoCapture(video_url)

            frame_count = 0  # Counter for frames processed
            frame_placeholder = st.empty()  # Placeholder for displaying video frames

            while True:
                if stop_event.is_set():
                    break

                ret, frame = video_capture.read()
                if not ret:
                    break

                # Run model inference on selected frames
                if frame_count % frame_skip == 0:
                    processed_frame, prediction_response = run_model_inference(frame, model_option)
                    video_buffers[index].append(processed_frame)

                    if prediction_response:
                        json_responses.append(json_format.MessageToJson(prediction_response))

                    # Display the processed frame
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb_frame, caption=f"Processed Video Frame from {video_url}")

                frame_count += 1  # Increment frame count

            video_capture.release()

        # Start processing videos
        for idx, (url, model_option) in enumerate(zip(url_list, model_options)):
            thread = threading.Thread(target=process_video, args=(url, idx, model_option, stop_event))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

# Display JSON results
display_json_responses()
