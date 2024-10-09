import streamlit as st
import cv2
import numpy as np
import threading
import time
from collections import deque
from clarifai.client.auth import create_stub
from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.client.user import User
from clarifai.client.model import Model
from clarifai.client.app import App
from clarifai.modules.css import ClarifaiStreamlitCSS
from google.protobuf import json_format, timestamp_pb2

def list_models():
    app_obj = App(user_id=userDataObject.user_id, app_id=userDataObject.app_id)
    all_models = list(app_obj.list_models())
    return [model.id for model in all_models]

def run_model_inference(frame, model_option):
    # Simulate model inference
    return cv2.putText(frame.copy(), model_option, (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2, cv2.LINE_AA), None

st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)

# Authentication and Clarifai setup
auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()

st.title("Video Processing & Monitoring")

# Section for playing and processing video frames
st.subheader("Video Frame Processing")
video_option = st.radio("Choose Video Input:", ("Webcam", "Multiple Video URLs"))

if video_option == "Webcam":
    # Option to capture video from webcam
    webcam_input = st.camera_input("Capture a frame from your webcam:")

    if webcam_input:
        # Read the uploaded image
        frame = cv2.imdecode(np.frombuffer(webcam_input.read(), np.uint8), cv2.IMREAD_COLOR)

        # Process the frame (if needed)
        frame = cv2.putText(frame, "Webcam Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2, cv2.LINE_AA)

        # Convert the frame from BGR to RGB (for displaying in Streamlit)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(rgb_frame, caption="Processed Webcam Frame")

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
        model_option = st.selectbox(f"Select a model for Video {idx+1}:", available_models, key=f"model_{idx}")
        model_options.append(model_option)

    if st.button("Process Videos"):
        if video_urls and len(model_options) == len(url_list):
            # Create a placeholder for the grid
            frame_placeholder = st.empty()

            # Initialize a list to hold buffers and threads
            video_buffers = [deque(maxlen=2) for _ in range(len(url_list))]  # Buffer for the latest 2 frames
            threads = []

            # Function to process each video
            def process_video(video_url, index, model_option):
                video_capture = cv2.VideoCapture(video_url)

                if not video_capture.isOpened():
                    st.error(f"Error: Could not open video at {video_url}.")
                    return

                frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
                frame_count = 0  # Initialize frame count

                while video_capture.isOpened():
                    ret, frame = video_capture.read()
                    # resize the frame to 640x480
                    frame = cv2.resize(frame, (640, 480))

                    if not ret:
                        break  # Stop the loop when no more frames

                    # Only process frames based on the user-selected frame skip
                    if frame_count % frame_skip == 0:
                        # Run inference on the frame with the selected model
                        processed_frame, model_response = run_model_inference(frame, model_option)

                        # Convert the frame from BGR to RGB (for displaying in Streamlit)
                        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                        # Add the frame to the buffer
                        video_buffers[index].append(rgb_frame)

                    frame_count += 1

                video_capture.release()

            # Start threads for each video URL with their corresponding model option
            for index, (video_url, model_option) in enumerate(zip(url_list, model_options)):
                thread = threading.Thread(target=process_video, args=(video_url, index, model_option))
                thread.start()
                threads.append(thread)

            # Monitor threads and update the grid image
            while any(thread.is_alive() for thread in threads):
                grid_frames = []

                for index in range(len(video_buffers)):
                    # If the buffer is filled, add the last frame to the grid
                    if len(video_buffers[index]) > 0:
                        grid_frames.append(video_buffers[index][-1])  # Append the latest frame

                # Create a grid image from the frames
                if grid_frames:
                    # If there's an odd number of frames, duplicate the last frame for even grid
                    if len(grid_frames) == 1:
                        grid_image = grid_frames[0]  # Only one frame, show it directly
                    else:
                        # Create grid layout (2 frames per row)
                        if len(grid_frames) % 2 != 0:
                            blank_frame = np.zeros_like(grid_frames[-1])  # Create a blank frame
                            grid_frames.append(blank_frame)  # Add the blank frame if odd

                        grid_image = np.concatenate([np.concatenate(grid_frames[i:i+2], axis=1) for i in range(0, len(grid_frames), 2)], axis=0)

                    # Display the grid image
                    frame_placeholder.image(grid_image, caption="Video Frames Grid")

                time.sleep(0.1)  # Slight delay to avoid overwhelming the stream

            # Wait for all threads to finish
            for thread in threads:
                thread.join()

        else:
            st.warning("Please provide valid video URLs and select models for each video.")
