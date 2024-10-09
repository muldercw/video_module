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
    try:
        app_obj = App(user_id=userDataObject.user_id, app_id=userDataObject.app_id)
        all_models = list(app_obj.list_models())
        usermodels = []
        for model in all_models:
            model_url = f"https://clarifai.com/{userDataObject.user_id}/{userDataObject.app_id}/models/{model.id}"
            _umod = {"Name": model.id, "URL": model_url, "type": "User"}
            usermodels.append(_umod)
        return list_community_models() + usermodels
    except Exception as e:
        st.error(f"Error listing models: {str(e)}")
        return []

def list_community_models():
    try:
        predefined_model_urls = [{"Name": "General-Image-Detection", "URL": "https://clarifai.com/clarifai/main/models/general-image-detection", "type":"Community"},
                                 {"Name": "Face Detection", "URL": "https://clarifai.com/clarifai/main/models/face-detection", "type":"Community"}]
        return predefined_model_urls
    except Exception as e:
        st.error(f"Error listing community models: {str(e)}")
        return []

def run_model_inference(frame, model_option):
    try:
        _frame = frame.copy()
        prediction_response = None
        
        # Perform inference logic here, currently returning "testing" as the response
        return _frame, "testing"
    except Exception as e:
        st.error(f"Error running model inference: {str(e)}")
        return frame, None

st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)

# Authentication and Clarifai setup
try:
    auth = ClarifaiAuthHelper.from_streamlit(st)
    stub = create_stub(auth)
    userDataObject = auth.get_user_app_id_proto()
except Exception as e:
    st.error(f"Error during authentication: {str(e)}")

st.title("Video Processing & Monitoring")

# Section for playing and processing video frames
st.subheader("Video Frame Processing")
video_option = st.radio("Choose Video Input:", ("Multiple Video URLs", "Webcam"), horizontal=True)

if video_option == "Webcam":
    try:
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
    except Exception as e:
        st.error(f"Error processing webcam input: {str(e)}")

else:
    try:
        # Input for multiple video URLs with prepopulated example URLs
        video_urls = st.text_area("Enter video URLs (one per line):",
                                   value="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4\nhttp://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4")

        # Slider for frame skip selection
        frame_skip = st.slider("Select frame inference interval:", min_value=10, max_value=30, value=1)

        # Obtain models from list_models()
        available_models = list_models()

        # Create a model selector for each video URL
        url_list = [url.strip() for url in video_urls.split('\n') if url.strip()]
        model_options = []
        for idx, url in enumerate(url_list):
            model_names = [model["Name"] for model in available_models]
            selected_model_name = st.selectbox(f"Select a model for Video {idx+1}:", model_names, key=f"model_{idx}")
            selected_model = next(model for model in available_models if model["Name"] == selected_model_name)
            model_options.append(selected_model)

        if st.button("Process Videos"):
            if video_urls and len(model_options) == len(url_list):
                # Create a placeholder for the grid
                frame_placeholder = st.empty()

                # Initialize a list to hold buffers and threads
                video_buffers = [deque(maxlen=2) for _ in range(len(url_list))]  # Buffer for the latest 2 frames
                threads = []

                # Function to process each video
                def process_video(video_url, index, model_option):
                    try:
                        video_capture = cv2.VideoCapture(video_url)

                        if not video_capture.isOpened():
                            st.error(f"Error: Could not open video at {video_url}.")
                            return

                        frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
                        if frame_skip > frame_rate:
                            frame_skip = frame_rate  # Ensure frame skip is within the frame rate
                        frame_count = 0  # Initialize frame count

                        while video_capture.isOpened():
                            ret, frame = video_capture.read()
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
                    except Exception as e:
                        st.error(f"Error processing video {video_url}: {str(e)}")

                # Start threads for each video URL with their corresponding model option
                for index, (video_url, model_option) in enumerate(zip(url_list, model_options)):
                    thread = threading.Thread(target=process_video, args=(video_url, index, model_option))
                    thread.start()
                    threads.append(thread)

                # Monitor threads and update the grid image
                while any(thread.is_alive() for thread in threads):
                    try:
                        grid_frames = []

                        for index in range(len(video_buffers)):
                            # If the buffer is filled, add the last frame to the grid
                            if len(video_buffers[index]) > 0:
                                grid_frames.append(video_buffers[index][-1])  # Append the latest frame

                        # Create a grid image from the frames
                        if grid_frames:
                            if len(grid_frames) == 1:
                                grid_image = grid_frames[0]  # Only one frame, show it directly
                            else:
                                if len(grid_frames) % 2 != 0:
                                    blank_frame = np.zeros_like(grid_frames[-1])  # Create a blank frame
                                    grid_frames.append(blank_frame)  # Add the blank frame if odd

                                grid_image = np.concatenate([np.concatenate(grid_frames[i:i+2], axis=1) for i in range(0, len(grid_frames), 2)], axis=0)

                            frame_placeholder.image(grid_image, caption="Video Frames Grid")

                        time.sleep(0.1)  # Slight delay to avoid overwhelming the stream
                    except Exception as e:
                        st.error(f"Error updating video grid: {str(e)}")

                # Wait for all threads to finish
                for thread in threads:
                    thread.join()
            else:
                st.warning("Please provide valid video URLs and select models for each video.")
    except Exception as e:
        st.error(f"Error in video URL processing section: {str(e)}")
