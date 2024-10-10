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
from google.protobuf import json_format

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

            # Function to process the stream
            def process_stream(stream_url, stop_event):
                video_capture = cv2.VideoCapture(stream_url)

                if not video_capture.isOpened():
                    st.error(f"Error: Could not open stream at {stream_url}.")
                    return

                while video_capture.isOpened() and not stop_event.is_set():
                    ret, frame = video_capture.read()
                    if not ret:
                        break  # Stop the loop when no more frames

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

                video_capture.release()

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
        selected_model_name = st.selectbox(f"Select a model for Video {idx+1}:", model_names, key=f"model_{idx}")
        selected_model = next(model for model in available_models if model["Name"] == selected_model_name)
        model_options.append(selected_model)

    # Event to stop processing
    stop_event = threading.Event()

    # Stop processing button
    if st.button("Stop Processing"):
        stop_event.set()

    # Process video button
    if st.button("Process Videos") and not stop_event.is_set():
        frame_placeholder = st.empty()

        # Initialize a list to hold buffers and threads
        video_buffers = [deque(maxlen=2) for _ in range(len(url_list))]  # Buffer for the latest 2 frames
        threads = []

        # Function to process each video
        def process_video(video_url, index, model_option, stop_event):
            video_capture = cv2.VideoCapture(video_url)

            if not video_capture.isOpened():
                st.error(f"Error: Could not open video at {video_url}.")
                return

            frame_count = 0  # Initialize frame count

            while video_capture.isOpened() and not stop_event.is_set():
                ret, frame = video_capture.read()
                frame = cv2.resize(frame, (640, 480))

                if not ret:
                    break  # Stop the loop when no more frames

                # Only process frames based on the user-selected frame skip
                if frame_count % frame_skip == 0:
                    # Run inference on the frame with the selected model
                    processed_frame, prediction_response = run_model_inference(frame, model_option)

                    if prediction_response:
                        # Append prediction results to JSON responses
                        json_responses.append(json_format.MessageToJson(prediction_response))

                    # Convert the frame from BGR to RGB (for displaying in Streamlit)
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                    # Store the latest frame in the buffer
                    video_buffers[index].append(rgb_frame)

                    # Display the processed frame
                    frame_placeholder.image(rgb_frame, caption=f"Processed Video Frame {index + 1}")

                frame_count += 1

            video_capture.release()

        # Start processing each video in a separate thread
        for idx, url in enumerate(url_list):
            thread = threading.Thread(target=process_video, args=(url, idx, model_options[idx], stop_event))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

# Display the JSON responses
display_json_responses()
