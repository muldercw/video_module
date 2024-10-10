import streamlit as st
import cv2
import numpy as np
import threading
import time
import subprocess
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
        {"Name": "Disable Detections", "URL": "xx", "type": "Disabled"},
        {"Name": "Weapon Detection", "URL": "https://clarifai.com/clarifai/main/models/weapon-detection", "type": "Community"},
        {"Name": "Vehicle Detection", "URL": "https://clarifai.com/clarifai/Roundabout-Aerial-Images-for-Vehicles-Det-Kaggle/models/vehicle-detector-alpha-x", "type": "Community"},
        {"Name": "Movement", "URL": "xx", "type": "Movement"}
    ]
    return predefined_model_urls

def movement_detection(overlay, overlay_counter, background_subtractor, frame, threshold=25):
    try:
      foreground_mask = background_subtractor.apply(frame)
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
      filtered_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)  # Closing to fill gaps
      filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel)     # Opening to remove noise
      contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      if contours:
          contours = [c for c in contours if cv2.contourArea(c) > threshold]  # Threshold to ignore small contours
          if contours:
              largest_contour = max(contours, key=cv2.contourArea)
              x, y, w, h = cv2.boundingRect(largest_contour)
              if overlay is None:
                  overlay = np.zeros_like(frame)
              cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
              overlay_counter = overlay_decay
      if overlay_counter > 0:
          combined_frame = cv2.addWeighted(frame, 1, overlay, 1, 0)
          overlay_counter -= 1
      else:
          combined_frame = frame
      return overlay, overlay_counter, combined_frame, None
    except Exception as e:
      print(e)
      return overlay, overlay_counter, frame, None


def draw_box_corners(frame, left, top, right, bottom, color, thickness=2, corner_length=10):
    # Top-left corner
    cv2.line(frame, (left, top), (left + corner_length, top), color, thickness)  # horizontal
    cv2.line(frame, (left, top), (left, top + corner_length), color, thickness)  # vertical

    # Top-right corner
    cv2.line(frame, (right, top), (right - corner_length, top), color, thickness)  # horizontal
    cv2.line(frame, (right, top), (right, top + corner_length), color, thickness)  # vertical

    # Bottom-left corner
    cv2.line(frame, (left, bottom), (left + corner_length, bottom), color, thickness)  # horizontal
    cv2.line(frame, (left, bottom), (left, bottom - corner_length), color, thickness)  # vertical

    # Bottom-right corner
    cv2.line(frame, (right, bottom), (right - corner_length, bottom), color, thickness)  # horizontal
    cv2.line(frame, (right, bottom), (right, bottom - corner_length), color, thickness)  # vertical

def run_model_inference(background_subtractor,overlay, overlay_counter, frame, model_option, color=(0, 255, 0)):
    if model_option['type'] == "Disabled":
        return overlay, overlay_counter, frame, None
    if model_option['type'] == "Movement":
        return movement_detection(background_subtractor,overlay, overlay_counter, frame, None)

    _frame = frame.copy()
    frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
    model_url = model_option['URL']
    detector_model = Model(url=model_url)

    # Put model name at top
    cv2.putText(_frame, model_option['Name'], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Perform prediction
    prediction_response = detector_model.predict_by_bytes(frame_bytes, input_type="image")
    regions = prediction_response.outputs[0].data.regions

    for region in regions:
        top_row = round(region.region_info.bounding_box.top_row, 3)
        left_col = round(region.region_info.bounding_box.left_col, 3)
        bottom_row = round(region.region_info.bounding_box.bottom_row, 3)
        right_col = round(region.region_info.bounding_box.right_col, 3)

        # Get absolute coordinates
        left = int(left_col * frame.shape[1])
        top = int(top_row * frame.shape[0])
        right = int(right_col * frame.shape[1])
        bottom = int(bottom_row * frame.shape[0])

        # Draw corners of the box
        draw_box_corners(_frame, left, top, right, bottom, color)

        for concept in region.data.concepts:
            name = concept.name
            value = round(concept.value, 4)

            # Place text between top corners
            text_position = (left + (right - left) // 4, top - 10)
            cv2.putText(_frame, f"{name}:{value}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    return _frame, prediction_response

def redraw_detections(previous_response, frame, model_option, color=(0, 255, 0)):
    if model_option['type'] == "disabled":
        return frame, None

    _frame = frame.copy()
    frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
    model_url = model_option['URL']
    detector_model = Model(url=model_url)

    # Put model name at top
    cv2.putText(_frame, model_option['Name'], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Perform prediction
    prediction_response = previous_response
    regions = prediction_response.outputs[0].data.regions

    for region in regions:
        top_row = round(region.region_info.bounding_box.top_row, 3)
        left_col = round(region.region_info.bounding_box.left_col, 3)
        bottom_row = round(region.region_info.bounding_box.bottom_row, 3)
        right_col = round(region.region_info.bounding_box.right_col, 3)

        # Get absolute coordinates
        left = int(left_col * frame.shape[1])
        top = int(top_row * frame.shape[0])
        right = int(right_col * frame.shape[1])
        bottom = int(bottom_row * frame.shape[0])

        # Draw corners of the box
        draw_box_corners(_frame, left, top, right, bottom, color)

        for concept in region.data.concepts:
            name = concept.name
            value = round(concept.value, 4)

            # Place text between top corners
            text_position = (left + (right - left) // 4, top - 10)
            cv2.putText(_frame, f"{name}:{value}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    return _frame, prediction_response

def verify_json_responses():
    if st.checkbox("Show JSON Results", value=False):
        st.subheader("Model Predictions (JSON Responses)")
        for idx, response in enumerate(json_responses):
            st.json(response)

st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)



# Authentication and Clarifai setup
auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()

background_subtractor = cv2.createBackgroundSubtractorMOG2(history=10000, varThreshold=40, detectShadows=False)
overlay = None
overlay_decay = 3  
overlay_counter = 0

st.title("Video Processing & Monitoring")

# Collapsible JSON results display
json_responses = []

# Section for playing and processing video frames
st.subheader("Video Frame Processing")
video_option = st.radio("Choose Video Input:", ("Standard Video File URLs","Webcam"), horizontal=True) #, "Webcam", "Streaming Video URLs"

if video_option == "Webcam":
    # Option to capture video from webcam
    enable = st.checkbox("Enable camera")
    webcam_input = st.camera_input("Capture a frame from your webcam:", disabled=not enable)
    if webcam_input:
        st.image(webcam_input)

    # if webcam_input:
    #     # Read the uploaded image
    #     frame = cv2.imdecode(np.frombuffer(webcam_input.read(), np.uint8), cv2.IMREAD_COLOR)

    #     # Process the frame (if needed)
    #     frame = cv2.putText(frame, "Webcam Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    #     # Convert the frame from BGR to RGB (for displaying in Streamlit)
    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     st.image(rgb_frame, caption="Processed Webcam Frame")
elif video_option == "Streaming Video":
    # Input for streaming video URL
    stream_urls = st.text_area("Enter video Streams (one per line):",
                               value="https://vs-dash-ww-rd-live.akamaized.net/pl/testcard2020/avc-mobile.m3u8\nrtsp://1701954d6d07.entrypoint.cloud.wowza.com:1935/app-m75436g0/27122ffc_stream2")
    frame_skip = st.slider("Select how many frames to skip:", min_value=1, max_value=20, value=2)
    available_models = list_models()
    stream_list = [url.strip() for url in stream_urls.split('\n') if url.strip()]
    model_options = []
    for idx, url in enumerate(stream_list):
        model_names = [model["Name"] for model in available_models]
        selected_model_name = st.selectbox(f"Select a model for Stream {idx + 1}:", model_names, key=f"model_{idx}")
        selected_model = next(model for model in available_models if model["Name"] == selected_model_name)
        model_options.append(selected_model)
    stop_event = threading.Event()
    if st.button("Stop Processing", style="danger"):
        stop_event.set()
    if st.button("Process Streams") and not stop_event.is_set():
        # use ffmpeg to stream the video
        video_buffers = [deque(maxlen=2) for _ in range(len(stream_list))]
        threads = []
        def process_video(video_url, index, model_option, stop_event):
            try:##ffmpeg to read the stream
              command = ['ffmpeg', '-i', video_url, '-f', 'image2pipe', '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-']
              process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
              frame_count = 0
              while not stop_event.is_set():
                  raw_frame = process.stdout.read(640 * 480 * 3)
                  if len(raw_frame) == 0:
                      break
                  frame = np.frombuffer(raw_frame, np.uint8).reshape(480, 640, 3)
                  if frame_count % frame_skip == 0:
                      overlay, overlay_counter, processed_frame, prediction_response = run_model_inference(background_subtractor, overlay, overlay_counter, frame, model_option)
                      if prediction_response:
                          json_responses.append(json_format.MessageToJson(prediction_response))
                      rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                      video_buffers[index].append(rgb_frame)
                  frame_count += 1
              process.kill()
            except Exception as e:
              print(e)
              json_responses.append(f"Error {e} processing video at {video_url}")

        for index, (video_url, model_option) in enumerate(zip(stream_list, model_options)):
            thread = threading.Thread(target=process_video, args=(video_url, index, model_option, stop_event))
            thread.start()
            threads.append(thread)
        while any(thread.is_alive() for thread in threads):
            grid_frames = []
            for index in range(len(video_buffers)):
                if len(video_buffers[index]) > 0:
                    grid_frames.append(video_buffers[index][-1])
            if grid_frames:
                if len(grid_frames) == 1:
                    grid_image = grid_frames[0]
                else:
                    if len(grid_frames) % 2 != 0:
                        blank_frame = np.zeros_like(grid_frames[-1])
                        grid_frames.append(blank_frame)
                    grid_image = np.concatenate([np.concatenate(grid_frames[i:i + 2], axis=1) for i in range(0, len(grid_frames), 2)], axis=0)
                st.image(grid_image, caption="Processed Video Frames")
            time.sleep(0.1)
        for thread in threads:
            thread.join()
        st.success("Video processing completed!")
    verify_json_responses()
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
        selected_model_name = st.selectbox(f"Select a model for Video {idx + 1}:", model_names, key=f"model_{idx}")
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
        video_buffers = [deque(maxlen=6) for _ in range(len(url_list))]  # Buffer for the latest 2 frames
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
                frame = cv2.resize(frame, (320, 240))
                #previous_response = None

                if not ret:
                    break  # Stop the loop when no more frames

                # Only process frames based on the user-selected frame skip
                if frame_count % frame_skip == 0:
                    # Run inference on the frame with the selected model
                    overlay, overlay_counter, processed_frame, prediction_response = run_model_inference(background_subtractor, overlay, overlay_counter, frame, model_option)
                    #previous_response = prediction_response

                    if prediction_response:
                        # Append prediction results to JSON responses
                        json_responses.append(json_format.MessageToJson(prediction_response))

                    # Convert the frame from BGR to RGB (for displaying in Streamlit)
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                    # Add the frame to the buffer
                    video_buffers[index].append(rgb_frame)
                # else:
                #     processed_frame, prediction_response = redraw_detections(previous_response, frame, model_option)
                #     previous_response = prediction_response
                #     rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                #     video_buffers[index].append(rgb_frame)


                frame_count += 1

            video_capture.release()

        # Start threads for each video URL with their corresponding model option
        for index, (video_url, model_option) in enumerate(zip(url_list, model_options)):
            thread = threading.Thread(target=process_video, args=(video_url, index, model_option, stop_event))
            thread.start()
            threads.append(thread)

        # Monitor threads and update the grid image
        while any(thread.is_alive() for thread in threads):
            grid_frames = []

            for index in range(len(video_buffers)):
                if len(video_buffers[index]) > 0:
                    grid_frames.append(video_buffers[index][-1])  # Append the latest frame

            if grid_frames:
                if len(grid_frames) == 1:
                    grid_image = grid_frames[0]  # Only one frame, show it directly
                else:
                    # Create grid layout (4 frames per row)
                    if len(grid_frames) % 4 != 0:
                      blank_frame = np.zeros_like(grid_frames[-1])  # Create a blank frame
                      while len(grid_frames) % 4 != 0:
                        grid_frames.append(blank_frame)  # Add blank frames to make it a multiple of 4

                    grid_image = np.concatenate([np.concatenate(grid_frames[i:i + 4], axis=1) for i in range(0, len(grid_frames), 4)], axis=0)

                frame_placeholder.image(grid_image, caption="Processed Video Frames")

            time.sleep(0.01)

        # Ensure all threads are finished
        for thread in threads:
            thread.join()

        st.success("Video processing completed!")

    verify_json_responses()

  
