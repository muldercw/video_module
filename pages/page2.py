import streamlit as st
import cv2
import os
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

import json
from google.protobuf import json_format



def list_models():
    app_obj = App(user_id=userDataObject.user_id, app_id=userDataObject.app_id)
    all_models = list(app_obj.list_models())
    usermodels = []
    for model in all_models:
        model_url = f"https://clarifai.com/{userDataObject.user_id}/{userDataObject.app_id}/models/{model.id}"
        _umod = {"Name": model.id, "URL": model_url, "type": "User"}
        usermodels.append(_umod)
    return list_community_models() + usermodels + list_custom_python_models()

def list_community_models():
   return [
        {"Name": "General-Image-Detection", "URL": "https://clarifai.com/clarifai/main/models/general-image-detection", "type": "Community"},
        {"Name": "Face Detection", "URL": "https://clarifai.com/clarifai/main/models/face-detection", "type": "Community"},
        {"Name": "Weapon Detection", "URL": "https://clarifai.com/clarifai/main/models/weapon-detection", "type": "Community"},
        {"Name": "Vehicle Detection", "URL": "https://clarifai.com/clarifai/Roundabout-Aerial-Images-for-Vehicles-Det-Kaggle/models/vehicle-detector-alpha-x", "type": "Community"},
    ]

def list_custom_python_models():
    return [
        {"Name": "Movement", "URL": "xx", "type": "Movement"},
        {"Name": "Disable Detections", "URL": "xx", "type": "Disabled"}
    ]

def compensate_camera_motion(prev_frame, current_frame):

    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints_prev, descriptors_prev = orb.detectAndCompute(gray_prev, None)
    keypoints_current, descriptors_current = orb.detectAndCompute(gray_current, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_prev, descriptors_current)
    matches = sorted(matches, key=lambda x: x.distance)
    points_prev = np.float32([keypoints_prev[m.queryIdx].pt for m in matches])
    points_current = np.float32([keypoints_current[m.trainIdx].pt for m in matches])
    H, _ = cv2.findHomography(points_current, points_prev, cv2.RANSAC)
    stabilized_frame = cv2.warpPerspective(current_frame, H, (current_frame.shape[1], current_frame.shape[0]))
    return stabilized_frame

def movement_detection(overlay, overlay_counter, background_subtractor, prev_frame, current_frame, threshold=25):
    if not isinstance(current_frame, np.ndarray):
        raise ValueError("The 'frame' is not a valid numpy array.")
    _frame = current_frame.copy()
    overlay_decay = 3
    try:
        if prev_frame is not None:
            stabilized_frame = compensate_camera_motion(prev_frame, current_frame)
        else:
            stabilized_frame = current_frame.copy()
        if overlay is None:
            overlay = np.zeros_like(_frame)
        elif not isinstance(overlay, np.ndarray):
            overlay = np.zeros_like(_frame)
        foreground_mask = background_subtractor.apply(stabilized_frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        filtered_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)  # Closing to fill gaps
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel)     # Opening to remove noise
        contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = [c for c in contours if cv2.contourArea(c) > threshold]
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                draw_box_corners(overlay, x, y, x + w, y + h, (0, 255, 0), thickness=2, corner_length=15)
                overlay_counter = overlay_decay
        if overlay_counter > 0:
            combined_frame = cv2.addWeighted(_frame, 1, overlay, 1, 0)
            overlay_counter -= 1
        else:
            combined_frame = _frame
        cv2.putText(combined_frame, "Movement Detection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return overlay, overlay_counter, combined_frame, None

    except Exception as e:
        st.error(str(e))
        error_message = str(e)
        wrapped_text = "\n".join([error_message[i:i + 40] for i in range(0, len(error_message), 40)])
        y0, dy = 50, 20
        for i, line in enumerate(wrapped_text.split('\n')):
            y = y0 + i * dy
            cv2.putText(_frame, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return overlay, overlay_counter, _frame, None



def draw_box_corners(frame, left, top, right, bottom, color, thickness=1, corner_length=15):
    cv2.line(frame, (left, top), (left + corner_length, top), color, thickness)  # horizontal
    cv2.line(frame, (left, top), (left, top + corner_length), color, thickness)  # vertical
    cv2.line(frame, (right, top), (right - corner_length, top), color, thickness)  # horizontal
    cv2.line(frame, (right, top), (right, top + corner_length), color, thickness)  # vertical
    cv2.line(frame, (left, bottom), (left + corner_length, bottom), color, thickness)  # horizontal
    cv2.line(frame, (left, bottom), (left, bottom - corner_length), color, thickness)  # vertical
    cv2.line(frame, (right, bottom), (right - corner_length, bottom), color, thickness)  # horizontal
    cv2.line(frame, (right, bottom), (right, bottom - corner_length), color, thickness)  # vertical

def run_model_inference(det_threshold, background_subtractor, overlay, overlay_counter, prev_frame, frame, model_option, color=(0, 255, 0)):
    try:
      if model_option['type'] == "Disabled":
          _disabled_frame = frame.copy()
          cv2.putText(_disabled_frame, "Detections Disabled", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
          return overlay, overlay_counter, _disabled_frame, None
      elif model_option['type'] == "Movement":
          return movement_detection(overlay, overlay_counter, background_subtractor, frame, threshold=25)
      else:
        _frame = frame.copy()
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        model_url = model_option['URL']
        detector_model = Model(url=model_url)
        cv2.putText(_frame, model_option['Name'], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        prediction_response = detector_model.predict_by_bytes(frame_bytes, input_type="image")
        regions = prediction_response.outputs[0].data.regions

        for region in regions:
            top_row = round(region.region_info.bounding_box.top_row, 3)
            left_col = round(region.region_info.bounding_box.left_col, 3)
            bottom_row = round(region.region_info.bounding_box.bottom_row, 3)
            right_col = round(region.region_info.bounding_box.right_col, 3)
            left = int(left_col * frame.shape[1])
            top = int(top_row * frame.shape[0])
            right = int(right_col * frame.shape[1])
            bottom = int(bottom_row * frame.shape[0])
            for concept in region.data.concepts:
                name = concept.name
                value = round(concept.value, 4)
                if value < det_threshold:
                    continue
                text_position = (left + (right - left) // 4, top - 10)
                cv2.putText(_frame, f"{name}:{value}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                draw_box_corners(_frame, left, top, right, bottom, color)

        return overlay, overlay_counter, _frame, prediction_response
    except Exception as e:
      st.success(e)
      cv2.putText(frame, f"{e}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
      return overlay, overlay_counter, frame, None

def redraw_detections(previous_response, frame, model_option, color=(0, 255, 0)):
    if model_option['type'] == "disabled":
        return frame, None
    _frame = frame.copy()
    cv2.putText(_frame, model_option['Name'], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    prediction_response = previous_response
    regions = prediction_response.outputs[0].data.regions

    for region in regions:
        top_row = round(region.region_info.bounding_box.top_row, 3)
        left_col = round(region.region_info.bounding_box.left_col, 3)
        bottom_row = round(region.region_info.bounding_box.bottom_row, 3)
        right_col = round(region.region_info.bounding_box.right_col, 3)
        left = int(left_col * frame.shape[1])
        top = int(top_row * frame.shape[0])
        right = int(right_col * frame.shape[1])
        bottom = int(bottom_row * frame.shape[0])
        draw_box_corners(_frame, left, top, right, bottom, color)
        for concept in region.data.concepts:
            name = concept.name
            value = round(concept.value, 4)
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
auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()
st.title("Video Processing & Monitoring")


json_responses = []

st.subheader("Video Frame Processing")
video_option = st.radio("Choose Video Input:", ("Standard Video File URLs","Streaming Video", "Webcam", "Youtube Streaming[beta]"), horizontal=True) #, "Webcam", "Streaming Video URLs"

if video_option == "Webcam":
    st.info("Note: The webcam feature may not work on all devices.")
    enable = st.checkbox("Enable camera")
    webcam_input = st.camera_input("Capture a frame from your webcam:", disabled=not enable)
    if webcam_input:
        st.image(webcam_input)
else:
    video_urls = st.text_area("Enter video URLs (one per line):",
                               value="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4\nhttp://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4")

    frame_skip = st.slider("Select how many frames to skip:", min_value=1, max_value=120, value=28)
    det_threshold = st.slider("Select detection threshold:", min_value=0.01, max_value=1.00, value=0.5)
    available_models = list_models()
    url_list = [url.strip() for url in video_urls.split('\n') if url.strip()]
    model_options = []
    for idx, url in enumerate(url_list):
        model_names = [model["Name"] for model in available_models]
        selected_model_name = st.selectbox(f"Select a model for Video {idx + 1}:", model_names, key=f"model_{idx}")
        selected_model = next(model for model in available_models if model["Name"] == selected_model_name)
        model_options.append(selected_model)
    stop_event = threading.Event()
    if st.button("Stop Processing"):
        stop_event.set()
    if st.button("Process Videos") and not stop_event.is_set():
        frame_placeholder = st.empty()
        try:
          video_buffers = [deque(maxlen=6) for _ in range(len(url_list))]  # Buffer for the latest 2 frames
          threads = []
          def process_video(video_url, index, model_option, stop_event):
              background_subtractor = cv2.createBackgroundSubtractorMOG2(history=10000, varThreshold=40, detectShadows=False)
              overlay = None
              overlay_decay = 3  
              overlay_counter = 0
              prev_frame = None
              video_capture = cv2.VideoCapture(video_url)
              if not video_capture.isOpened():
                  st.error(f"Error: Could not open video at {video_url}.")
                  return
              frame_count = 0 
              while video_capture.isOpened() and not stop_event.is_set():
                  ret, frame = video_capture.read()
                  frame = cv2.resize(frame, (640, 280))
                  if prev_frame is None:
                      prev_frame = frame

                  if not ret:
                      break
                  if frame_count % frame_skip == 0:
                      overlay, overlay_counter, processed_frame, prediction_response = run_model_inference(det_threshold,background_subtractor, overlay, overlay_counter, prev_frame, frame, model_option)
                      if prediction_response:
                          json_responses.append(json_format.MessageToJson(prediction_response))
                      rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                      video_buffers[index].append(rgb_frame)
                  frame_count += 1
                  prev_frame = frame
              video_capture.release()
          for index, (video_url, model_option) in enumerate(zip(url_list, model_options)):
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
                      if len(grid_frames) % 4 != 0:
                        blank_frame = np.zeros_like(grid_frames[-1])
                        while len(grid_frames) % 4 != 0:
                          grid_frames.append(blank_frame)
                      grid_image = np.concatenate([np.concatenate(grid_frames[i:i + 4], axis=1) for i in range(0, len(grid_frames), 4)], axis=0)
                  frame_placeholder.image(grid_image, caption="Processed Video Frames")
              time.sleep(0.01)
          for thread in threads:
              thread.join()
        except Exception as e:
          st.error(e)
          json_responses.append(f"Error {e} processing video at {video_url}")
        st.success(json_responses)
    verify_json_responses()

  
