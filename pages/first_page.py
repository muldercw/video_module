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
    # Convert frames to grayscale for feature matching
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    keypoints_prev, descriptors_prev = orb.detectAndCompute(gray_prev, None)
    keypoints_current, descriptors_current = orb.detectAndCompute(gray_current, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_prev, descriptors_current)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    points_prev = np.float32([keypoints_prev[m.queryIdx].pt for m in matches])
    points_current = np.float32([keypoints_current[m.trainIdx].pt for m in matches])

    # Find the transformation matrix for camera motion compensation
    H, _ = cv2.findHomography(points_current, points_prev, cv2.RANSAC)

    # Warp the current frame to align with the previous frame
    stabilized_frame = cv2.warpPerspective(current_frame, H, (current_frame.shape[1], current_frame.shape[0]))

    return stabilized_frame

def movement_detection(overlay, overlay_counter, background_subtractor, prev_frame, current_frame, threshold=25):
    if not isinstance(current_frame, np.ndarray):
        raise ValueError("The 'frame' is not a valid numpy array.")
    
    _frame = current_frame.copy()
    overlay_decay = 3

    try:
        # Compensate for camera motion before applying background subtraction
        if prev_frame is not None:
            stabilized_frame = compensate_camera_motion(prev_frame, current_frame)
        else:
            stabilized_frame = current_frame.copy()

        # Ensure overlay is properly initialized
        if overlay is None:
            overlay = np.zeros_like(_frame)
        elif not isinstance(overlay, np.ndarray):
            overlay = np.zeros_like(_frame)

        # Apply background subtraction to stabilized frame
        foreground_mask = background_subtractor.apply(stabilized_frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        filtered_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)  # Closing to fill gaps
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel)     # Opening to remove noise

        # Find contours in the mask
        contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Filter out small contours based on the area threshold
            contours = [c for c in contours if cv2.contourArea(c) > threshold]
            if contours:
                # Get the largest contour and its bounding box
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Draw a rectangle or custom corner box on the overlay
                draw_box_corners(overlay, x, y, x + w, y + h, (0, 255, 0), thickness=2, corner_length=15)
                overlay_counter = overlay_decay
        
        # Combine overlay with the frame if overlay_counter is active
        if overlay_counter > 0:
            combined_frame = cv2.addWeighted(_frame, 1, overlay, 1, 0)
            overlay_counter -= 1
        else:
            combined_frame = _frame
        
        # Add label text to the frame
        cv2.putText(combined_frame, "Movement Detection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return overlay, overlay_counter, combined_frame, None

    except Exception as e:
        st.error(str(e))
        error_message = str(e)

        # Add error message to the frame for debugging purposes
        wrapped_text = "\n".join([error_message[i:i + 40] for i in range(0, len(error_message), 40)])
        y0, dy = 50, 20
        for i, line in enumerate(wrapped_text.split('\n')):
            y = y0 + i * dy
            cv2.putText(_frame, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        return overlay, overlay_counter, _frame, None


import json
from google.protobuf import json_format
import yt_dlp

# Function to get YouTube stream URL using yt-dlp
def get_stream_url(video_url):
    #get key from env
    api_key = os.getenv('GOOGLE_API_KEY')
    ydl_opts = {
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.youtube.com/',
            'api_key': api_key,
        },
        'nocheckcertificate': True,  # In case certificates create issues
        'verbose': True,  # Optional: For more detailed output

        'format': 'best',  # You can change this to select the quality
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        stream_url = info['url']
    st.text_area(f"Stream URL for {video_url}:", value=stream_url)
    return stream_url



def draw_box_corners(frame, left, top, right, bottom, color, thickness=1, corner_length=15):
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
            #draw_box_corners(_frame, left, top, right, bottom, color)

            for concept in region.data.concepts:
                name = concept.name
                value = round(concept.value, 4)
                if value < det_threshold:
                    continue
                # Place text between top corners
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
    #frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
    #model_url = model_option['URL']
    #detector_model = Model(url=model_url)

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



st.title("Video Processing & Monitoring")

# Collapsible JSON results display
json_responses = []

# Section for playing and processing video frames
st.subheader("Video Frame Processing")
video_option = st.radio("Choose Video Input:", ("Standard Video File URLs","Webcam", "Youtube Streaming"), horizontal=True) #, "Webcam", "Streaming Video URLs"

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
elif video_option == "Youtube Streaming":
    # Input for multiple YouTube URLs with prepopulated example URLs
    youtube_urls = st.text_area("Enter YouTube URLs (one per line):",
                                value="https://www.youtube.com/watch?v=GIUTYf0Fpic\nhttps://www.youtube.com/watch?v=RDchI1SLh4Q")

    # Slider for frame skip selection
    frame_skip = st.slider("Select how many frames to skip:", min_value=1, max_value=120, value=28)
    det_threshold = st.slider("Select detection threshold:", min_value=0.01, max_value=1.00, value=0.5)

    # Obtain models from list_models()
    available_models = list_models()

    # Create a model selector for each YouTube URL
    url_list = [url.strip() for url in youtube_urls.split('\n') if url.strip()]
    json_responses.append(f"URL List: {url_list}")
    model_options = []
    for idx, url in enumerate(url_list):
        model_names = [model["Name"] for model in available_models]
        selected_model_name = st.selectbox(f"Select a model for YouTube Video {idx + 1}: {url}", model_names, key=f"youtube_model_{idx}")
        selected_model = next(model for model in available_models if model["Name"] == selected_model_name)
        model_options.append(selected_model)
    json_responses.append(f"Model Options: {model_options}")
    # Event to stop processing
    stop_event = threading.Event()

    # Stop processing button
    if st.button("Stop Processing"):
        stop_event.set()

    # Process YouTube videos button
    if st.button("Process YouTube Videos") and not stop_event.is_set():
        frame_placeholder = st.empty()
        try:
            # Initialize buffers and threads
            video_buffers = [deque(maxlen=6) for _ in range(len(url_list))]
            threads = []

            # Function to process each YouTube video
            def process_youtube_video(youtube_url, index, model_option, stop_event):
                try:
                    stream_url = get_stream_url(youtube_url)  # Use yt-dlp to get stream URL
                    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=10000, varThreshold=40, detectShadows=False)
                    overlay = None
                    overlay_decay = 3  
                    overlay_counter = 0
                    prev_frame = None
                    video_capture = cv2.VideoCapture(stream_url)
                    if not video_capture.isOpened():
                        st.error(f"Error: Could not open YouTube video at {stream_url}.")
                        return
                    
                    frame_count = 0

                    while video_capture.isOpened() and not stop_event.is_set():
                        ret, frame = video_capture.read()
                        frame = cv2.resize(frame, (640, 280))

                        if prev_frame is None:
                            prev_frame = frame

                        if not ret:
                            json_responses.append(f"Error: Failed to grab a frame from YouTube video at {stream_url}.")
                            break  # Stop the loop when no more frames

                        # Only process frames based on the user-selected frame skip
                        # if frame_count % frame_skip == 0:
                        #     # Run inference on the frame with the selected model
                        #     overlay, overlay_counter, processed_frame, prediction_response = run_model_inference(
                        #         det_threshold, background_subtractor, overlay, overlay_counter, prev_frame, frame, model_option
                        #     )

                        #     if prediction_response:
                        #         # Append prediction results to JSON responses
                        #         json_responses.append(json_format.MessageToJson(prediction_response))

                        #     # Convert the frame from BGR to RGB for displaying
                        #     rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                        #     # Add the frame to the buffer
                        #     video_buffers[index].append(rgb_frame)
                        rgb_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
                        video_buffers[index].append(rgb_frame)
                        frame_count += 1
                        prev_frame = frame

                    video_capture.release()
                except Exception as e:
                    st.error(e)
                    json_responses.append(f"Error {e} processing YouTube video at {youtube_url}")

            # Start threads for each YouTube URL with their corresponding model option
            for index, (youtube_url, model_option) in enumerate(zip(url_list, model_options)):
                thread = threading.Thread(target=process_youtube_video, args=(youtube_url, index, model_option, stop_event))
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

                        grid_image = np.concatenate(
                            [np.concatenate(grid_frames[i:i + 4], axis=1) for i in range(0, len(grid_frames), 4)],
                            axis=0
                        )

                    frame_placeholder.image(grid_image, caption="Processed YouTube Video Frames")

                time.sleep(0.01)

            # Ensure all threads are finished
            for thread in threads:
                thread.join()

        except Exception as e:
            st.error(e)
            json_responses.append(f"Error {e} processing YouTube video")

        st.success(f"Json Response: {json_responses}")

    verify_json_responses()


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
    if st.button("Stop Processing"):
        stop_event.set()
    if st.button("Process Streams") and not stop_event.is_set():
        # use ffmpeg to stream the video
        video_buffers = [deque(maxlen=2) for _ in range(len(stream_list))]
        threads = []
        def process_video(video_url, index, model_option, stop_event):
            background_subtractor = cv2.createBackgroundSubtractorMOG2(history=10000, varThreshold=40, detectShadows=False)
            overlay = None
            overlay_decay = 3  
            overlay_counter = 0
            prev_frame = None
            try:##ffmpeg to read the stream
              command = ['ffmpeg', '-i', video_url, '-f', 'image2pipe', '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-']
              process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
              frame_count = 0
              while not stop_event.is_set():
                  raw_frame = process.stdout.read(640 * 480 * 3)
                  if len(raw_frame) == 0:
                      break
                  frame = np.frombuffer(raw_frame, np.uint8).reshape(480, 640, 3)
                  if prev_frame is None:
                      prev_frame = frame
                  if frame_count % frame_skip == 0:
                      overlay, overlay_counter, processed_frame, prediction_response = run_model_inference(det_threshold, background_subtractor, overlay, overlay_counter, prev_frame, frame, model_option)
                      if prediction_response:
                          json_responses.append(json_format.MessageToJson(prediction_response))
                      rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                      video_buffers[index].append(rgb_frame)
                  frame_count += 1
                  prev_frame = frame
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
    frame_skip = st.slider("Select how many frames to skip:", min_value=1, max_value=120, value=28)
    det_threshold = st.slider("Select detection threshold:", min_value=0.01, max_value=1.00, value=0.5)
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
        try:

          # Initialize a list to hold buffers and threads
          video_buffers = [deque(maxlen=6) for _ in range(len(url_list))]  # Buffer for the latest 2 frames
          threads = []

          # Function to process each video
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
            
              frame_count = 0  # Initialize frame count

              while video_capture.isOpened() and not stop_event.is_set():
                  ret, frame = video_capture.read()
                  frame = cv2.resize(frame, (640, 280))
                  #previous_response = None
                  if prev_frame is None:
                      prev_frame = frame

                  if not ret:
                      break  # Stop the loop when no more frames

                  # Only process frames based on the user-selected frame skip
                  if frame_count % frame_skip == 0:
                      # Run inference on the frame with the selected model
                      overlay, overlay_counter, processed_frame, prediction_response = run_model_inference(det_threshold,background_subtractor, overlay, overlay_counter, prev_frame, frame, model_option)
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
                  prev_frame = frame

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
       
        except Exception as e:
          st.error(e)
          json_responses.append(f"Error {e} processing video at {video_url}")

        st.success(json_responses)

    verify_json_responses()

  
