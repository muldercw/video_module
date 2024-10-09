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
    usermodels = []
    for model in all_models:
        model_url = f"https://clarifai.com/{userDataObject.user_id}/{userDataObject.app_id}/models/{model.id}"
        _umod = {"Name": model.id, "URL": model_url, "type": "User"}
        usermodels.append(_umod)
    return list_community_models() + usermodels

def list_community_models():
    predefined_model_urls = [{"Name": "General-Image-Detection", "URL": "https://clarifai.com/clarifai/main/models/general-image-detection", "type":"Community"},
                             {"Name": "Face Detection", "URL": "https://clarifai.com/clarifai/main/models/face-detection", "type":"Community"}]
    return predefined_model_urls

def numpy_array_to_bytes(np_array, image_format='.jpg'):
    # Encode the image into the desired format (JPEG, PNG, etc.)
    success, encoded_image = cv2.imencode(image_format, np_array)

    if success:
        # Convert the encoded image to a byte array
        byte_array = encoded_image.tobytes()
        return byte_array
    else:
        raise ValueError("Failed to encode the NumPy array as an image.")
    
def run_model_inference(frame, model_option):
    _frame = frame.copy()
    try:
        
      frame_bytes = numpy_array_to_bytes(_frame)


      #_model = Model(model_id=model_option['Name'])
      #_model_versions = list(_model.list_versions())

      #model_url = "https://clarifai.com/clarifai/main/models/face-detection"
      model_url = model_option['URL']
      detector_model = Model(url = model_url)
      #prediction_response = f"Frame type is: {type(_frame)} and model is: {model_option['Name']} and model URL is: {model_option['URL']}"
      #cv2.putText(_frame, prediction_response, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
      prediction_response = detector_model.predict_by_bytes(frame_bytes, input_type="image")
      length, width = _frame.shape[:2]
      
      regions = prediction_response.outputs[0].data.regions

      for region in regions:
          # Accessing and rounding the bounding box values
          top_row = round(region.region_info.bounding_box.top_row, 3)
          left_col = round(region.region_info.bounding_box.left_col, 3)
          bottom_row = round(region.region_info.bounding_box.bottom_row, 3)
          right_col = round(region.region_info.bounding_box.right_col, 3)

          for concept in region.data.concepts:
              # Accessing and rounding the concept value
              name = concept.name
              value = round(concept.value, 4)

              print(
                  (f"{name}: {value} BBox: {top_row}, {left_col}, {bottom_row}, {right_col}")
              )
              cv2.rectangle(_frame, (int(left_col * width), int(top_row * length)),
                            (int(right_col * width), int(bottom_row * length)), (0, 255, 0), 2)
              cv2.putText(_frame, f"{name}: {value}", (int(left_col * width), int(top_row * length - 10)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

              # cv2.rectangle(_frame, (int(left_col * frame.shape[1]), int(top_row * frame.shape[0])),
              #                     (int(right_col * frame.shape[1]), int(bottom_row * frame.shape[0])), (0, 255, 0), 2)
              # # Draw corners instead of full rectangle
              # corner_length = 10  # Length of the corner lines

              # # Top-left corner
              # cv2.line(_frame, (int(left_col * _frame.shape[1]), int(top_row * _frame.shape[0])),
              #       (int(left_col * _frame.shape[1]) + corner_length, int(top_row * _frame.shape[0])), (0, 255, 0), 2)
              # cv2.line(_frame, (int(left_col * _frame.shape[1]), int(top_row * _frame.shape[0])),
              #       (int(left_col * _frame.shape[1]), int(top_row * _frame.shape[0]) + corner_length), (0, 255, 0), 2)

              # # Top-right corner
              # cv2.line(_frame, (int(right_col * _frame.shape[1]), int(top_row * _frame.shape[0])),
              #       (int(right_col * _frame.shape[1]) - corner_length, int(top_row * _frame.shape[0])), (0, 255, 0), 2)
              # cv2.line(_frame, (int(right_col * _frame.shape[1]), int(top_row * _frame.shape[0])),
              #       (int(right_col * _frame.shape[1]), int(top_row * _frame.shape[0]) + corner_length), (0, 255, 0), 2)

              # # Bottom-left corner
              # cv2.line(_frame, (int(left_col * _frame.shape[1]), int(bottom_row * _frame.shape[0])),
              #       (int(left_col * _frame.shape[1]) + corner_length, int(bottom_row * _frame.shape[0])), (0, 255, 0), 2)
              # cv2.line(_frame, (int(left_col * _frame.shape[1]), int(bottom_row * _frame.shape[0])),
              #       (int(left_col * _frame.shape[1]), int(bottom_row * _frame.shape[0]) - corner_length), (0, 255, 0), 2)

              # # Bottom-right corner
              # cv2.line(_frame, (int(right_col * _frame.shape[1]), int(bottom_row * _frame.shape[0])),
              #       (int(right_col * _frame.shape[1]) - corner_length, int(bottom_row * _frame.shape[0])), (0, 255, 0), 2)
              # cv2.line(_frame, (int(right_col * _frame.shape[1]), int(bottom_row * _frame.shape[0])),
              #       (int(right_col * _frame.shape[1]), int(bottom_row * _frame.shape[0]) - corner_length), (0, 255, 0), 2)
      return _frame, prediction_response
    except Exception as e:
      cv2.putText(_frame, f"{str(e)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
      return _frame, None
    
st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)

# Authentication and Clarifai setup
auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()

st.title("Video Processing & Monitoring")

# Section for playing and processing video frames
st.subheader("Video Frame Processing")
video_option = st.radio("Choose Video Input:", ("Multiple Video URLs", "Webcam"), horizontal=True)

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
