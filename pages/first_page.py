import streamlit as st
import cv2
import numpy as np
import time
from collections import deque
from clarifai.client.auth import create_stub
from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.client.user import User
from clarifai.modules.css import ClarifaiStreamlitCSS
from google.protobuf import json_format, timestamp_pb2

st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)

# Authentication and Clarifai setup
auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()

st.title("Clarifai Input and Video Frame Processing")

# Form to get user input for the number of inputs to display
with st.form(key="data-inputs"):
    mtotal = st.number_input(
        "Select number of inputs to view in a table:", min_value=5, max_value=100)
    submitted = st.form_submit_button('Submit')

if submitted:
    if mtotal is None or mtotal == 0:
        st.warning("Number of inputs must be provided.")
        st.stop()
    else:
        st.write(f"Number of inputs in table will be: {mtotal}")

    # Retrieve inputs from Clarifai app
    input_obj = User(user_id=userDataObject.user_id).app(app_id=userDataObject.app_id).inputs()
    all_inputs = input_obj.list_inputs()

    # Check if there are enough inputs to display
    if len(all_inputs) < mtotal:
        raise Exception(f"Number of inputs is less than {mtotal}. Please add more inputs or reduce the inputs to be displayed!")

    else:
        data = []
        # Collect input data along with metadata
        for inp in range(mtotal):
            data.append({
                "id": all_inputs[inp].id,
                "data_url": all_inputs[inp].data.image.url,
                "status": all_inputs[inp].status.description,
                "created_at": timestamp_pb2.Timestamp.ToDatetime(all_inputs[inp].created_at),
                "modified_at": timestamp_pb2.Timestamp.ToDatetime(all_inputs[inp].modified_at),
                "metadata": json_format.MessageToDict(all_inputs[inp].data.metadata),
            })

        # Display data as a table
        st.dataframe(data)

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
    # Input for multiple video URLs
    video_urls = st.text_area("Enter video URLs (one per line):")
    
    # Slider for frame skip selection
    frame_skip = st.slider("Select how many frames to skip:", min_value=1, max_value=10, value=2)

    if st.button("Process Videos"):
        if video_urls:
            # Split the input into a list of URLs
            url_list = [url.strip() for url in video_urls.split('\n') if url.strip()]

            # Create a placeholder for the grid
            frame_placeholder = st.empty()

            # Loop through the URLs and create a frame buffer for each video
            video_buffers = [deque(maxlen=2) for _ in range(len(url_list))]  # Buffer for the latest 2 frames

            # Start processing videos
            while True:
                grid_frames = []  # List to hold frames for the grid

                for index, video_url in enumerate(url_list):
                    # Open the video stream directly from the URL using OpenCV
                    video_capture = cv2.VideoCapture(video_url)

                    if not video_capture.isOpened():
                        st.error(f"Error: Could not open video at {video_url}.")
                        continue

                    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
                    frame_count = 0  # Initialize frame count

                    while video_capture.isOpened():
                        ret, frame = video_capture.read()

                        if not ret:
                            break  # Stop the loop when no more frames

                        # Only process frames based on the user-selected frame skip
                        if frame_count % frame_skip == 0:
                            # Add a text box to the frame
                            frame = cv2.putText(frame, "Processed Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                                1, (255, 0, 0), 2, cv2.LINE_AA)

                            # Convert the frame from BGR to RGB (for displaying in Streamlit)
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            # Add the frame to the buffer
                            video_buffers[index].append(rgb_frame)

                            # If the buffer is filled, add the last frame to the grid
                            if len(video_buffers[index]) > 0:
                                grid_frames.append(video_buffers[index][-1])  # Append the latest frame

                        frame_count += 1

                        # Sleep dynamically based on the frame rate for smooth playback
                        time.sleep(1 / frame_rate)

                    video_capture.release()

                # Create a grid image from the frames
                if grid_frames:
                    grid_image = np.concatenate([np.concatenate(grid_frames[i:i+2], axis=1) for i in range(0, len(grid_frames), 2)], axis=0)

                    # Display the grid image
                    frame_placeholder.image(grid_image, caption="Video Frames Grid")

                time.sleep(0.1)  # Slight delay to avoid overwhelming the stream

        else:
            st.warning("Please provide valid video URLs.")
