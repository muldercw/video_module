import streamlit as st
import cv2
import tempfile
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
video_url = st.text_input("Enter a video URL:")

if st.button("Process Video"):
    if video_url:
        # Download the video to a temporary file
        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_capture = cv2.VideoCapture(video_url)

        # Process every other frame
        frame_count = 0
        processed_frames = []

        while video_capture.isOpened():
            ret, frame = video_capture.read()

            if not ret:
                break  # Stop the loop when no more frames

            # Only process every second frame
            if frame_count % 2 == 0:
                # Add a text box to the frame
                frame = cv2.putText(frame, "Processed Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (255, 0, 0), 2, cv2.LINE_AA)

                # Convert the frame from BGR to RGB (for displaying in Streamlit)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frames.append(rgb_frame)

            frame_count += 1

        video_capture.release()

        # Display processed frames in Streamlit
        for processed_frame in processed_frames:
            st.image(processed_frame)

    else:
        st.warning("Please provide a valid video URL.")
