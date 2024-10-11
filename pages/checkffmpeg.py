import streamlit as st
import subprocess

def check_ffmpeg_installed():
    try:
        # Run `ffmpeg -version` to check if FFmpeg is installed
        result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            # FFmpeg is installed, return the version details
            return result.stdout.split('\n')[0]  # Get only the first line with the version
        else:
            return "FFmpeg is not installed."
    except FileNotFoundError:
        return "FFmpeg is not installed."

# Streamlit app
st.title("FFmpeg Version Checker")

ffmpeg_version = check_ffmpeg_installed()

# Display result on Streamlit page
st.write(f"FFmpeg status: {ffmpeg_version}")
