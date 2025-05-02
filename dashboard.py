import streamlit as st
from PIL import Image as PILImage
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Steering Dashboard", layout="wide")

st.title("Autonomous Driving Dashboard")
st.markdown("Live updates with real-time steering prediction, throttle control, and Grad-CAM heatmaps.")

# Create placeholders
image_col, cam_col = st.columns(2)
image_placeholder = image_col.empty()
cam_placeholder = cam_col.empty()

status_col1, status_col2, status_col3 = st.columns(3)
angle_placeholder = status_col1.metric("Steering Angle", "---")
speed_placeholder = status_col2.metric("Speed", "---")
throttle_placeholder = status_col3.metric("Throttle", "---")

st.markdown("---")
graph_placeholder = st.empty()
download_placeholder = st.empty()

# Init log
log_file = "logs.csv"
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write("timestamp,steering,speed,throttle\n")

# Main loop
while True:
    try:
        if all(os.path.exists(f) for f in ["latest_image.jpg", "latest_cam.jpg", "latest_angle.txt", "latest_speed.txt", "latest_throttle.txt"]):
            # Safe image read
            try:
                image = PILImage.open("latest_image.jpg").copy()
                cam = PILImage.open("latest_cam.jpg").copy()
            except:
                # time.sleep(0.00001)
                continue

            # Read values
            with open("latest_angle.txt", "r") as f:
                angle = float(f.read().strip())

            with open("latest_speed.txt", "r") as f:
                speed = float(f.read().strip())

            with open("latest_throttle.txt", "r") as f:
                throttle = float(f.read().strip())

            # Display images and metrics
            image_placeholder.image(image, caption="Input Image", use_column_width=True)
            cam_placeholder.image(cam, caption="Grad-CAM Heatmap", use_column_width=True)
            angle_placeholder.metric("Steering Angle", f"{angle:.2f}°")
            speed_placeholder.metric("Speed", f"{speed:.2f}")
            throttle_placeholder.metric("Throttle", f"{throttle:.2f}")

            # Append to logs
            with open(log_file, "a") as f:
                f.write(f"{time.time()},{angle},{speed},{throttle}\n")

            # Load logs
            df = pd.read_csv(log_file, names=["timestamp", "steering", "speed", "throttle"], skiprows=1)
            df["time"] = pd.to_datetime(df["timestamp"], unit='s')
            last_df = df.tail(50)

            # Plotting
            fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            ax[0].plot(last_df["time"], last_df["steering"], color="blue")
            ax[0].set_title("Steering Angle")

            ax[1].plot(last_df["time"], last_df["speed"], color="green")
            ax[1].set_title("Speed")

            ax[2].plot(last_df["time"], last_df["throttle"], color="red")
            ax[2].set_title("Throttle")

            for a in ax:
                a.grid(True)
                a.set_ylabel("Value")
            ax[2].set_xlabel("Time")

            graph_placeholder.pyplot(fig)

            # Download button
            download_placeholder.download_button(
                label="Download Logs",
                data=open(log_file, "rb").read(),
                file_name="steering_logs.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.warning(f"Waiting for data or delayed update: {e}")

    # time.sleep(1)
