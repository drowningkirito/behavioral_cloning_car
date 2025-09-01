import socketio
import eventlet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from flask import Flask
from PIL import Image
import base64
from io import BytesIO

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import streamlit as st
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import os

# ==========================
# Shared Variables
# ==========================
shared_data = {}
data_lock = threading.Lock()
log_file = "logs.csv"
latency_log_file = "gradcam_times.csv"

# ==========================
# PyTorch Model Definition
# ==========================
class NvidiaTorchModel(nn.Module):
    def __init__(self):
        super(NvidiaTorchModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.ELU(),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 100), nn.ELU(), nn.Dropout(0.5),
            nn.Linear(100, 50), nn.ELU(), nn.Dropout(0.5),
            nn.Linear(50, 10), nn.ELU(), nn.Dropout(0.5),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.model(x)

# Load model
model = NvidiaTorchModel()
model.load_state_dict(torch.load("model_torch1.pth", map_location=torch.device("cpu")))
model.eval()

# ==========================
# Helper Functions
# ==========================
def preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

def image_to_tensor(image):
    return torch.tensor(image).float().permute(0, 3, 1, 2)

def generate_gradcam(model, input_tensor, target_layer):
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    return grayscale_cam

def measure_gradcam_time(model, input_tensor, target_layer):
    start_time = time.time()
    grayscale_cam = generate_gradcam(model, input_tensor, target_layer)
    end_time = time.time()
    duration = end_time - start_time

    with open(latency_log_file, "a") as f:
        f.write(f"{time.time()},{duration:.6f}\n")

    return grayscale_cam, duration

def encode_image(np_img):
    img = (np_img * 255).astype(np.uint8)
    success, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8') if success else ""

def encode_cam(np_img):
    success, buffer = cv2.imencode('.jpg', np_img)
    return base64.b64encode(buffer).decode('utf-8') if success else ""

def decode_img(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str)))

# ==========================
# SocketIO Server Thread
# ==========================
def start_socketio_server():
    sio = socketio.Server(cors_allowed_origins='*')
    flask_app = Flask(__name__)

    @sio.on("telemetry")
    def telemetry(sid, data):
        speed = float(data['speed'])
        gt_angle = float(data['steering_angle'])

        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image_np = np.asarray(image)
        image_pre = preprocess(image_np)
        input_tensor = image_to_tensor(np.array([image_pre]))

        with torch.no_grad():
            pred_angle = model(input_tensor).item()

        # GradCAM with timing
        grayscale_cam, gradcam_time = measure_gradcam_time(model, input_tensor, model.model[9])
        img_np = np.transpose(input_tensor.squeeze().numpy(), (1, 2, 0))
        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        throttle = 1.0 - speed / 10

        packet = {
            'pred_angle': round(pred_angle, 2),
            'gt_angle': round(gt_angle, 2),
            'speed': round(speed, 2),
            'throttle': round(throttle, 2),
            'image': encode_image(img_np),
            'gradcam': encode_cam(cam_image),
            'gradcam_latency': gradcam_time
        }

        with data_lock:
            shared_data.clear()
            shared_data.update(packet)

        sio.emit("dashboard_data", packet)
        sio.emit("steer", {"steering_angle": str(pred_angle), "throttle": str(throttle)})

    @sio.on("connect")
    def connect(sid, environ):
        print("Simulator connected")
        sio.emit("steer", {"steering_angle": "0", "throttle": "0"})

    eventlet.wsgi.server(eventlet.listen(('', 4567)), socketio.WSGIApp(sio, flask_app))

# Start server in background
threading.Thread(target=start_socketio_server, daemon=True).start()

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="Real-Time Driving Dashboard", layout="wide")
st.title("Real-Time Driving Dashboard 🚗")

img_col, cam_col = st.columns(2)
img_placeholder = img_col.empty()
cam_placeholder = cam_col.empty()

latency_placeholder = st.empty()
status_placeholder = st.empty()  # for "waiting" message

a_col, s_col, t_col = st.columns(3)
angle_placeholder = a_col.metric("Predicted Steering", "---")
gt_angle_placeholder = s_col.metric("Ground Truth Steering", "---")
speed_placeholder = st.metric("Speed", "---")
throttle_placeholder = st.metric("Throttle", "---")

st.markdown("---")
graph_placeholder = st.empty()
download_placeholder = st.empty()

# Init logs
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write("timestamp,pred_steering,gt_steering,speed,throttle\n")

if not os.path.exists(latency_log_file):
    with open(latency_log_file, "w") as f:
        f.write("timestamp,gradcam_latency\n")

log_deque = deque(maxlen=50)
last_plot_time = 0
log_counter = 0

# Main loop
while True:
    time.sleep(0.05)
    with data_lock:
        d = shared_data.copy()

    try:
        if not d:
            status_placeholder.info("⏳ Waiting for telemetry data...")
            continue
        else:
            status_placeholder.empty()

        if d.get("image"):
            img_placeholder.image(decode_img(d["image"]), caption="Input Image", use_column_width=True)
        if d.get("gradcam"):
            cam_placeholder.image(decode_img(d["gradcam"]), caption="Grad-CAM", use_column_width=True)
        if d.get("pred_angle") is not None:
            angle_placeholder.metric("Predicted Steering", f"{d['pred_angle']}°")
        if d.get("gt_angle") is not None:
            gt_angle_placeholder.metric("Ground Truth Steering", f"{d['gt_angle']}°")
        if d.get("speed") is not None:
            speed_placeholder.metric("Speed", f"{d['speed']}")
        if d.get("throttle") is not None:
            throttle_placeholder.metric("Throttle", f"{d['throttle']}")
        if d.get("gradcam_latency") is not None:
            latency_placeholder.markdown(f"⏱ **Grad-CAM Latency:** {d['gradcam_latency']*1000:.2f} ms")

        # Logging
        log_counter += 1
        if log_counter >= 5:
            log_counter = 0
            ts = time.time()
            with open(log_file, "a") as f:
                f.write(f"{ts},{d['pred_angle']},{d['gt_angle']},{d['speed']},{d['throttle']}\n")
            log_deque.append((ts, d['pred_angle'], d['gt_angle'], d['speed'], d['throttle']))

        # Plotting
        if time.time() - last_plot_time > 2 and len(log_deque) >= 2:
            last_plot_time = time.time()
            df = pd.DataFrame(log_deque, columns=["timestamp", "pred_steering", "gt_steering", "speed", "throttle"])
            df["time"] = pd.to_datetime(df["timestamp"], unit='s')

            fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            ax[0].plot(df["time"], df["pred_steering"], color="blue", label="Predicted")
            ax[0].plot(df["time"], df["gt_steering"], color="orange", linestyle="--", label="Ground Truth")
            ax[0].set_title("Steering Angle")
            ax[0].legend()

            ax[1].plot(df["time"], df["speed"], color="green")
            ax[1].set_title("Speed")

            ax[2].plot(df["time"], df["throttle"], color="red")
            ax[2].set_title("Throttle")

            for a in ax:
                a.grid(True)
                a.set_ylabel("Value")
            ax[2].set_xlabel("Time")

            graph_placeholder.pyplot(fig)

            with open(log_file, "rb") as f:
                csv_data = f.read()
            download_placeholder.download_button(
                label="📥 Download Logs",
                data=csv_data,
                file_name="steering_logs.csv",
                mime="text/csv",
                key="download_logs"  # unique key
            )

    except Exception:
        continue

    except Exception:
        continue

