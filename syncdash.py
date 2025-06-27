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
# 1. Shared Variables
# ==========================
shared_data = {}
data_lock = threading.Lock()
log_file = "logs.csv"

# ==========================
# 2. Model Definition
# ==========================
# class NvidiaTorchModel(nn.Module):
#     def __init__(self):
#         super(NvidiaTorchModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
#         self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
#         self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
#         self.conv4 = nn.Conv2d(48, 64, kernel_size=5)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(64 * 1 * 18, 100)
#         self.fc2 = nn.Linear(100, 50)
#         self.fc3 = nn.Linear(50, 10)
#         self.fc4 = nn.Linear(10, 1)

#     def forward(self, x):
#         x = F.elu(self.conv1(x))
#         x = F.elu(self.conv2(x))
#         x = F.elu(self.conv3(x))
#         x = F.elu(self.conv4(x))
#         x = self.flatten(x)
#         x = F.elu(self.fc1(x))
#         x = F.elu(self.fc2(x))
#         x = F.elu(self.fc3(x))
#         x = self.fc4(x)
#         return x

class NvidiaTorchModel(nn.Module):
    def __init__(self):
        super(NvidiaTorchModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)  # (66, 200) -> (31, 98)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2) # -> (14, 47)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2) # -> (5, 22)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=5)            # -> (1, 18)

        # Compute flatten size dynamically
        self._to_linear = None
        self._get_conv_output()

        self.fc1 = nn.Linear(self._to_linear, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)

    def _get_conv_output(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 66, 200)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return x    



model = NvidiaTorchModel()
model.load_state_dict(torch.load("nvidia_model.pth", map_location=torch.device("cpu")))
model.eval()

# ==========================
# 3. Helpers
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
# 4. SocketIO Server
# ==========================
def start_socketio_server():
    sio = socketio.Server(cors_allowed_origins='*')
    flask_app = Flask(__name__)

    @sio.on("telemetry")
    def telemetry(sid, data):
        speed = float(data['speed'])
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image_np = np.asarray(image)
        image_pre = preprocess(image_np)
        input_tensor = image_to_tensor(np.array([image_pre]))

        with torch.no_grad():
            angle = model(input_tensor).item()

        grayscale_cam = generate_gradcam(model, input_tensor, model.conv4)
        img_np = np.transpose(input_tensor.squeeze().numpy(), (1, 2, 0))
        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        throttle = 1.0 - speed / 10

        packet = {
            'angle': round(angle, 2),
            'speed': round(speed, 2),
            'throttle': round(throttle, 2),
            'image': encode_image(img_np),
            'gradcam': encode_cam(cam_image)
        }

        with data_lock:
            shared_data.clear()
            shared_data.update(packet)

        sio.emit("dashboard_data", packet)
        sio.emit("steer", {"steering_angle": str(angle), "throttle": str(throttle)})

    @sio.on("connect")
    def connect(sid, environ):
        print("Simulator connected")
        sio.emit("steer", {"steering_angle": "0", "throttle": "0"})

    eventlet.wsgi.server(eventlet.listen(('', 4567)), socketio.WSGIApp(sio, flask_app))

# Start server in background
threading.Thread(target=start_socketio_server, daemon=True).start()

# ==========================
# 5. Streamlit UI
# ==========================
st.set_page_config(page_title="Real-Time Driving Dashboard", layout="wide")
st.title("Real-Time Driving Dashboard 🚗")

img_col, cam_col = st.columns(2)
img_placeholder = img_col.empty()
cam_placeholder = cam_col.empty()

a_col, s_col, t_col = st.columns(3)
angle_placeholder = a_col.metric("Steering Angle", "---")
speed_placeholder = s_col.metric("Speed", "---")
throttle_placeholder = t_col.metric("Throttle", "---")

st.markdown("---")
graph_placeholder = st.empty()
download_placeholder = st.empty()

if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write("timestamp,steering,speed,throttle\n")

log_deque = deque(maxlen=50)
last_plot_time = 0
log_counter = 0

while True:
    time.sleep(0.05)
    with data_lock:
        d = shared_data.copy()

    try:
        if d.get("image"):
            img_placeholder.image(decode_img(d["image"]), caption="Input Image", use_column_width=True)
        if d.get("gradcam"):
            cam_placeholder.image(decode_img(d["gradcam"]), caption="Grad-CAM", use_column_width=True)
        if d.get("angle") is not None:
            angle_placeholder.metric("Steering Angle", f"{d['angle']}°")
        if d.get("speed") is not None:
            speed_placeholder.metric("Speed", f"{d['speed']}")
        if d.get("throttle") is not None:
            throttle_placeholder.metric("Throttle", f"{d['throttle']}")

        # Logging every 5 frames
        log_counter += 1
        if log_counter >= 5:
            log_counter = 0
            ts = time.time()
            with open(log_file, "a") as f:
                f.write(f"{ts},{d['angle']},{d['speed']},{d['throttle']}\n")
            log_deque.append((ts, d['angle'], d['speed'], d['throttle']))

        # Plot every 2 seconds
        if time.time() - last_plot_time > 2 and len(log_deque) >= 2:
            last_plot_time = time.time()
            df = pd.DataFrame(log_deque, columns=["timestamp", "steering", "speed", "throttle"])
            df["time"] = pd.to_datetime(df["timestamp"], unit='s')

            fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            ax[0].plot(df["time"], df["steering"], color="blue")
            ax[0].set_title("Steering Angle")

            ax[1].plot(df["time"], df["speed"], color="green")
            ax[1].set_title("Speed")

            ax[2].plot(df["time"], df["throttle"], color="red")
            ax[2].set_title("Throttle")

            for a in ax:
                a.grid(True)
                a.set_ylabel("Value")
            ax[2].set_xlabel("Time")

            graph_placeholder.pyplot(fig)

            # Download button
            with open(log_file, "rb") as f:
                csv_data = f.read()
            download_placeholder.download_button(
                label="📥 Download Logs",
                data=csv_data,
                file_name="steering_logs.csv",
                mime="text/csv"
            )

    except:
        continue

