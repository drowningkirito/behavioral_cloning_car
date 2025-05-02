import socketio
import eventlet
import numpy as np
import torch
from flask import Flask
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image

import streamlit as st
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import base64
from io import BytesIO

# Define NvidiaTorchModel architecture
class NvidiaTorchModel(nn.Module):
    def __init__(self):
        super(NvidiaTorchModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 1 * 18, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)  # Output layer for steering angle

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.flatten(x)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return x

# Load the model
model = NvidiaTorchModel()
model.load_state_dict(torch.load('model.pth'))  # Load the saved weights
model.eval()  # Set the model to evaluation mode

# Preprocess image
def img_preprocess(img):
    img = img[60:135, :, :]  # Crop image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert to YUV
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian blur
    img = cv2.resize(img, (200, 66))  # Resize image
    img = img / 255.0  # Normalize image
    return img

# Convert image to tensor
def image_to_tensor(image):
    image_tensor = torch.tensor(image).float().permute(0, 3, 1, 2)  # Convert to tensor and reorder dimensions
    return image_tensor

# Grad-CAM Setup
def generate_gradcam(model, input_tensor, target_layer):
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    return grayscale_cam

# SocketIO and Flask Setup
sio = socketio.Server()
app = Flask(__name__)

# Telemetry function to process incoming image data and make predictions
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    input_tensor = image_to_tensor(np.array([image]))  # Convert to tensor
    

    # # Predict the steering angle using the model
    # with torch.no_grad():
    #     steering_angle = model(input_tensor).item()

    # st.success(f"Predicted Steering Angle: {steering_angle:.2f}")

    # # Generate Grad-CAM for visualization
    # target_layer = model.conv4  # Last convolutional layer
    # grayscale_cam = generate_gradcam(model, input_tensor, target_layer)

    # img_np = np.transpose(input_tensor.squeeze().numpy(), (1, 2, 0))  # Convert tensor to numpy
    # visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    # # Show the visualization in Streamlit
    # st.image(visualization, caption="Grad-CAM Heatmap", use_column_width=True)
    with torch.no_grad():
        steering_angle = model(input_tensor).item()

    target_layer = model.conv4  # Last convolutional layer
    grayscale_cam = generate_gradcam(model, input_tensor, target_layer)

    img_np = np.transpose(input_tensor.squeeze().numpy(), (1, 2, 0))  # Convert tensor to numpy
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    
    Image.fromarray((img_np * 255).astype(np.uint8)).save("latest_image.jpg")
    plt.imsave("latest_cam.jpg", visualization)
    with open("latest_angle.txt", "w") as f:
        f.write(f"{steering_angle:.2f}")
    with open("latest_speed.txt", "w") as f:
        f.write(f"{speed:.2f}")
        

    # Compute throttle
    throttle = 1.0 - speed / 10  # Adjust according to the speed limit
    with open("latest_throttle.txt", "w") as f:
        f.write(f"{throttle:.2f}")
    print(f"Steering Angle: {steering_angle}, Throttle: {throttle}, Speed: {speed}")

    # Send control to the car (steering and throttle)
    send_control(steering_angle, throttle)

# Send control to car
def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

# Connect to the car
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

# Run Flask and SocketIO server
if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


