import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ------------------ MODEL ------------------
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
        self.fc4 = nn.Linear(10, 1) 
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


# ------------------ STREAMLIT APP ------------------
st.title("Autonomous Car Simulation Dashboard")

# ----- LOSS PLOTS -----
st.header("1. Loss Curves")
try:
    loss_data = pd.read_csv("loss_history.csv")  # columns: epoch, train_loss, val_loss
    fig1, ax1 = plt.subplots()
    ax1.plot(loss_data["epoch"], loss_data["loss"], label="Train Loss")
    ax1.plot(loss_data["epoch"], loss_data["val_loss"], label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    st.pyplot(fig1)
except Exception as e:
    st.warning(f"Could not load loss_log.csv: {e}")

# ----- PRED VS ACTUAL -----
st.header("2. Predicted vs Actual Steering")
try:
    pred_data = pd.read_csv("pred_vs_actual.csv")  # columns: frame, predicted, actual
    fig2, ax2 = plt.subplots()
    ax2.plot(pred_data["frame"], pred_data["predicted"], label="Predicted")
    ax2.plot(pred_data["frame"], pred_data["actual"], label="Actual")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Steering Angle")
    ax2.legend()
    st.pyplot(fig2)
except Exception as e:
    st.warning(f"Could not load pred_vs_actual.csv: {e}")

# ----- REAL-TIME INFERENCE -----
st.header("3. Real-time Inference with Grad-CAM")
uploaded_file = st.file_uploader("Upload an image (road view)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Read and preprocess image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((66, 200)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image_rgb).unsqueeze(0)

    # Load model
    model = NvidiaTorchModel()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()

    # Predict steering angle
    with torch.no_grad():
        steering_angle = model(input_tensor).item()
    st.success(f"Predicted Steering Angle: {steering_angle:.2f}")

    # Grad-CAM setup
    target_layer = model.conv4
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]

    img_np = np.transpose(input_tensor.squeeze().numpy(), (1, 2, 0))
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    st.image(visualization, caption="Grad-CAM Heatmap", use_column_width=True)
