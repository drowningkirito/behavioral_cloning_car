# 🛣️ Behavioral Cloning Self-Driving Car

This project implements a **behavioral cloning model** to control a self-driving car in a simulator using deep learning techniques. The project uses data collected from the [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) and trains a convolutional neural network (CNN) to predict steering angles based on camera images.

---

## 🚗 Project Overview

The core idea is to mimic human driving behavior using images from the simulator's front-facing camera. A CNN is trained on these images along with the corresponding steering angles, allowing the model to autonomously drive the car in the simulator.

---

## 🧠 Model Architecture

- **Input**: RGB images from center/left/right camera
- **Preprocessing**:
  - Cropping top/bottom areas (e.g., sky)
  - Normalization of pixel values
- **CNN**:
  - Several convolutional layers with ReLU activations
  - Fully connected layers
- **Output**: Predicted steering angle

Architecture inspired by NVIDIA's end-to-end self-driving car paper.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy, OpenCV , Streamlit,Python-Socketio,Eventlet,GradCam
- Jupyter Notebook
- Udacity Self-Driving Car Simulator


## 🧪 How to Run the Project

Follow the steps below to run the **dashboard**, **Grad-CAM visualizer**, and the **Udacity Self-Driving Car Simulator** together.




