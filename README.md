# 🧠 NeuroGuard AI – Driver Drowsiness Detection System

NeuroGuard AI is a real-time driver drowsiness detection system built using Python, Deep Learning, and Computer Vision. It monitors eye closure and yawning through a webcam and generates alerts when signs of drowsiness are detected.

## ✨ Features

- 👁️ Real-time eye monitoring
- 🥱 Yawning detection
- 🚨 Audio alert for detected drowsiness
- 📊 Live monitoring dashboard
- 🎥 Webcam-based face tracking
- 🤖 Deep learning models using TensorFlow and Keras

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy

## 📁 Project Structure

```text
NeuroGuard-AI/
├── NeuroGuard_Professional_v2.py
├── eye_model.keras
├── mouth_model.keras
└── README.md
```

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/NeuroGuard-AI.git
cd NeuroGuard-AI
```

**2. Install the required libraries**

```bash
python -m pip install tensorflow opencv-python mediapipe numpy
```

**3. Run the application**

```bash
python NeuroGuard_Professional_v2.py
```

Make sure both trained model files (`eye_model.keras` and `mouth_model.keras`) are in the same folder as the Python script.

## 🎮 Controls

- `Q` or `Esc` – Quit the application
- `D` – Toggle debugging information
- `M` – Toggle camera mirror mode

## 🎯 Objective

The goal of NeuroGuard AI is to demonstrate how computer vision and deep learning can help identify visual signs of driver fatigue and support safer driving.

## ⚠️ Disclaimer

This project is an educational prototype and is not a certified vehicle safety system. It may produce false detections and should not be relied upon as the sole method of preventing accidents.

## 👨‍💻 Author

Created as an AI and Computer Vision project.

---

⭐ If you find this project useful, consider giving the repository a star!
