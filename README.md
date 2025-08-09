# ESP32 Human Activity Recognition with MobileNetV2-LSTM

## 📌 Overview
This project implements a **real-time human activity recognition system** using **IMU sensors** (Accelerometer + Gyroscope) and a **heart rate sensor (BPM)**. It is built with an **ESP8266** (sensor data transmitter), an **ESP32** (receiver and inference device), and **Firebase Realtime Database** for storing results.

The deep learning model is based on **MobileNetV2 + LSTM** using **transfer learning**. Raw time-series sensor data is converted into **spectrogram images**, enabling robust classification of the following activities:
- **Idle**
- **Walking**
- **Running**

## 🛠️ Features
- **End-to-end data pipeline**: preprocessing, training, evaluation, and deployment
- **MobileNetV2-LSTM** transfer learning architecture
- **Spectrogram-based feature extraction** (STFT) consistent between training and inference
- **Hybrid decision-making** combining heuristic rules and model predictions
- **Real-time inference** via ESP32 connected over serial
- **Automatic ESP32 port detection**
- **Heart rate tracking** (instantaneous BPM and average BPM)
- **Firebase integration** for live updates of activity and heart rate

## 📂 Project Structure
```
├── pre-trained.py                # Model training & evaluation pipeline (MobileNetV2-LSTM)
├── esp32b.py                     # Real-time inference and Firebase integration
├── WemosD1Mini_Pengirim.ino      # ESP8266 sensor data transmitter code
├── ESP32___Penerima.ino          # ESP32 sensor data receiver (firmware)
├── models_mobilenet/             # Saved model (.keras), scaler.pkl, label_encoder.pkl
└── dataset/                      # Processed dataset (train/validation/test CSV)
```

## 📊 Workflow
1. **Data Preparation & Preprocessing**
   - Load IMU + heart rate dataset (CSV)
   - Clean and normalize data
   - Segment into windows and convert each to spectrogram images
2. **Model Training (`pre-trained.py`)**
   - Transfer learning with MobileNetV2 backbone
   - Temporal modeling using stacked LSTMs
   - Save trained model, scaler, and label encoder
3. **Real-time Inference (`esp32b.py`)**
   - Detect and connect to ESP32
   - Continuously read sensor data
   - Convert windows into spectrogram sequences
   - Predict activity with MobileNetV2-LSTM
   - Push results to Firebase
4. **Hardware Integration**
   - **ESP8266** collects accelerometer, gyroscope, and BPM data
   - **ESP32** receives data, performs inference, and uploads to Firebase

## 📡 Expected Sensor Data Format
```
timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,bmp,beatAvg,fingerDetected,dataRate
```
- Minimum **10 columns**
- Units:
  - Accelerometer: m/s²
  - Gyroscope: °/s
  - Heart rate: BPM

## 🔧 Requirements
### Python
- Python 3.8+
- TensorFlow 2.x
- NumPy, Pandas, SciPy
- scikit-learn, joblib
- Matplotlib, Seaborn
- firebase-admin

Install dependencies:
```bash
pip install -r requirements.txt
```

### Hardware
- **ESP8266** (WeMos D1 Mini or similar) with accelerometer + gyroscope + heart rate sensor
- **ESP32** for receiving data and performing inference
- USB cable for programming and serial communication

## 🚀 Usage
### 1. Train the Model
```bash
python pre-trained.py
```
This will:
- Load and preprocess the dataset
- Train MobileNetV2-LSTM
- Evaluate on validation & test sets
- Save the model and preprocessing files in `models_mobilenet/`

### 2. Run Real-Time Inference
```bash
python esp32b.py
```
- Automatically detects ESP32 USB port
- Reads incoming sensor data
- Predicts current activity
- Uploads to Firebase

## 📈 Example Output
```
🎯 ACTIVITY DETECTED: WALKING
📊 Confidence: 92.5%
📈 Probability breakdown:
   🎯 walking : 0.925 ████████████████████
      idle    : 0.050 ███
      running : 0.025 █
❤️  Current BPM: 78.0
❤️  Average BPM: 77
⬆️  Firebase: Activity=walking, AvgBPM=77
```

## 📜 License
This project is licensed under the MIT License – feel free to modify and use for your own applications.
