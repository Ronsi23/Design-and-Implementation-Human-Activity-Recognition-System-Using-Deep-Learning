"""
ESP32 Python Activity Recognition - Dataset Compatible
======================================================
Compatible dengan dataset: Arm_Acc_X, Arm_Acc_Y, Arm_Acc_Z, Arm_Gyro_X, Arm_Gyro_Y, Arm_Gyro_Z
Activities: diam, jalan, lari
ESP32 sebagai receiver untuk data sensor
ESP8266 sebagai transmitter data sensor (accelerometer + gyroscope)
AI untuk klasifikasi aktivitas
Python untuk processing dan inference
"""

import serial
import serial.tools.list_ports
import numpy as np
import tensorflow as tf
import joblib
import time
import os
import pandas as pd
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime
from scipy import signal

class ESP32ActivityRecognition:
    def __init__(self, model_folder="models_mobilenet"):
        """Initialize ESP32 Activity Recognition System"""
        self.model_folder = model_folder
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.ser = None
        self.firebase_initialized = False
        self.db_ref = None

        # Parameters (EXACT match dengan training dataset)
        self.sensor_columns = ['Arm_Acc_X', 'Arm_Acc_Y', 'Arm_Acc_Z', 
                            'Arm_Gyro_X', 'Arm_Gyro_Y', 'Arm_Gyro_Z']
        self.window_size = 128
        self.sequence_length = 16
        self.step_size = 4  # 128 // 32
        self.image_size = 32
        self.buffer = []
        
        # Statistics
        self.data_count = 0
        self.prediction_count = 0
        self.invalid_count = 0
        self.start_time = time.time()
        self.last_timestamp = 0
        
        # PERBAIKAN: Heart rate tracking variables - pastikan ada di sini
        self.current_bmp = 72.0  # Instantaneous BPM
        self.current_beat_avg = 72  # Average BPM
        
        # Dataset statistics for validation (from training analysis)
        self.dataset_stats = {
            'diam': {
                'acc_magnitude': 9.75,
                'gyro_magnitude': 1.11,
                'acc_range': [-10.44, 2.55],
                'gyro_range': [-1.11, 0.94]
            },
            'jalan': {
                'acc_magnitude': 10.27,
                'gyro_magnitude': 1.12,
                'acc_range': [-18.60, 6.29],
                'gyro_range': [-1.13, 1.12]
            },
            'lari': {
                'acc_magnitude': 18.66,
                'gyro_magnitude': 1.12,
                'acc_range': [-22.02, 24.64],
                'gyro_range': [-1.14, 1.19]
            }
        }
        
        print("🎯 ESP32 Activity Recognition System - 10 Column Compatible")
        print("=" * 60)
        print(f"📊 Compatible with dataset columns: {self.sensor_columns}")
        print(f"🎯 Target activities: diam, jalan, lari")
        print(f"📡 Format: timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,bmp,beatAvg,fingerDetected,dataRate")
        print(f"❤️  Heart rate tracking: enabled")
    
    def load_model_components(self):
        """Load trained model dan preprocessing components"""
        print("📦 Loading model components...")
        
        try:
            # Load model
            model_path = os.path.join(self.model_folder, "mobilenetv2_lstm_model.keras")
            self.model = tf.keras.models.load_model(model_path)
            print(f"   ✅ Model: {model_path}")
            
            # Load scaler
            scaler_path = os.path.join(self.model_folder, "scaler.pkl")
            self.scaler = joblib.load(scaler_path)
            print(f"   ✅ Scaler: {scaler_path}")
            
            # Load label encoder
            encoder_path = os.path.join(self.model_folder, "label_encoder.pkl")
            self.label_encoder = joblib.load(encoder_path)
            print(f"   ✅ Encoder: {encoder_path}")
            
            print(f"   🎯 Classes: {list(self.label_encoder.classes_)}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
def find_esp32_port(self):
    """Auto-detect ESP32 port dengan debug mode"""
    print("🔍 Scanning for ESP32...")

    ports = serial.tools.list_ports.comports()
    if not ports:
        print("   ❌ No serial ports found")
        return None

    print(f"   Found {len(ports)} port(s):")

    for port in ports:
        print(f"   📍 Testing {port.device}")
        print(f"      Description: {port.description}")
        
        # Check if it's likely ESP32
        keywords = ['usb', 'serial', 'uart', 'ch340', 'cp210', 'ftdi', 'silicon labs']
        if not any(keyword in port.description.lower() for keyword in keywords):
            print(f"      ⏭️  Skipping (not USB device)")
            continue
        
        # Test data format
        try:
            test_ser = serial.Serial(port.device, 115200, timeout=3)
            time.sleep(2)  # Give ESP32 time to initialize
            
            print(f"      🔍 DEBUG: Reading data for 5 seconds...")
            
            all_lines = []
            valid_count = 0
            
            for i in range(50):  # Test for 5 seconds
                if test_ser.in_waiting > 0:
                    try:
                        line = test_ser.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            all_lines.append(line)
                            print(f"      📥 {i:2d}: {line}")
                            
                            # Skip status messages and headers
                            if not line or line.startswith(('✅', '❌', '📈', '📊', 'timestamp', '#', 'Sent:')):
                                print(f"         ⏭️  Skipped (status message)")
                                continue
                                
                            parts = line.split(',')
                            print(f"         🔢 Parts: {len(parts)} columns")
                            
                            # Accept 10+ kolom seperti esp32.py
                            if len(parts) >= 10:
                                try:
                                    timestamp = int(parts[0])
                                    # Ambil 6 sensor data pertama
                                    acc_x, acc_y, acc_z = float(parts[1]), float(parts[2]), float(parts[3])
                                    gyro_x, gyro_y, gyro_z = float(parts[4]), float(parts[5]), float(parts[6])
                                    # Heart rate data (optional)
                                    bmp = float(parts[7]) if parts[7] != '0.0' else 0.0
                                    beat_avg = int(parts[8]) if parts[8] != '0' else 0
                                    
                                    print(f"         📊 Parsed: ts={timestamp}, acc=({acc_x:.2f},{acc_y:.2f},{acc_z:.2f}), gyro=({gyro_x:.2f},{gyro_y:.2f},{gyro_z:.2f}), bmp={bmp:.1f}")
                                    
                                    # Check if values are in reasonable range
                                    acc_ok = all(-50 < x < 50 for x in [acc_x, acc_y, acc_z])
                                    gyro_ok = all(-20 < x < 20 for x in [gyro_x, gyro_y, gyro_z])
                                    
                                    print(f"         ✓ Range check: acc_ok={acc_ok}, gyro_ok={gyro_ok}")
                                    
                                    if acc_ok and gyro_ok:
                                        valid_count += 1
                                        print(f"         ✅ VALID sample #{valid_count}")
                                        if valid_count == 1:
                                            print(f"      📥 Sample: acc({acc_x:.2f},{acc_y:.2f},{acc_z:.2f}) "
                                                f"gyro({gyro_x:.2f},{gyro_y:.2f},{gyro_z:.2f}) BPM:{bmp:.0f} Avg:{beat_avg}")
                                    else:
                                        print(f"         ❌ Range check failed")
                                            
                                except (ValueError, IndexError) as e:
                                    print(f"         ❌ Parse error: {e}")
                            else:
                                print(f"         ❌ Insufficient columns: {len(parts)}")
                    except UnicodeDecodeError as e:
                        print(f"         ❌ Decode error: {e}")
                else:
                    print(f"      ⏳ {i:2d}: No data...")
                time.sleep(0.1)
            
            test_ser.close()
            
            print(f"      📊 Summary:")
            print(f"         Total lines received: {len(all_lines)}")
            print(f"         Valid samples: {valid_count}")
            
            if len(all_lines) == 0:
                print(f"      ❌ No data received at all!")
                print(f"         Check: ESP32 program running? Baud rate 115200?")
            elif valid_count == 0:
                print(f"      ❌ Data received but format invalid")
                print(f"         Sample lines:")
                for line in all_lines[:3]:
                    print(f"           {line}")
            
            if valid_count >= 3:
                print(f"      ✅ ESP32 detected! ({valid_count} valid samples)")
                return port.device
            else:
                print(f"      ❌ Invalid data format ({valid_count} valid samples)")
                
        except Exception as e:
            print(f"      ❌ Test failed: {e}")
    
    # PERBAIKAN: return None harus di level fungsi, bukan di dalam loop
    return None
    
def initialize_firebase(self, service_account_key_path, database_url):
        """Inisialisasi Firebase Admin SDK"""
        print("\n🔥 Initializing Firebase...")
        
        # Check if file exists
        if not os.path.exists(service_account_key_path):
            print(f"    ❌ Service account key file not found: {service_account_key_path}")
            return False
            
        try:
            cred = credentials.Certificate(service_account_key_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': database_url
            })
            # Reference langsung ke root database
            self.db_ref = db.reference('/')
            self.firebase_initialized = True
            print("    ✅ Firebase initialized successfully!")
            print(f"    Database URL: {database_url}")
            print(f"    Writing format: Activity + AvgBPM")
            return True
        except Exception as e:
            print(f"    ❌ Failed to initialize Firebase: {e}")
            self.firebase_initialized = False
            return False
    
def send_to_firebase(self, activity_label, avg_bpm_value):
        """Kirim data sederhana ke Firebase - Activity dan AvgBPM"""
        if not self.firebase_initialized or self.db_ref is None:
            print("    ⚠️  Firebase not initialized. Skipping data upload.")
            return
        
        try:
            # Data sederhana - Activity dan AvgBPM (beat average)
            simple_data = {
                'Activity': activity_label,  # diam, jalan, atau lari
                'Bpm': int(avg_bpm_value) if avg_bpm_value > 0 else 72  # Default ke 72 jika tidak ada data
            }
            
            # Update langsung ke root database
            self.db_ref.update(simple_data)
            
            print(f"    ⬆️  Firebase: Activity={activity_label}, AvgBPM={simple_data['Bpm']}")
            
        except Exception as e:
            print(f"    ❌ Error sending data to Firebase: {e}")

def extract_heart_rate_data(self, sensor_line):
        """Extract heart rate data from sensor line"""
        try:
            parts = sensor_line.split(',')
            if len(parts) >= 10:
                # Format: timestamp,accX,accY,accZ,gyroX,gyroY,gyroZ,bmp,beatAvg,fingerDetected,dataRate
                instantaneous_bmp = float(parts[7]) if parts[7] != '0.0' else 0.0
                beat_avg = int(parts[8]) if parts[8] != '0' else 0
                
                # Update values
                if beat_avg > 0 and beat_avg < 200:  # Validate beat average
                    self.current_beat_avg = beat_avg
                elif instantaneous_bmp > 20 and instantaneous_bmp < 200:
                    # Fallback to instantaneous if beat_avg not available
                    self.current_beat_avg = int(instantaneous_bmp)
                
                # Update instantaneous BPM
                if instantaneous_bmp > 20 and instantaneous_bmp < 200:
                    self.current_bpm = instantaneous_bmp
                
        except (ValueError, IndexError):
            pass

def connect_esp32(self):
        """Connect to ESP32"""
        esp32_port = self.find_esp32_port()
        
        if not esp32_port:
            print("\n❌ ESP32 not found!")
            print("\n🔧 Troubleshooting:")
            print("   1. ESP32 connected via USB")
            print("   2. ESP32 receiver program running") 
            print("   3. ESP8266 transmitting sensor data")
            print("   4. Data format: timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z")
            return False
        
        try:
            print(f"📡 Connecting to ESP32 on {esp32_port}...")
            self.ser = serial.Serial(esp32_port, 115200, timeout=2)
            time.sleep(2)
            print(f"   ✅ Connected!")
            return True
            
        except Exception as e:
            print(f"   ❌ Connection failed: {e}")
            return False
    
def validate_sensor_data(self, timestamp, sensor_data):
        """Validate sensor data terhadap dataset characteristics"""
        # Check timestamp progression
        if hasattr(self, 'last_timestamp') and timestamp <= self.last_timestamp:
            return False, "Timestamp regression"
        
        # Extract accelerometer dan gyroscope data
        acc_x, acc_y, acc_z = sensor_data[:3]
        gyro_x, gyro_y, gyro_z = sensor_data[3:]
        
        # Validate against expanded ranges (same as esp32.py)
        if not all(-50 < x < 50 for x in [acc_x, acc_y, acc_z]):
            return False, "Accelerometer out of range"
        
        if not all(-20 < x < 20 for x in [gyro_x, gyro_y, gyro_z]):
            return False, "Gyroscope out of range"
        
        self.last_timestamp = timestamp
        return True, "Valid"
    
def convert_to_spectrogram(self, window_data):
        """
        Convert time series data ke spectrogram (SAME AS TRAINING)
        Compatible dengan dataset format
        """
        try:
            spectrograms = []
            
            # Process each sensor channel (6 channels like dataset)
            for i in range(6):
                data = window_data[:, i]
                
                # Add noise if constant
                if np.std(data) < 1e-8:
                    data = data + np.random.normal(0, 1e-6, len(data))
                
                # STFT
                frequencies, times, Zxx = signal.stft(
                    data, 
                    fs=50,  # Sampling frequency
                    window='hann',
                    nperseg=min(32, len(data)//2),
                    noverlap=min(16, len(data)//4)
                )
                
                # Convert to magnitude spectrogram
                magnitude = np.abs(Zxx)
                
                # Normalize
                if magnitude.max() > 0:
                    magnitude = magnitude / magnitude.max()
                
                spectrograms.append(magnitude)
            
            # Combine spectrograms into RGB-like image
            # Use first 3 sensors (accelerometer) for RGB channels
            rgb_image = np.zeros((self.image_size, self.image_size, 3))
            
            for i in range(3):
                if len(spectrograms) > i and spectrograms[i].size > 0:
                    # Resize spectrogram to image_size x image_size
                    resized = tf.image.resize(
                        spectrograms[i][..., np.newaxis], 
                        [self.image_size, self.image_size]
                    ).numpy().squeeze()
                    rgb_image[:, :, i] = resized
                else:
                    rgb_image[:, :, i] = np.random.uniform(0, 0.1, (self.image_size, self.image_size))
            
            return rgb_image
            
        except Exception as e:
            print(f"   ⚠️  Spectrogram conversion error: {e}")
            return np.random.uniform(0, 0.1, (self.image_size, self.image_size, 3))
    
def preprocess_window(self, window):
        """
        Preprocess window untuk model inference
        Compatible dengan training preprocessing
        """
        try:
            sequences = []
            
            # Create sequences exactly like training
            for i in range(0, len(window) - self.step_size + 1, self.step_size):
                sub_window = window[i:i+self.step_size]
                sub_window = np.array(sub_window)
                
                # Convert to spectrogram image
                image_2d = self.convert_to_spectrogram(sub_window)
                sequences.append(image_2d)
            
            # Pad or truncate sequence to fixed length
            if len(sequences) > self.sequence_length:
                sequences = sequences[:self.sequence_length]
            elif len(sequences) < self.sequence_length:
                while len(sequences) < self.sequence_length:
                    if len(sequences) > 0:
                        sequences.append(sequences[-1])
                    else:
                        sequences.append(np.random.uniform(0, 0.1, (self.image_size, self.image_size, 3)))
            
            # Convert to model input format
            X = np.array(sequences).reshape(1, self.sequence_length, self.image_size, self.image_size, 3)
            
            # Apply scaler (same as training)
            X_flat = X.reshape(-1, self.image_size * self.image_size * 3)
            X_scaled = self.scaler.transform(X_flat)
            X_scaled = X_scaled.reshape(X.shape)
            
            return X_scaled
            
        except Exception as e:
            print(f"   ❌ Preprocessing error: {e}")
            return None
    
def classify_activity_heuristic(self, window_data):
        """
        Heuristic classification berdasarkan dataset statistics
        """
        window_array = np.array(window_data)
        acc_data = window_array[:, :3]  # Arm_Acc_X, Y, Z
        gyro_data = window_array[:, 3:]  # Arm_Gyro_X, Y, Z
        
        # Calculate magnitudes
        acc_magnitude = np.mean(np.sqrt(np.sum(acc_data**2, axis=1)))
        gyro_magnitude = np.mean(np.sqrt(np.sum(gyro_data**2, axis=1)))
        
        # Calculate variability
        acc_std = np.std(acc_data)
        gyro_std = np.std(gyro_data)
        
        # Heuristic classification based on dataset statistics
        # DIAM: Low magnitude, low variability
        if acc_magnitude < 10.5 and gyro_magnitude < 0.8 and acc_std < 1.0:
            return "diam", acc_magnitude, gyro_magnitude, acc_std, gyro_std
        
        # LARI: High magnitude, high variability  
        elif acc_magnitude > 15 or gyro_magnitude > 2.0 or acc_std > 3.0:
            return "lari", acc_magnitude, gyro_magnitude, acc_std, gyro_std
        
        # JALAN: Medium range
        else:
            return "jalan", acc_magnitude, gyro_magnitude, acc_std, gyro_std
    
def predict_activity(self, window):
        """Make activity prediction with dataset-aware logic"""
        try:
            # Heuristic analysis first
            heuristic_label, acc_mag, gyro_mag, acc_std, gyro_std = self.classify_activity_heuristic(window)
            
            print(f"   📊 Data analysis:")
            print(f"      Acc magnitude: {acc_mag:.2f}")
            print(f"      Gyro magnitude: {gyro_mag:.2f}")
            print(f"      Acc std: {acc_std:.3f}")
            print(f"      Gyro std: {gyro_std:.3f}")
            print(f"      🧠 Heuristic: {heuristic_label.upper()}")
            
            # Strong override untuk clear patterns
            if (heuristic_label == "diam" and acc_mag < 10 and gyro_mag < 0.5 and acc_std < 0.5):
                print(f"   🎯 STRONG OVERRIDE: Clear DIAM pattern")
                override_probs = np.array([0.90, 0.08, 0.02])  # [diam, jalan, lari]
                return "diam", 0.90, override_probs
            
            # Model inference
            input_data = self.preprocess_window(window)
            if input_data is None:
                return None, 0.0, None
            
            print(f"   🤖 Running model inference...")
            prediction = self.model.predict(input_data, verbose=0)
            predicted_class = np.argmax(prediction)
            model_label = self.label_encoder.inverse_transform([predicted_class])[0]
            confidence = np.max(prediction)
            
            print(f"   📊 Model prediction: {model_label.upper()} ({confidence:.1%})")
            
            # Hybrid decision: combine model + heuristic
            if model_label == heuristic_label:
                # Both agree - high confidence
                print(f"   ✅ Model and heuristic agree!")
                return model_label, min(confidence + 0.1, 1.0), prediction[0]
            
            elif confidence < 0.7:
                # Low model confidence - trust heuristic
                print(f"   🔄 Low model confidence, using heuristic")
                # Create synthetic prediction favoring heuristic
                synthetic_probs = prediction[0].copy()
                heuristic_idx = list(self.label_encoder.classes_).index(heuristic_label)
                synthetic_probs[heuristic_idx] = 0.75
                synthetic_probs = synthetic_probs / np.sum(synthetic_probs)
                
                return heuristic_label, 0.75, synthetic_probs
            
            else:
                # High model confidence but disagrees with heuristic
                print(f"   ⚠️  Disagreement: Model={model_label.upper()}, Heuristic={heuristic_label.upper()}")
                return model_label, confidence * 0.8, prediction[0]  # Reduce confidence
            
        except Exception as e:
            print(f"   ❌ Prediction error: {e}")
            return None, 0.0, None
    
def print_prediction_results(self, label, confidence, all_predictions):
        """Print detailed prediction results"""
        print(f"\n🎯 ACTIVITY DETECTED: {label.upper()}")
        print(f"📊 Confidence: {confidence:.1%}")
        
        if all_predictions is not None:
            print(f"📈 Probability breakdown:")
            for i, (activity, prob) in enumerate(zip(self.label_encoder.classes_, all_predictions)):
                bar = "█" * int(prob * 25)
                marker = "🎯" if activity == label else "  "
                print(f"   {marker} {activity:<8}: {prob:.3f} {bar}")
        
        # Dataset context
        if label in self.dataset_stats:
            stats = self.dataset_stats[label]
            print(f"📚 Dataset reference for '{label}':")
            print(f"   Expected acc magnitude: ~{stats['acc_magnitude']:.1f}")
            print(f"   Expected gyro magnitude: ~{stats['gyro_magnitude']:.1f}")
        
        # Confidence interpretation
        if confidence > 0.85:
            print(f"✅ Very high confidence")
        elif confidence > 0.70:
            print(f"✅ High confidence")
        elif confidence > 0.55:
            print(f"⚠️  Medium confidence")
        else:
            print(f"❓ Low confidence - uncertain")
    
def run_real_time_recognition(self):
        """Main real-time recognition loop"""
        print("\n🚀 Starting Real-time Activity Recognition")
        print("=" * 60)
        print(f"📱 Model: MobileNetV2-LSTM")
        print(f"📊 Window: {self.window_size} samples ({self.window_size/50:.1f}s @ 50Hz)")
        print(f"🎯 Activities: {', '.join(self.label_encoder.classes_)}")
        print(f"📡 Format: 10+ columns (timestamp + 6 sensors + heart rate + extras)")
        print(f"🔥 Firebase: {'enabled' if self.firebase_initialized else 'disabled'}")
        print("=" * 60)
        print("🔄 Collecting sensor data...")
        print("-" * 60)
        
        try:
            while True:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    # Skip comments, empty lines, dan status messages
                    if not line or line.startswith(('✅', '❌', '📈', '📊', 'timestamp', '#', 'Sent:')):
                        continue
                    
                    # Extract heart rate data first
                    self.extract_heart_rate_data(line)
                    
                    # Parse data
                    parts = line.split(',')
                    # Accept 10+ kolom, ambil hanya sensor data
                    if len(parts) < 10:
                        self.invalid_count += 1
                        continue
                    
                    try:
                        # Extract timestamp and sensor data (hanya 6 sensor pertama)
                        timestamp = int(parts[0])
                        sensor_data = [float(x) for x in parts[1:7]]
                        
                        # Validate data
                        is_valid, reason = self.validate_sensor_data(timestamp, sensor_data)
                        if not is_valid:
                            self.invalid_count += 1
                            continue
                        
                        self.buffer.append(sensor_data)
                        self.data_count += 1
                        
                        # Progress update - dengan heart rate info
                        if self.data_count % 25 == 0:
                            progress = len(self.buffer) / self.window_size * 100
                            rate = self.data_count / (time.time() - self.start_time)
                            time_str = datetime.now().strftime("%H:%M:%S")
                            print(f"[{time_str}] 📥 Data: {self.data_count} | "
                                f"Buffer: {len(self.buffer)}/{self.window_size} ({progress:.1f}%) | "
                                f"Rate: {rate:.1f}Hz | BPM:{self.current_bmp:.0f} Avg:{self.current_beat_avg}")
                        
                        # Process when buffer full
                        if len(self.buffer) == self.window_size:
                            self.prediction_count += 1
                            print(f"\n🎯 Processing prediction #{self.prediction_count}")
                            
                            start_time = time.time()
                            label, confidence, all_preds = self.predict_activity(self.buffer)
                            process_time = time.time() - start_time
                            
                            if label:
                                self.print_prediction_results(label, confidence, all_preds)
                                print(f"❤️  Current BPM: {self.current_bmp:.1f}")
                                print(f"❤️  Average BPM: {self.current_beat_avg} ← (sent to Firebase)")
                                print(f"⏱️  Processing time: {process_time:.2f}s")
                                
                                # TAMBAH: Send to Firebase - use beat average
                                self.send_to_firebase(label, self.current_beat_avg)
                                
                                print("-" * 60)
                            
                            # Reset buffer
                            self.buffer = []
                            
                    except ValueError:
                        self.invalid_count += 1
                        continue
                
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print(f"\n👋 System stopped by user")
            self.print_final_statistics()
        except Exception as e:
            print(f"\n❌ System error: {e}")
        finally:
            if self.ser:
                self.ser.close()
                print("✅ Serial connection closed")

def print_final_statistics(self):
        """Print final session statistics"""
        runtime = time.time() - self.start_time
        total_data = self.data_count + self.invalid_count
        
        print(f"\n📊 Session Statistics:")
        print(f"   Runtime: {runtime/60:.1f} minutes")
        print(f"   Total data received: {total_data}")
        print(f"   Valid data: {self.data_count}")
        print(f"   Invalid data: {self.invalid_count}")
        print(f"   Data quality: {self.data_count/total_data*100:.1f}%" if total_data > 0 else "   Data quality: 0%")
        print(f"   Total predictions: {self.prediction_count}")
        print(f"   Average data rate: {self.data_count/runtime:.1f} Hz" if runtime > 0 else "   Average rate: 0 Hz")
        print(f"   Predictions per minute: {self.prediction_count/(runtime/60):.1f}" if runtime > 60 else "   Predictions: Too short to calculate")
        print(f"   Final BPM: {self.current_bmp:.1f}")
        print(f"   Final Average BPM: {self.current_beat_avg}")
        print(f"   Firebase: {'enabled' if self.firebase_initialized else 'disabled'}")

def run(self, firebase_service_account_path=None, firebase_database_url=None):
        """Main entry point"""
        # Load model components
        if not self.load_model_components():
            return False
        
        # Initialize Firebase if credentials provided
        if firebase_service_account_path and firebase_database_url:
            if not self.initialize_firebase(firebase_service_account_path, firebase_database_url):
                print("⚠️  Continuing without Firebase")
        else:
            print("⚠️  Firebase disabled")
        
        # Connect to ESP32
        if not self.connect_esp32():
            return False
        
        # Start real-time recognition
        self.run_real_time_recognition()
        
        return True

def main():
    """Main function"""
    print("🎯 ESP32 Activity Recognition System - 10 Column + Firebase")
    print("=" * 65)
    
    # Check model folder
    model_folder = "models_mobilenet"
    if not os.path.exists(model_folder):
        print(f"❌ Model folder '{model_folder}' not found!")
        print("💡 Make sure you have trained model available")
        return
    
    # === FIREBASE CONFIGURATION ===
    # Ganti dengan nama file service account key Anda
    FIREBASE_SERVICE_ACCOUNT_KEY_PATH = "firebase_credentials.json"
    
    # URL Firebase Anda
    FIREBASE_DATABASE_URL = "https://bpmmonitor-212-default-rtdb.asia-southeast1.firebasedatabase.app/"
    
    # Uncomment untuk mengaktifkan Firebase
    # FIREBASE_SERVICE_ACCOUNT_KEY_PATH = "your-service-account-file.json"
    # === END CONFIGURATION ===
    
    # Set to None untuk disable Firebase
    # FIREBASE_SERVICE_ACCOUNT_KEY_PATH = None
    # FIREBASE_DATABASE_URL = None

    print(f"🔥 Firebase: {'enabled' if FIREBASE_SERVICE_ACCOUNT_KEY_PATH else 'disabled'}")
    if FIREBASE_SERVICE_ACCOUNT_KEY_PATH:
        print(f"   File: {FIREBASE_SERVICE_ACCOUNT_KEY_PATH}")
        print(f"   URL: {FIREBASE_DATABASE_URL}")
        print(f"   Data: Activity + AvgBPM (beat average)")

    print("\n📡 Expected data format:")
    print("   timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,bmp,beatAvg,fingerDetected,dataRate")
    print("   (10+ columns, heart rate data displayed and sent to Firebase)")
    
    print("\n" + "=" * 65)
    
    # Initialize and run system
    system = ESP32ActivityRecognition(model_folder)
    success = system.run(
        firebase_service_account_path=FIREBASE_SERVICE_ACCOUNT_KEY_PATH,
        firebase_database_url=FIREBASE_DATABASE_URL
    )
    
    if success:
        print("\n🎉 Activity recognition session completed!")
    else:
        print("\n❌ Failed to start activity recognition")

if __name__ == "__main__":
    main()