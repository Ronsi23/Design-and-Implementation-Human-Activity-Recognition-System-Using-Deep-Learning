import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import joblib
import warnings
warnings.filterwarnings('ignore')


# Ini harus ada di sini, di awal skrip utama atau di bagian atas file
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs.")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

class MobileNetV2LSTMClassifier:
    def __init__(self, processed_folder="dataset/processed"):
        self.processed_folder = processed_folder
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None
        
        # Configuration
        self.sensor_columns = ['Arm_Acc_X', 'Arm_Acc_Y', 'Arm_Acc_Z', 
                              'Arm_Gyro_X', 'Arm_Gyro_Y', 'Arm_Gyro_Z']
        self.window_size = 128
        self.overlap = 0.5
        
        # Model architecture settings
        self.image_size = 32  # Size for MobileNetV2 input
        self.sequence_length = 16  # Number of time steps for LSTM
        self.step_size = self.window_size // self.sequence_length
        
        print("🚀 MobileNetV2-LSTM Activity Classifier initialized")
        print(f"   Window size: {self.window_size}")
        print(f"   Image size: {self.image_size}x{self.image_size}")
        print(f"   Sequence length: {self.sequence_length}")
        print(f"   Transfer Learning: MobileNetV2 + LSTM")
    
    def load_data(self):
        """Load train, validation, dan test data"""
        print("\n" + "="*60)
        print("LOADING INTEGRATED DATASETS")
        print("="*60)
        
        try:
            # Load datasets
            train_path = os.path.join(self.processed_folder, "train.csv")
            val_path = os.path.join(self.processed_folder, "validation.csv")
            test_path = os.path.join(self.processed_folder, "test.csv")
            
            self.train_df = pd.read_csv(train_path)
            self.val_df = pd.read_csv(val_path)
            self.test_df = pd.read_csv(test_path)
            
            # Clean column names dan activity names
            for df in [self.train_df, self.val_df, self.test_df]:
                df.columns = df.columns.str.strip()
                if 'Simple_Activity' in df.columns:
                    df['Simple_Activity'] = df['Simple_Activity'].str.strip()
            
            print(f"✅ Train data loaded: {self.train_df.shape}")
            print(f"✅ Validation data loaded: {self.val_df.shape}")
            print(f"✅ Test data loaded: {self.test_df.shape}")
            print(f"✅ Activities: {sorted(self.train_df['Simple_Activity'].unique())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def convert_timeseries_to_spectrogram(self, window_data):
        """
        Convert time series data ke spectrogram representation untuk CNN
        Input: (window_size, 6) sensor data
        Output: (image_size, image_size, 3) RGB-like image
        """
        spectrograms = []
        
        # Process each sensor channel
        for i in range(6):
            data = window_data[:, i]
            
            # Compute Short-Time Fourier Transform (STFT)
            frequencies, times, Zxx = signal.stft(
                data, 
                fs=50,  # Sampling frequency (adjust based on your data)
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
        # Use first 3 sensors for RGB channels
        if len(spectrograms) >= 3:
            # Resize spectrograms to target size
            rgb_image = np.zeros((self.image_size, self.image_size, 3))
            
            for i in range(3):
                # Resize spectrogram to image_size x image_size
                resized = tf.image.resize(
                    spectrograms[i][..., np.newaxis], 
                    [self.image_size, self.image_size]
                ).numpy().squeeze()
                rgb_image[:, :, i] = resized
        else:
            # Fallback: grayscale to RGB
            gray = spectrograms[0] if spectrograms else np.zeros((self.image_size, self.image_size))
            rgb_image = np.stack([gray, gray, gray], axis=-1)
        
        return rgb_image
    
    def convert_timeseries_to_patches(self, window_data):
        """
        Alternative: Convert time series ke 2D patches
        Input: (window_size, 6) sensor data  
        Output: (image_size, image_size, 3) patch-based image
        """
        # Reshape time series into 2D patches
        patch_height = self.image_size // 8  # 8x8 patches for 64x64 image
        patch_width = self.image_size // 8
        
        # Create 2D representation
        image = np.zeros((self.image_size, self.image_size, 3))
        
        # Method 1: Each sensor becomes a channel pattern
        for sensor_idx in range(min(6, 3)):  # Use first 3 sensors
            sensor_data = window_data[:, sensor_idx]
            
            # Normalize data
            if sensor_data.std() > 0:
                normalized_data = (sensor_data - sensor_data.mean()) / sensor_data.std()
            else:
                normalized_data = sensor_data
            
            # Map to [0, 1] range
            normalized_data = (normalized_data - normalized_data.min()) / (normalized_data.max() - normalized_data.min() + 1e-8)
            
            # Create 2D pattern
            for i in range(patch_height):
                for j in range(patch_width):
                    idx = (i * patch_width + j) % len(normalized_data)
                    y_start, y_end = i * 8, (i + 1) * 8
                    x_start, x_end = j * 8, (j + 1) * 8
                    image[y_start:y_end, x_start:x_end, sensor_idx] = normalized_data[idx]
        
        return image
    
    def create_sequences_for_mobilenet_lstm(self, df, method='spectrogram'):
        """
        Create sequences untuk MobileNetV2-LSTM
        Each sequence contains multiple 2D representations for temporal modeling
        """
        print(f"\n📊 Creating sequences for MobileNetV2-LSTM (method: {method})")
        
        sequences = []
        labels = []
        subject_ids = []
        
        step_size = int(self.window_size * (1 - self.overlap))
        
        for subject_id in sorted(df['Subject_ID'].unique()):
            subject_data = df[df['Subject_ID'] == subject_id]
            
            for activity in subject_data['Simple_Activity'].unique():
                activity_data = subject_data[subject_data['Simple_Activity'] == activity]
                sensor_data = activity_data[self.sensor_columns].values
                
                # Create overlapping windows
                for i in range(0, len(sensor_data) - self.window_size + 1, step_size):
                    window = sensor_data[i:i + self.window_size]
                    
                    # Create sequence of 2D representations
                    sequence = []
                    for t in range(0, self.window_size - self.step_size + 1, self.step_size):
                        sub_window = window[t:t + self.step_size]
                        
                        # Convert to 2D representation
                        if method == 'spectrogram':
                            image_2d = self.convert_timeseries_to_spectrogram(sub_window)
                        else:  # patches
                            image_2d = self.convert_timeseries_to_patches(sub_window)
                        
                        sequence.append(image_2d)
                    
                    # Pad or truncate sequence to fixed length
                    if len(sequence) > self.sequence_length:
                        sequence = sequence[:self.sequence_length]
                    elif len(sequence) < self.sequence_length:
                        # Repeat last frame to reach sequence_length
                        while len(sequence) < self.sequence_length:
                            sequence.append(sequence[-1])
                    
                    sequences.append(np.array(sequence))
                    labels.append(activity)
                    subject_ids.append(subject_id)
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        subject_ids = np.array(subject_ids)
        
        print(f"✅ Created {len(sequences)} sequences")
        print(f"   Shape: {sequences.shape}")
        print(f"   Activities: {np.unique(labels)}")
        print(f"   Subjects: {np.unique(subject_ids)}")
        
        return sequences, labels, subject_ids
    
    def prepare_data_for_training(self, method='spectrogram'):
        """
        Prepare data untuk MobileNetV2-LSTM training
        """
        print("\n" + "="*60)
        print("PREPARING DATA FOR MOBILENETV2-LSTM")
        print("="*60)
        
        # Create sequences untuk setiap split
        print("Creating training sequences...")
        X_train, y_train, train_subjects = self.create_sequences_for_mobilenet_lstm(
            self.train_df, method
        )
        
        print("Creating validation sequences...")
        X_val, y_val, val_subjects = self.create_sequences_for_mobilenet_lstm(
            self.val_df, method
        )
        
        print("Creating test sequences...")
        X_test, y_test, test_subjects = self.create_sequences_for_mobilenet_lstm(
            self.test_df, method
        )
        
        # Normalize sequences (per channel)
        print(f"\n📊 Normalizing data...")
        
        # Reshape untuk normalization
        original_shape = X_train.shape
        X_train_flat = X_train.reshape(-1, self.image_size * self.image_size * 3)
        X_val_flat = X_val.reshape(-1, self.image_size * self.image_size * 3)
        X_test_flat = X_test.reshape(-1, self.image_size * self.image_size * 3)
        
        # Fit scaler dan transform
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        X_val_scaled = self.scaler.transform(X_val_flat)
        X_test_scaled = self.scaler.transform(X_test_flat)
        
        # Reshape back
        self.X_train = X_train_scaled.reshape(original_shape)
        self.X_val = X_val_scaled.reshape(X_val.shape)
        self.X_test = X_test_scaled.reshape(X_test.shape)
        
        # Encode labels
        y_all = np.concatenate([y_train, y_val, y_test])
        self.label_encoder.fit(y_all)
        
        self.y_train = self.label_encoder.transform(y_train)
        self.y_val = self.label_encoder.transform(y_val)
        self.y_test = self.label_encoder.transform(y_test)
        
        # Convert to categorical
        self.num_classes = len(self.label_encoder.classes_)
        self.y_train_cat = to_categorical(self.y_train, self.num_classes)
        self.y_val_cat = to_categorical(self.y_val, self.num_classes)
        self.y_test_cat = to_categorical(self.y_test, self.num_classes)
        
        print(f"\n✅ Data preparation completed!")
        print(f"   X_train: {self.X_train.shape}")
        print(f"   X_val: {self.X_val.shape}")
        print(f"   X_test: {self.X_test.shape}")
        print(f"   Classes: {list(self.label_encoder.classes_)}")
        print(f"   Method: {method}")
        
        return True
    
    def build_mobilenetv2_lstm_model(self, fine_tune_layers=50):
        """
        Build MobileNetV2-LSTM model dengan transfer learning
        """
        print(f"\n🔧 Building MobileNetV2-LSTM model...")
        
        # Input layer
        sequence_input = Input(shape=(self.sequence_length, self.image_size, self.image_size, 3))
        
        # Load pre-trained MobileNetV2 (without top layers)
        base_model = MobileNetV2(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        # Freeze base model layers (transfer learning)
        base_model.trainable = False
        if fine_tune_layers > 0:
            # Unfreeze top layers untuk fine-tuning
            for layer in base_model.layers[-fine_tune_layers:]:
                layer.trainable = True
        
        print(f"   MobileNetV2 trainable layers: {sum([layer.trainable for layer in base_model.layers])}")
        
        # TimeDistributed wrapper untuk apply MobileNetV2 ke each frame
        cnn_features = TimeDistributed(base_model)(sequence_input)
        
        # Add some Dense layers after CNN
        x = TimeDistributed(Dense(256, activation='relu'))(cnn_features)
        x = TimeDistributed(Dropout(0.3))(x)
        x = TimeDistributed(Dense(128, activation='relu'))(x)
        
        # LSTM layers untuk temporal modeling
        x = LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(x)
        x = LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)(x)
        
        # Final dense layers
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=sequence_input, outputs=outputs)
        
        # Compile model
        optimizer = Adam(learning_rate=1e-4)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print(f"✅ MobileNetV2-LSTM model built successfully!")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        # Model summary (brief)
        model.summary(show_trainable=True)
        
        return model
    
    def train_model(self, epochs=100, batch_size=16, patience=15):
        """
        Train MobileNetV2-LSTM model
        """
        print("\n" + "="*60)
        print("TRAINING MOBILENETV2-LSTM MODEL")
        print("="*60)
        
        # Build model
        self.build_mobilenetv2_lstm_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_mobilenetv2_lstm.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print(f"🚀 Starting training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Early stopping patience: {patience}")
        
        # Train model
        self.history = self.model.fit(
            self.X_train, self.y_train_cat,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val_cat),
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        
        return self.history
    
    def evaluate_model(self):
        """
        Evaluate model performance
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Evaluate on validation dan test set
        val_loss, val_accuracy = self.model.evaluate(self.X_val, self.y_val_cat, verbose=0)
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test_cat, verbose=0)
        
        print(f"📊 MobileNetV2-LSTM Performance:")
        print(f"   Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        print(f"   Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
        
        # Predictions
        y_val_pred_prob = self.model.predict(self.X_val, verbose=0)
        y_test_pred_prob = self.model.predict(self.X_test, verbose=0)
        
        y_val_pred = np.argmax(y_val_pred_prob, axis=1)
        y_test_pred = np.argmax(y_test_pred_prob, axis=1)
        
        # Classification reports
        print(f"\n📋 Validation Set Classification Report:")
        print(classification_report(self.y_val, y_val_pred, 
                                  target_names=self.label_encoder.classes_, zero_division=0))
        
        print(f"\n📋 Test Set Classification Report:")
        print(classification_report(self.y_test, y_test_pred, 
                                  target_names=self.label_encoder.classes_, zero_division=0))
        
        # Store results
        self.results = {
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'y_val_true': self.y_val,
            'y_val_pred': y_val_pred,
            'y_test_true': self.y_test,
            'y_test_pred': y_test_pred
        }
        
        return self.results
    
    def plot_training_history(self):
        """
        Plot training history dan model performance
        """
        if self.history is None:
            print("❌ No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MobileNetV2-LSTM Training Results', fontsize=16, fontweight='bold')
        
        # Loss plot
        axes[0,0].plot(self.history.history['loss'], label='Training Loss', color='blue')
        axes[0,0].plot(self.history.history['val_loss'], label='Validation Loss', color='red')
        axes[0,0].set_title('Model Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0,1].plot(self.history.history['accuracy'], label='Training Accuracy', color='blue')
        axes[0,1].plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[0,1].set_title('Model Accuracy')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Confusion matrices
        if hasattr(self, 'results'):
            # Validation confusion matrix
            cm_val = confusion_matrix(self.results['y_val_true'], self.results['y_val_pred'])
            sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.label_encoder.classes_,
                        yticklabels=self.label_encoder.classes_, ax=axes[1,0])
            axes[1,0].set_title('Validation Confusion Matrix')
            axes[1,0].set_xlabel('Predicted')
            axes[1,0].set_ylabel('True')
            
            # Test confusion matrix
            cm_test = confusion_matrix(self.results['y_test_true'], self.results['y_test_pred'])
            sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens',
                        xticklabels=self.label_encoder.classes_,
                        yticklabels=self.label_encoder.classes_, ax=axes[1,1])
            axes[1,1].set_title('Test Confusion Matrix')
            axes[1,1].set_xlabel('Predicted')
            axes[1,1].set_ylabel('True')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_folder="models_mobilenet"):
        """
        Save trained model dan components
        """
        print(f"\n💾 Saving MobileNetV2-LSTM model to {model_folder}/")
        
        os.makedirs(model_folder, exist_ok=True)
        
        try:
            # Save model
            model_path = os.path.join(model_folder, "mobilenetv2_lstm_model.keras")
            self.model.save(model_path)
            print(f"✅ Model saved: {model_path}")
            
            # Save preprocessors
            scaler_path = os.path.join(model_folder, "scaler.pkl")
            encoder_path = os.path.join(model_folder, "label_encoder.pkl")
            
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.label_encoder, encoder_path)
            
            print(f"✅ Scaler saved: {scaler_path}")
            print(f"✅ Label encoder saved: {encoder_path}")
            
            # Save results summary
            if hasattr(self, 'results'):
                results_path = os.path.join(model_folder, "results_summary.txt")
                with open(results_path, 'w') as f:
                    f.write("MobileNetV2-LSTM Model Results Summary\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Validation Accuracy: {self.results['val_accuracy']:.4f}\n")
                    f.write(f"Test Accuracy: {self.results['test_accuracy']:.4f}\n")
                    f.write(f"Validation Loss: {self.results['val_loss']:.4f}\n")
                    f.write(f"Test Loss: {self.results['test_loss']:.4f}\n")
                    f.write(f"Classes: {list(self.label_encoder.classes_)}\n")
                    f.write(f"Architecture: MobileNetV2 + LSTM\n")
                    f.write(f"Image Size: {self.image_size}x{self.image_size}\n")
                    f.write(f"Sequence Length: {self.sequence_length}\n")
                print(f"✅ Results summary saved: {results_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def run_complete_pipeline(self, method='spectrogram', epochs=50, batch_size=8, 
                            fine_tune_layers=50, save_model=True):
        """
        Run complete MobileNetV2-LSTM training pipeline
        """
        print("🚀 " + "="*58)
        print("    MOBILENETV2-LSTM ACTIVITY CLASSIFICATION PIPELINE")
        print("🚀 " + "="*58)
        
        # Step 1: Load data
        if not self.load_data():
            return False
        
        # Step 2: Prepare data
        if not self.prepare_data_for_training(method):
            return False
        
        # Step 3: Train model
        self.train_model(epochs, batch_size)
        
        # Step 4: Evaluate model
        self.evaluate_model()
        
        # Step 5: Generate visualizations
        self.plot_training_history()
        
        # Step 6: Save model
        if save_model:
            self.save_model()
        
        # Final summary
        print("\n🎉 " + "="*58)
        print("    MOBILENETV2-LSTM PIPELINE COMPLETED!")
        print("🎉 " + "="*58)
        
        print(f"\n📊 Final Results:")
        print(f"   ✅ Validation Accuracy: {self.results['val_accuracy']:.4f}")
        print(f"   ✅ Test Accuracy: {self.results['test_accuracy']:.4f}")
        print(f"   🏗️  Architecture: MobileNetV2 + LSTM")
        print(f"   📱 Mobile-ready: YES")
        
        return True

# Main execution
if __name__ == "__main__":
    # Initialize classifier
    classifier = MobileNetV2LSTMClassifier(processed_folder="dataset/processed")
    
    # Run complete pipeline
    # Parameters yang bisa disesuaikan:
    success = classifier.run_complete_pipeline(
        method='spectrogram',     # 'spectrogram' or 'patches'
        epochs= 75,               # Training epochs (start with less untuk testing)
        batch_size=4,            # Small batch size karena memory intensive
        fine_tune_layers=50,     # Number of layers to fine-tune
        save_model=True
    )
    
    if success:
        print("\n🎯 Next steps:")
        print("  1. Compare dengan CNN-BiLSTM dan Window-based ML")
        print("  2. Test dengan different conversion methods")
        print("  3. Deploy untuk real-time inference")
        print("  4. Prepare comprehensive comparison untuk sidang")
    else:
        print("\n❌ Pipeline gagal. Check error messages di atas.")