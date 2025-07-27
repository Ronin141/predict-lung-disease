# Lung Sound Classification Training - Fixed GPU Version
# Optimized for Google Colab with improved GPU handling

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten, GlobalAveragePooling2D, SeparableConv2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import os
import pickle
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = '/content/drive/MyDrive/ICBHI_final_database'
MODEL_PATH = '/content/drive/MyDrive/lung_sound_model.h5'
ENCODER_PATH = '/content/drive/MyDrive/label_encoder.pkl'

class LungSoundTrainer:
    def __init__(self, data_path, model_path, encoder_path):
        self.data_path = data_path
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.gpu_available = self.setup_gpu()
        self.batch_size = self.determine_batch_size()
        self.label_encoder = LabelEncoder()

    def setup_gpu(self):
        """Comprehensive GPU setup with proper error handling"""
        print("=== GPU Configuration ===")

        # Check TensorFlow version
        print(f"TensorFlow version: {tf.__version__}")

        try:
            # Check for available GPUs
            gpus = tf.config.list_physical_devices('GPU')
            print(f"Physical GPUs detected: {len(gpus)}")

            if len(gpus) == 0:
                print("❌ No GPU found. Training will use CPU.")
                return False

            # Print GPU details
            for i, gpu in enumerate(gpus):
                print(f"GPU {i}: {gpu}")

            # Get logical GPUs (after memory growth is set)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Logical GPUs: {len(logical_gpus)}")

            # Configure GPU memory growth
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ GPU memory growth enabled")
            except RuntimeError as e:
                print(f"⚠️  Memory growth must be set before GPUs have been initialized: {e}")
                # This is normal if GPUs have already been initialized

            # Test GPU with a simple operation
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                result = tf.linalg.matmul(test_tensor, test_tensor)
                print(f"✅ GPU test successful: {result.numpy()}")

            # Check current device
            print(f"Current device: {tf.config.list_logical_devices()}")

            # Enable mixed precision if available
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("✅ Mixed precision enabled (float16)")
            except Exception as e:
                print(f"⚠️  Mixed precision not available: {e}")

            return True

        except Exception as e:
            print(f"❌ GPU setup failed: {e}")
            print("Training will continue on CPU")
            return False

    def determine_batch_size(self):
        """Determine optimal batch size based on available resources"""
        if self.gpu_available:
            try:
                # Try to get GPU memory info
                gpu_devices = tf.config.list_physical_devices('GPU')
                if gpu_devices:
                    # Default batch sizes based on typical GPU memory
                    return 64  # Start with larger batch size for GPU
            except:
                pass
            return 32
        else:
            return 16  # Smaller batch size for CPU

    def check_gpu_utilization(self):
        """Check if GPU is actually being used during training"""
        try:
            if tf.config.list_physical_devices('GPU'):
                with tf.device('/GPU:0'):
                    # Create a small computation to test GPU usage
                    a = tf.random.normal((1000, 1000))
                    b = tf.random.normal((1000, 1000))
                    c = tf.matmul(a, b)
                    print("✅ GPU computation test passed")
                    return True
        except Exception as e:
            print(f"⚠️  GPU utilization check failed: {e}")
        return False

    def pad_or_truncate(self, mel_spec, target_length=128):
        """Pad or truncate mel-spectrogram to target length"""
        if mel_spec.shape[1] > target_length:
            return mel_spec[:, :target_length]
        else:
            pad_width = target_length - mel_spec.shape[1]
            return np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')

    def load_icbhi_data(self):
        """Load ICBHI Dataset with error handling"""
        features = []
        labels = []

        print(f"Loading data from: {self.data_path}")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path {self.data_path} not found!")

        # Get txt files, excluding patient_diagnosis.txt
        txt_files = [f for f in os.listdir(self.data_path)
                    if f.endswith('.txt') and not f.startswith('patient_diagnosis')]
        total_files = len(txt_files)
        processed_files = 0

        print(f"Found {total_files} annotation files")

        for txt_file in txt_files:
            base_filename = txt_file.replace('.txt', '')
            audio_file = base_filename + '.wav'

            if os.path.exists(os.path.join(self.data_path, audio_file)):
                try:
                    # Load audio
                    y, sr = librosa.load(os.path.join(self.data_path, audio_file), sr=4000)

                    # Read annotations
                    with open(os.path.join(self.data_path, txt_file), 'r') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 4:
                                start_time = float(parts[0])
                                end_time = float(parts[1])
                                crackles = int(parts[2]) if parts[2].isdigit() else 0
                                wheezes = int(parts[3]) if parts[3].isdigit() else 0

                                # Create label
                                if crackles == 0 and wheezes == 0:
                                    label = 0  # normal
                                elif crackles == 1 and wheezes == 0:
                                    label = 1  # crackles
                                elif crackles == 0 and wheezes == 1:
                                    label = 2  # wheezes
                                else:
                                    label = 3  # both

                                # Extract audio segment
                                start_sample = int(start_time * sr)
                                end_sample = int(end_time * sr)

                                if end_sample > len(y):
                                    end_sample = len(y)

                                if start_sample >= end_sample:
                                    continue

                                segment = y[start_sample:end_sample]

                                if len(segment) > sr:  # At least 1 second
                                    # Convert to mel-spectrogram
                                    mel_spec = librosa.feature.melspectrogram(
                                        y=segment, sr=sr, n_mels=64, n_fft=512
                                    )
                                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                                    mel_spec_db = self.pad_or_truncate(mel_spec_db)

                                    features.append(mel_spec_db)
                                    labels.append(label)

                    processed_files += 1
                    if processed_files % 100 == 0:
                        print(f"Processed {processed_files}/{total_files} files...")

                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue

        print(f"Loaded {len(features)} samples from {processed_files} files")

        # Print class distribution
        if len(labels) > 0:
            unique, counts = np.unique(labels, return_counts=True)
            print("Class distribution:")
            class_names = ['Normal', 'Crackles', 'Wheezes', 'Both']
            for i, (cls, count) in enumerate(zip(unique, counts)):
                print(f"  {class_names[cls]}: {count} samples ({count/len(labels)*100:.1f}%)")

        return np.array(features), np.array(labels)

    def create_improved_cnn_model(self, input_shape, num_classes):
        """Create improved CNN model for lung sound classification"""
        model = Sequential([
            # First Conv Block
            Conv2D(32, (5, 5), activation='relu', input_shape=input_shape,
                   padding='same', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Second Conv Block
            Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Third Conv Block
            Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),

            # Fourth Conv Block
            Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),

            # Global Average Pooling
            GlobalAveragePooling2D(),

            # Dense layers
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),

            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),

            # Output layer
            Dense(num_classes, activation='softmax',
                  dtype='float32' if self.gpu_available else 'float32')
        ])

        return model

    def create_lightweight_model(self, input_shape, num_classes):
        """Create lightweight model for faster training"""
        model = Sequential([
            SeparableConv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),

            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax', dtype='float32')
        ])

        return model

    def create_callbacks(self):
        """Create training callbacks"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True,
                verbose=1,
                monitor='val_accuracy',
                min_delta=0.001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model_checkpoint.h5',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1,
                save_weights_only=False
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1,
                monitor='val_loss',
                cooldown=3
            ),
            tf.keras.callbacks.CSVLogger(
                'training_log.csv',
                append=False,
                separator=','
            )
        ]

        # Add TensorBoard callback if GPU is available
        if self.gpu_available:
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir='./tensorboard_logs',
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq='epoch'
                )
            )

        return callbacks

    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Learning Rate
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)

        # Training Summary
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        total_epochs = len(history.history['loss'])

        axes[1, 1].text(0.1, 0.8, f'Best Validation Accuracy: {best_val_acc:.4f}',
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'Best Epoch: {best_epoch}',
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.4, f'Total Epochs: {total_epochs}',
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.2, f'Device: {"GPU" if self.gpu_available else "CPU"}',
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate_model(self, model, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n=== Model Evaluation ===")

        # Check GPU usage during prediction
        if self.gpu_available:
            print("Performing GPU prediction test...")
            self.check_gpu_utilization()

        # Predictions
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # Basic metrics
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        precision = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
        recall = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
        f1 = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (macro): {precision:.4f}")
        print(f"Recall (macro): {recall:.4f}")
        print(f"F1-score (macro): {f1:.4f}")

        # Classification report
        class_names = ['Normal', 'Crackles', 'Wheezes', 'Both']
        try:
            print("\nClassification Report:")
            print(classification_report(y_true_classes, y_pred_classes,
                                      target_names=class_names, zero_division=0))
        except Exception as e:
            print(f"Error in classification report: {e}")

        # Confusion Matrix
        try:
            cm = confusion_matrix(y_true_classes, y_pred_classes)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")

        return accuracy, precision, recall, f1

    def train(self, model_type='improved', epochs=50):
        """Main training function"""
        print("🫁 Starting Lung Sound Classification Training...")
        print(f"Device: {'GPU' if self.gpu_available else 'CPU'}")

        # Load data
        X, y = self.load_icbhi_data()

        if len(X) == 0:
            raise ValueError("No data loaded! Please check your data path.")

        # Preprocessing
        print("Preprocessing data...")
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        X = X.astype(np.float32)

        # Normalize
        X_min, X_max = X.min(), X.max()
        if X_max > X_min:
            X = (X - X_min) / (X_max - X_min)

        print(f"Data shape: {X.shape}")
        print(f"Data range: [{X.min():.4f}, {X.max():.4f}]")

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded).astype(np.float32)

        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Class distribution: {np.bincount(y_encoded)}")

        # Train-validation-test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_categorical, test_size=0.15, random_state=42, stratify=y_categorical
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
        )

        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")

        # Create model
        input_shape = X_train.shape[1:]
        num_classes = y_categorical.shape[1]

        print(f"Creating {model_type} model...")
        if model_type == 'improved':
            model = self.create_improved_cnn_model(input_shape, num_classes)
        elif model_type == 'lightweight':
            model = self.create_lightweight_model(input_shape, num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Compile model
        if self.gpu_available:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=0.001,
                weight_decay=0.01
            )
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("Model created successfully!")
        print(f"Total parameters: {model.count_params():,}")

        # Print model summary for first few layers
        print("\nModel architecture:")
        model.summary()

        # Final GPU check before training
        if self.gpu_available:
            print("Performing final GPU check...")
            self.check_gpu_utilization()

        # Create callbacks
        callbacks = self.create_callbacks()

        # Training
        print(f"\n🚀 Starting training for {epochs} epochs...")
        print(f"Batch size: {self.batch_size}")

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluation
        print("\n📊 Evaluating model...")
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Detailed evaluation
        accuracy, precision, recall, f1 = self.evaluate_model(model, X_test, y_test)

        # Plot training history
        self.plot_training_history(history)

        # Save model and results
        print("💾 Saving model and results...")
        model.save(self.model_path)

        with open(self.encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)

        with open('training_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)

        print(f"Model saved as: {self.model_path}")
        print(f"Label encoder saved as: {self.encoder_path}")

        # Training summary
        print(f"\n{'='*60}")
        print("🎯 TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Device: {'🚀 GPU' if self.gpu_available else '💻 CPU'}")
        print(f"Model Type: {model_type}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Total Epochs: {len(history.history['loss'])}")
        print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
        print(f"Final Test Accuracy: {test_accuracy:.4f}")
        print(f"Final Test Precision: {precision:.4f}")
        print(f"Final Test Recall: {recall:.4f}")
        print(f"Final Test F1-Score: {f1:.4f}")
        if self.gpu_available:
            print("TensorBoard logs: ./tensorboard_logs")
        print(f"{'='*60}")

        return model, history

# Usage example
def main():
    """Main function to run the training"""
    trainer = LungSoundTrainer(DATA_PATH, MODEL_PATH, ENCODER_PATH)

    # Train with improved model
    model, history = trainer.train(model_type='improved', epochs=50)

    print("✅ Training completed successfully!")

if __name__ == "__main__":
    main()