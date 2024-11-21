import pandas as pd
import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Activation
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Define the base path for your dataset
BASE_PATH = '/home/sjagird1/DL Project/Emotions_Subset_3000/Emotions_Subset_3000/Emotions_Subset_3000/'  # Update this path
PLOT_DIR = 'plots'
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

def create_dataframe():
    """Create a dataframe with emotion labels and file paths"""
    file_emotion = []
    file_path = []
    
    for emotion in os.listdir(BASE_PATH):
        emotion_dir = os.path.join(BASE_PATH, emotion)
        
        if not os.path.isdir(emotion_dir):
            continue
            
        for filename in os.listdir(emotion_dir):
            if filename.endswith(('.wav', '.WAV')):
                file_path.append(os.path.join(emotion_dir, filename))
                file_emotion.append(emotion)
    
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    return pd.concat([emotion_df, path_df], axis=1)

def extract_melspectrogram(data, sr=22050):
    """Extract mel-spectrogram features from audio data"""
    # Generate mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=data,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        fmax=8000
    )
    
    # Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    return mel_spec_norm

def plot_training_history(history, save_path='plots/training_history.png'):
    """Enhanced plotting function with robust styling and error handling"""
    # Use basic matplotlib parameters
    plt.rcParams.update({
        'figure.figsize': (20, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 12
    })
    
    fig, axes = plt.subplots(1, 2)
    
    # Plot training & validation accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy', pad=15)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    
    # Plot training & validation loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss', pad=15)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    
    # Ensure plots directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Warning: Could not save plot to {save_path}: {str(e)}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_path='plots/confusion_matrix.png'):
    """Enhanced confusion matrix plotting with robust styling"""
    # Reset matplotlib parameters
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        'figure.figsize': (12, 10),
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    fig, ax = plt.subplots()
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    im = ax.imshow(cm_normalized, cmap='Blues')
    
    # Add colorbar
    plt.colorbar(im)
    
    # Set labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                         ha="center", va="center", color="black")
    
    # Add titles and labels
    ax.set_title('Normalized Confusion Matrix', pad=15)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # Ensure plots directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Warning: Could not save confusion matrix to {save_path}: {str(e)}")
    plt.close()

def build_conv2d_model(input_shape, num_emotions):
    """Build improved model with Conv2D layers"""
    model = Sequential([
        # First Conv2D block
        Conv2D(64, kernel_size=(3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.2),
        
        # Second Conv2D block
        Conv2D(128, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.2),
        
        # Third Conv2D block
        Conv2D(256, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.2),
        
        # Fourth Conv2D block
        Conv2D(512, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.2),
        
        # Flatten and Dense layers
        Flatten(),
        Dense(512),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        
        Dense(num_emotions, activation='softmax')
    ])
    
    # Use AdamW optimizer with weight decay
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=0.0001
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def compute_class_weights(y):
    """Compute balanced class weights"""
    y_integers = np.argmax(y, axis=1)
    class_weights = dict(zip(
        range(len(np.unique(y_integers))),
        len(y_integers) / (len(np.unique(y_integers)) * np.bincount(y_integers))
    ))
    return class_weights

def evaluate_model(model, x_test, y_test, encoder):
    """Comprehensive model evaluation with enhanced error handling"""
    try:
        # Get predictions
        predictions = model.predict(x_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=encoder.categories_[0]))
        
        # Plot confusion matrix
        try:
            plot_confusion_matrix(y_true, y_pred, classes=encoder.categories_[0])
        except Exception as e:
            print(f"Warning: Could not generate confusion matrix plot: {str(e)}")
        
        # Calculate and print additional metrics
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Additional metrics analysis
        print("\nPer-class accuracy:")
        for i, emotion in enumerate(encoder.categories_[0]):
            mask = y_true == i
            class_acc = np.mean(y_pred[mask] == y_true[mask])
            print(f"{emotion}: {class_acc*100:.2f}%")
            
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")

def main():
    print("Creating dataset from directory structure...")
    data_path = create_dataframe()
    print(f"Found {len(data_path)} audio files")
    print("\nEmotion distribution:")
    print(data_path['Emotions'].value_counts())
    
    print("\nExtracting mel-spectrograms...")
    X, Y = [], []
    target_length = 173  # Approximately 4 seconds of audio at our hop length
    
    # Keep track of shapes for debugging
    shapes_log = []
    
    for idx, (path, emotion) in enumerate(zip(data_path.Path, data_path.Emotions)):
        if idx % 100 == 0:
            print(f"Processing file {idx+1}/{len(data_path)}")
        
        try:
            # Load audio file
            data, sr = librosa.load(path, duration=4, offset=0.5, sr=22050)
            
            # Ensure consistent length
            if len(data) < sr * 4:
                data = np.pad(data, (0, sr * 4 - len(data)))
            else:
                data = data[:sr * 4]
            
            # Extract mel-spectrogram with fixed parameters
            mel_spec = extract_melspectrogram(data, sr)
            
            # Ensure consistent size and handle padding explicitly
            if mel_spec.shape[1] > target_length:
                mel_spec = mel_spec[:, :target_length]
            elif mel_spec.shape[1] < target_length:
                pad_width = target_length - mel_spec.shape[1]
                mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
            
            # Verify shape before adding
            if mel_spec.shape != (128, target_length):
                print(f"Unexpected shape {mel_spec.shape} for file {path}")
                continue
                
            X.append(mel_spec)
            Y.append(emotion)
            shapes_log.append(mel_spec.shape)
            
            # Data augmentation with shape verification
            # Time stretching
            stretched_data = librosa.effects.time_stretch(data, rate=1.2)
            stretched_spec = extract_melspectrogram(stretched_data, sr)
            stretched_spec = stretched_spec[:, :target_length] if stretched_spec.shape[1] > target_length else \
                           np.pad(stretched_spec, ((0, 0), (0, target_length - stretched_spec.shape[1])), mode='constant')
            
            if stretched_spec.shape == (128, target_length):
                X.append(stretched_spec)
                Y.append(emotion)
                shapes_log.append(stretched_spec.shape)
            
            # Pitch shifting
            shifted_data = librosa.effects.pitch_shift(data, sr=sr, n_steps=2)
            shifted_spec = extract_melspectrogram(shifted_data, sr)
            shifted_spec = shifted_spec[:, :target_length] if shifted_spec.shape[1] > target_length else \
                          np.pad(shifted_spec, ((0, 0), (0, target_length - shifted_spec.shape[1])), mode='constant')
            
            if shifted_spec.shape == (128, target_length):
                X.append(shifted_spec)
                Y.append(emotion)
                shapes_log.append(shifted_spec.shape)
            
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            continue
    
    # Verify all shapes are consistent before converting to numpy array
    unique_shapes = set(str(shape) for shape in shapes_log)
    if len(unique_shapes) > 1:
        print("Warning: Inconsistent shapes detected:", unique_shapes)
        return
    
    # Convert to numpy arrays with explicit type and shape
    X = np.array(X, dtype=np.float32)
    X = np.expand_dims(X, axis=-1)  # Add channel dimension for Conv2D
    
    # Encode labels
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(np.array(Y).reshape(-1,1))
    
    print(f"Final dataset shape: X: {X.shape}, Y: {Y.shape}")
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    # Initialize the model with correct input shape and number of emotions
    input_shape = (128, 173, 1)  # (mel_bins, time_steps, channels)
    num_emotions = Y.shape[1]  # Number of emotion classes
    model = build_conv2d_model(input_shape, num_emotions)
    
    # Training callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            mode='min'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=1e-6,
            mode='min'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model_conv2d.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Calculate class weights
    class_weights = compute_class_weights(y_train)
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=100,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        class_weight=class_weights,
        shuffle=True
    )
    
    # Plot training history and evaluate
    plot_training_history(history)
    evaluate_model(model, x_test, y_test, encoder)
    
    # Save the final model
    model.save('final_model_conv2d.keras')
    print("\nTraining completed. Model saved as 'final_model_conv2d.keras'")

if __name__ == "__main__":
    main()