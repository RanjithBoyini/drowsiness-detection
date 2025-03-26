import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Automatically get the dataset folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets the 'scripts' folder path
PROJECT_DIR = os.path.dirname(BASE_DIR)  # Moves one level up to the main project folder

TRAIN_DIR = os.path.join(PROJECT_DIR, "dataset/train")
VAL_DIR = os.path.join(PROJECT_DIR, "dataset/val")

# ✅ Step 1: Check if dataset folders exist
print(f"🔍 Checking dataset paths...")
print(f"TRAIN_DIR: {TRAIN_DIR}")
print(f"VAL_DIR: {VAL_DIR}")

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"❌ Train directory not found: {TRAIN_DIR}")
if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(f"❌ Validation directory not found: {VAL_DIR}")

# ✅ Step 2: Check if dataset contains images
def count_images(folder):
    return sum([len(files) for _, _, files in os.walk(folder)])

num_train_images = count_images(TRAIN_DIR)
num_val_images = count_images(VAL_DIR)

print(f"📸 Training images: {num_train_images}")
print(f"📸 Validation images: {num_val_images}")

if num_train_images == 0 or num_val_images == 0:
    raise ValueError("🚨 Error: No images found in dataset! Add images to 'train/' and 'val/' folders.")

# ✅ Step 3: Define image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ✅ Step 4: Data augmentation & preprocessing
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')

val_generator = datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')

# ✅ Step 5: Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (alert vs. drowsy)
])

# ✅ Step 6: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Step 7: Train the model
print("\n🚀 Training CNN Model...\n")
model.fit(train_generator, validation_data=val_generator, epochs=10)

# ✅ Step 8: Save the trained model dynamically
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)  # Create 'models' folder if it doesn't exist

MODEL_PATH = os.path.join(MODEL_DIR, "cnn_drowsiness.h5")
model.save(MODEL_PATH)

print(f"\n✅ CNN Model Trained and Saved at: {MODEL_PATH}")
