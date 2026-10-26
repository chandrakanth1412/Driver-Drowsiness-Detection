from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = str(PROJECT_ROOT / 'yawn_dataset' / 'dataset')

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# -------------------------------
# IMAGE PARAMETERS
# -------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# -------------------------------
# DATA GENERATORS
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

print("Train class labels:", train_generator.class_indices)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

print("Test class labels:", test_generator.class_indices)

# -------------------------------
# VGG16 BASE MODEL
# -------------------------------
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in base_model.layers:
    layer.trainable = False

# -------------------------------
# CUSTOM CLASSIFIER
# -------------------------------
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(2, activation="softmax")  # yawn / no_yawn
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# MODEL SUMMARY
# -------------------------------
model.summary()

# -------------------------------
# TRAIN MODEL
# -------------------------------
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10
)

# -------------------------------
# SAVE MODEL
# -------------------------------
model.save(str(PROJECT_ROOT / "code" / "yawn_drowsiness_vgg16.h5"))

print("✅ Yawning model training completed and model saved.")
