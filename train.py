import tensorflow as tf
from tensorflow.keras import layers, models

# Smaller image size
data_dir = "dataset/asl_alphabet_train"   # <--- CHANGE if your dataset folder is different
#image_size = (224, 224)
#batch_size = 32
#num_classes = 29   # A-Z
epochs = 5

img_height, img_width = 100, 100
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical"  # Important
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical"
)

# 🛑 Shrink dataset for faster testing
train_ds = train_ds.shuffle(10000).take(10000)
val_ds = val_ds.shuffle(2000).take(2000)

# Define simple model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(29, activation='softmax')  # 29 letters
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train FAST
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5  # Only 5 for speed
)

# Save if you want
model.save('mini_model.h5')


"""import tensorflow as tf
from tensorflow.keras import layers, models
import os

# -------------- CONFIG --------------
data_dir = "dataset/asl_alphabet_train"   # <--- CHANGE if your dataset folder is different
image_size = (224, 224)
batch_size = 32
num_classes = 29   # A-Z
epochs = 5

# -------------- LOAD DATA --------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical"
)

# -------------- PREPROCESSING --------------
# Preprocess inputs like MobileNetV2 expects
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

# Prefetch for speed
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# -------------- MODEL --------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze the convolutional base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------- TRAIN --------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# -------------- SAVE MODEL --------------
model.save('asl_model_ugh.h5')

print("✅ Model training complete and saved as 'asl_model.h5'")
"""