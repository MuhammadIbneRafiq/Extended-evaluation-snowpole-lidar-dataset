import os
import glob
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set parameters for dataset and model
selected_permutation = "Permutation1"
img_height = 128
img_width = 1024
batch_size = 16
epochs = 10

def load_images_from_directory(directory_path, img_height, img_width):
    image_paths = glob.glob(os.path.join(directory_path, "*.png"))
    if not image_paths:
        raise ValueError(f"No PNG images found in {directory_path}")
    
    print(f"Found {len(image_paths)} images in {directory_path}")
    
    # Load and preprocess images
    images = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=(img_height, img_width))
        img_array = img_to_array(img)
        images.append(img_array)
    
    # Convert to numpy array and normalize
    images = np.array(images)
    images = images / 255.0  # Normalize to [0,1]
    
    # Create simple labels (just sequential numbers)
    labels = np.arange(len(images))
    
    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset, len(images)

# Build path to dataset
permutation_path = os.path.join("datasets", selected_permutation)
if not os.path.exists(permutation_path):
    raise ValueError(f"Dataset directory not found: {permutation_path}")

print(f"Using dataset: {permutation_path}")

# Load training data
train_dir = os.path.join(permutation_path, 'train')
print(f"Loading training data from {train_dir}...")
train_ds, num_classes = load_images_from_directory(train_dir, img_height, img_width)

# Split into train and validation
total_size = len(list(train_ds))
val_size = int(0.2 * total_size)
train_size = total_size - val_size

train_ds = train_ds.shuffle(total_size, seed=123)
val_ds = train_ds.take(val_size)
train_ds = train_ds.skip(val_size)

# Batch the datasets
train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Load test data if available
test_dir = os.path.join(permutation_path, 'test')
has_test_data = os.path.exists(test_dir)
if has_test_data:
    print(f"Loading test data from {test_dir}...")
    test_ds, _ = load_images_from_directory(test_dir, img_height, img_width)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

print(f"Number of unique images (classes): {num_classes}")

# Create model
print("Building ResNet50 model...")
resnet_model = Sequential()

# Load pretrained ResNet50
pretrained_model = ResNet50(
    include_top=False,
    input_shape=(img_height, img_width, 3),
    pooling='avg',
    weights='imagenet'
)

# Freeze the pretrained layers
for layer in pretrained_model.layers:
    layer.trainable = False

# Add layers to the model
resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(num_classes, activation='softmax'))

# Compile the model
resnet_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
resnet_model.summary()

# Directory to save the best weights
checkpoint_dir = "resnet50_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "best_model.weights.h5")

# Create ModelCheckpoint callback
checkpoint_cb = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# Train the model
print(f"Training model for {epochs} epochs...")
history = resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint_cb]
)

# Load the best weights before evaluation
resnet_model.load_weights(checkpoint_path)

if has_test_data:
    print("Evaluating on test dataset with best weights...")
    test_results = resnet_model.evaluate(test_ds)
    print(f"Test loss: {test_results[0]:.4f}")
    print(f"Test accuracy: {test_results[1]:.4f}")
