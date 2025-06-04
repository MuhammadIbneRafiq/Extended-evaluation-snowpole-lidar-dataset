import os
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

selected_image = "combined_color"
img_height = 128
img_width = 1024
batch_size = 8   # Reduced batch size
epochs = 5       # Reduced number of epochs
num_classes = 1  # Only one class: 'pole'

# --- Visualize a few images with YOLO bounding boxes ---
def visualize_yolo_bboxes(images_dir, labels_dir, num_samples=5, img_height=128, img_width=1024):
    image_paths = glob.glob(os.path.join(images_dir, '*.png'))
    for img_path in image_paths[:num_samples]:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_width, img_height))
        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{img_basename}.txt")
        bboxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5 and parts[0] == '0':
                        x_center, y_center, w, h = map(float, parts[1:])
                        # Convert from normalized YOLO to pixel coordinates
                        x_center *= img_width
                        y_center *= img_height
                        w *= img_width
                        h *= img_height
                        x1 = int(x_center - w / 2)
                        y1 = int(y_center - h / 2)
                        x2 = int(x_center + w / 2)
                        y2 = int(y_center + h / 2)
                        bboxes.append((x1, y1, x2, y2))
        # Draw bboxes
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        plt.figure(figsize=(12, 3))
        plt.imshow(img)
        plt.title(f"{img_basename}.png with {len(bboxes)} pole(s)")
        plt.axis('off')
        plt.show()

# Example usage: visualize a few training images
train_images_dir = os.path.join("only_labels", "SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions", "SnowPole_Detection_Dataset", "combined_color", 'train')
train_labels_dir = os.path.join("labels", 'train')
visualize_yolo_bboxes(train_images_dir, train_labels_dir, num_samples=5, img_height=128, img_width=1024)

# --- End visualization section ---

def get_image_paths_and_labels(images_dir, labels_dir):
    image_paths = glob.glob(os.path.join(images_dir, "*.png"))
    binary_labels = []
    for img_path in image_paths:
        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{img_basename}.txt")
        label = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip() and line.strip().split()[0] == '0':
                        label = 1
                        break
        binary_labels.append(label)
    return image_paths, binary_labels

def parse_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def make_dataset(images_dir, labels_dir, batch_size, shuffle=True):
    image_paths, binary_labels = get_image_paths_and_labels(images_dir, labels_dir)
    image_paths = np.array(image_paths)
    binary_labels = np.array(binary_labels, dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((image_paths, binary_labels))
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, image_paths, binary_labels

# Build paths to dataset
images_path = os.path.join("only_labels", "SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions", 
                          "SnowPole_Detection_Dataset", "combined_color")
labels_path = "labels"

# Check if directories exist and if not, try alternate paths
if not os.path.exists(images_path):
    print(f"Warning: Could not find {images_path}")
    # Try alternate paths
    possible_paths = [
        "SnowPole_Detection_Dataset/combined_color",
        "combined_color"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            images_path = path
            print(f"Using alternate image path: {images_path}")
            break

print(f"Using images from: {images_path}")
print(f"Using labels from: {labels_path}")

# Load training data
train_images_dir = os.path.join(images_path, 'train')
train_labels_dir = os.path.join(labels_path, 'train')
print(f"Loading training data from {train_images_dir} and {train_labels_dir}...")
train_ds, train_image_paths, train_labels = make_dataset(train_images_dir, train_labels_dir, batch_size, shuffle=True)

# Load validation data
val_images_dir = os.path.join(images_path, 'valid')
val_labels_dir = os.path.join(labels_path, 'valid')
print(f"Loading validation data from {val_images_dir} and {val_labels_dir}...")
val_ds, val_image_paths, val_labels = make_dataset(val_images_dir, val_labels_dir, batch_size, shuffle=False)

# Load test data if available
test_images_dir = os.path.join(images_path, 'test')
test_labels_dir = os.path.join(labels_path, 'test')
has_test_data = os.path.exists(test_images_dir) and os.path.exists(test_labels_dir)

test_ds = None
test_image_paths = None
test_labels = None
if has_test_data:
    print(f"Loading test data from {test_images_dir} and {test_labels_dir}...")
    test_ds, test_image_paths, test_labels = make_dataset(test_images_dir, test_labels_dir, batch_size, shuffle=False)

# Create model
print("Building ResNet50 model for binary classification...")
resnet_model = Sequential()

# Load pretrained ResNet50 with global average pooling already included
pretrained_model = ResNet50(
    include_top=False,
    input_shape=(img_height, img_width, 3),
    pooling='avg',  # This already applies global average pooling
    weights='imagenet'
)

# Freeze the pretrained layers
for layer in pretrained_model.layers:
    layer.trainable = False

# Add layers to the model
resnet_model.add(pretrained_model)
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(1, activation='sigmoid'))  # Binary classification: pole present or not

# Compile the model
resnet_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print model summary
resnet_model.summary()

# Directory to save the best weights
checkpoint_dir = "resnet50_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "best_model.weights.h5")

# Create callbacks
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

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
history_path = os.path.join(checkpoint_dir, "training_history.png")
plt.savefig(history_path)
plt.close()
print(f"Training history saved to {history_path}")

# Load the best weights before evaluation
resnet_model.load_weights(checkpoint_path)

if has_test_data and test_ds is not None:
    print("Evaluating on test dataset with best weights...")
    # Get all test labels and predictions
    y_true = test_labels.astype(int)
    y_pred = []
    for batch_images, _ in test_ds:
        batch_preds = resnet_model.predict(batch_images)
        y_pred.extend((batch_preds > 0.5).astype(int).flatten())
    y_pred = np.array(y_pred)
    # Evaluate
    test_results = resnet_model.evaluate(test_ds)
    print(f"Test loss: {test_results[0]:.4f}")
    print(f"Test accuracy: {test_results[1]:.4f}")
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Pole', 'Pole'], labels=[0, 1], zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Pole', 'Pole'],
                yticklabels=['No Pole', 'Pole'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(checkpoint_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    # Print detailed report for a sample of test images
    print("\nDetailed Test Sample Report (Max 10 samples):")
    num_samples_to_show = min(len(test_image_paths), 10)
    for i in range(num_samples_to_show):
        filename = os.path.basename(test_image_paths[i])
        true_label = 'Pole' if y_true[i] == 1 else 'No Pole'
        pred_label = 'Pole' if y_pred[i] == 1 else 'No Pole'
        print(f"\n  Image: {filename}")
        print(f"    True Label: {true_label}")
        print(f"    Predicted Label: {pred_label}")
    print("\nNote on Metrics:")
    print("The model is a binary classifier predicting the presence/absence of poles.")
    print("It does not predict bounding box locations. Therefore, object detection metrics like mAP")
    print("cannot be calculated with this model. The report above includes ground truth bounding")
    print("boxes for informational purposes, showing the data that was used to determine the binary labels.")
    print("To get mAP, the model architecture would need to be changed to an object detection model (e.g., YOLO, SSD, Faster R-CNN).")
elif not has_test_data:
    print("No test dataset found. Skipping evaluation on test data.")
else:
    print("Test data was not loaded correctly, skipping evaluation.")