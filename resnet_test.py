import os
print("Script execution started...")
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

# Set parameters for dataset and model
selected_image = "combined_color"
img_height = 128
img_width = 1024
batch_size = 8   # Reduced batch size
epochs = 5       # Reduced number of epochs
num_classes = 1  # Only one class: 'pole'

def load_images_and_labels(images_dir, labels_dir, img_height, img_width):
    # Check if directories exist
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory does not exist: {images_dir}")
    if not os.path.exists(labels_dir):
        raise ValueError(f"Labels directory does not exist: {labels_dir}")
        
    # Get all image paths
    image_paths = glob.glob(os.path.join(images_dir, "*.png"))
    if not image_paths:
        raise ValueError(f"No PNG images found in {images_dir}")
    
    print(f"Found {len(image_paths)} images in {images_dir}")
    
    # Initialize lists for images, labels, filenames, and all boxes
    images = []
    binary_labels = []
    image_filenames = []
    all_boxes = [] # To store list of boxes for each image
    
    for img_path in image_paths:
        # Load and preprocess image first
        img = load_img(img_path, target_size=(img_height, img_width))
        img_array = img_to_array(img)
        
        # Get image base name (without extension)
        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Path to corresponding label file
        label_path = os.path.join(labels_dir, f"{img_basename}.txt")
        
        # Add image and filename to lists regardless of label presence
        images.append(img_array)
        image_filenames.append(img_basename)
        
        current_image_boxes = [] # Initialize for this image
        # Check if label exists
        if os.path.exists(label_path):
            # Load label file
            with open(label_path, 'r') as f:
                # Count number of pole instances in this image
                # Format: class_id x_center y_center width height
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 5 and int(parts[0]) == 0:  # Class 0 is 'pole'
                            center_x, center_y = float(parts[1]), float(parts[2])
                            width, height = float(parts[3]), float(parts[4])
                            current_image_boxes.append([center_x, center_y, width, height])
            
            all_boxes.append(current_image_boxes) # Store boxes for this image
            # If label file exists, label is 1 if poles are present, 0 otherwise
            binary_labels.append(1 if current_image_boxes else 0)
        else:
            print(f"Warning: No label file found for {img_basename} at {label_path}. Treating as a negative sample (no pole).")
            all_boxes.append([]) # No bounding boxes for a negative sample
            binary_labels.append(0) # Label as 0 (no pole)
    
    if not images:
        raise ValueError(f"No valid images with corresponding labels found.")
    
    # Convert to numpy arrays
    images_np = np.array(images)
    binary_labels_np = np.array(binary_labels)
    
    # Normalize image data
    images_np = images_np / 255.0
    
    print(f"Loaded {len(images_np)} images with corresponding labels")
    print(f"Positive samples (with poles): {np.sum(binary_labels_np)}")
    print(f"Negative samples (no poles): {len(binary_labels_np) - np.sum(binary_labels_np)}")
    
    return images_np, binary_labels_np, image_filenames, all_boxes

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
train_images_np, train_binary_labels_np, train_filenames, train_gt_boxes = load_images_and_labels(train_images_dir, train_labels_dir, img_height, img_width)
train_ds = tf.data.Dataset.from_tensor_slices((train_images_np, train_binary_labels_np))

# Load validation data
val_images_dir = os.path.join(images_path, 'valid')
val_labels_dir = os.path.join(labels_path, 'valid')
print(f"Loading validation data from {val_images_dir} and {val_labels_dir}...")
val_images_np, val_binary_labels_np, val_filenames, val_gt_boxes = load_images_and_labels(val_images_dir, val_labels_dir, img_height, img_width)
val_ds = tf.data.Dataset.from_tensor_slices((val_images_np, val_binary_labels_np))

# Batch the datasets
train_ds = train_ds.shuffle(len(train_images_np)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Load test data if available
test_images_dir = os.path.join(images_path, 'test')
test_labels_dir = os.path.join(labels_path, 'test')
has_test_data = os.path.exists(test_images_dir) and os.path.exists(test_labels_dir)

# These will store the full test data details for evaluation
test_images_np_full = None
test_binary_labels_np_full = None
test_filenames_full = None
test_gt_boxes_full = None
test_ds_for_evaluation = None # This will be used for model.evaluate and model.predict

if has_test_data:
    print(f"Loading test data from {test_images_dir} and {test_labels_dir}...")
    test_images_np_full, test_binary_labels_np_full, test_filenames_full, test_gt_boxes_full = load_images_and_labels(test_images_dir, test_labels_dir, img_height, img_width)
    # Create a tf.data.Dataset for model.evaluate and model.predict, containing only images and binary labels
    test_ds_for_evaluation = tf.data.Dataset.from_tensor_slices((test_images_np_full, test_binary_labels_np_full))
    test_ds_for_evaluation = test_ds_for_evaluation.batch(batch_size).prefetch(tf.data.AUTOTUNE)

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
# Do not add another pooling layer since ResNet with pooling='avg' already does this
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

if has_test_data and test_ds_for_evaluation is not None and test_images_np_full is not None:
    print("Evaluating on test dataset with best weights...")
    
    # Make predictions using the batched dataset for efficiency
    # test_images_np_full is already normalized and prepared by load_images_and_labels
    test_predictions_probs = resnet_model.predict(test_images_np_full) # Predict on the numpy array directly
    test_pred_binary_classes = (test_predictions_probs > 0.5).astype(int).flatten()
    
    # test_binary_labels_np_full are the true binary labels
    true_binary_labels = test_binary_labels_np_full 

    # Calculate standard metrics using the batched dataset for model.evaluate
    test_results = resnet_model.evaluate(test_ds_for_evaluation) # Use the dataset for evaluation
    print(f"Test loss: {test_results[0]:.4f}")
    print(f"Test accuracy: {test_results[1]:.4f}")
    
    # Calculate additional metrics using the flattened predictions and true labels
    precision = precision_score(true_binary_labels, test_pred_binary_classes, zero_division=0)
    recall = recall_score(true_binary_labels, test_pred_binary_classes, zero_division=0)
    f1 = f1_score(true_binary_labels, test_pred_binary_classes, zero_division=0)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print full classification report
    print("\nClassification Report:")
    print(classification_report(true_binary_labels, test_pred_binary_classes, target_names=['No Pole', 'Pole'], labels=[0, 1], zero_division=0))
    
    # Generate and plot confusion matrix
    cm = confusion_matrix(true_binary_labels, test_pred_binary_classes)
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
    num_samples_to_show = min(len(test_filenames_full), 10)
    for i in range(num_samples_to_show):
        filename = test_filenames_full[i]
        true_label = 'Pole' if true_binary_labels[i] == 1 else 'No Pole'
        pred_label = 'Pole' if test_pred_binary_classes[i] == 1 else 'No Pole'
        gt_boxes = test_gt_boxes_full[i]
        
        print(f"\n  Image: {filename}.png")
        print(f"    True Label: {true_label}")
        print(f"    Predicted Label: {pred_label}")
        if gt_boxes:
            print(f"    Ground Truth Bounding Boxes (YOLO format: x_center y_center width height):")
            for box in gt_boxes:
                print(f"      {box}")
        else:
            print("    Ground Truth Bounding Boxes: None")
    
    # Revised note about object detection metrics
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
