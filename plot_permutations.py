import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

# Define the permutation directories (still using Permutation for directory names)
permutations = [f"Permutation{i}" for i in range(1, 7)]  # Permutation1 through Permutation6

# Set up the plot grid (2x3 grid for 6 permutations)
fig, axes = plt.subplots(2, 3, figsize=(12, 7))  # Reduced height to bring rows closer

# Adjust the spacing between subplots - reduced vertical spacing
plt.subplots_adjust(wspace=0.01, hspace=0.005)  # Reduced hspace further

# Flatten axes for easier iteration
axes = axes.flatten()

# For each permutation, load and display a sample image
for idx, perm in enumerate(permutations):
    # Build path to the train directory of each permutation
    train_dir = os.path.join('datasets', perm, 'train')
    
    # Get first image from the directory
    try:
        image_files = [f for f in os.listdir(train_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            img_path = os.path.join(train_dir, image_files[0])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            # Display image with "Combination" label instead of "Permutation"
            axes[idx].imshow(img)
            axes[idx].set_title(f"Combination{idx+1}", pad=1, fontsize=10)  # Changed label and reduced pad
            axes[idx].axis('off')
    except Exception as e:
        print(f"Error loading images from {perm}: {e}")

# Save with minimal borders
plt.savefig('all_permutations_overview.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close() 