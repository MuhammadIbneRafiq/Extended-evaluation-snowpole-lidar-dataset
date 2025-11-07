"""
Quick script to view the normalization comparison plot
"""
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pathlib import Path

# Path to comparison plot
comparison_plot = Path(r"C:\Users\muham\OneDrive - TU Eindhoven\Extended-evaluation-snowpole-lidar-dataset\SnowPole_Detection_Dataset\range-normalized\comparison_plots\image_1_comparison.png")

if comparison_plot.exists():
    print("=" * 80)
    print("NORMALIZATION COMPARISON VISUALIZATION")
    print("=" * 80)
    print(f"\nDisplaying: {comparison_plot.name}")
    print("\nThis plot shows:")
    print("  - Top row: Visual comparison of original vs. 5 normalization methods")
    print("  - Bottom row: Histogram distributions for each method")
    print("\n" + "=" * 80)
    
    # Load and display the comparison plot
    img = imread(str(comparison_plot))
    
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('Range Image Normalization Comparison\n(Original vs. 5 Normalization Techniques)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Comparison plot displayed successfully!")
    print(f"\nFull path: {comparison_plot}")
else:
    print(f"Error: Comparison plot not found at {comparison_plot}")
