import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from pathlib import Path
from ultralytics import YOLO
import warnings

class WorkingMultiSpectralDetector:
    def __init__(self, base_dir=".", version_config=None):
        self.base_dir = Path(base_dir)
        dataset_path = self.base_dir / "main_images/SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions/SnowPole_Detection_Dataset"
        self.modality_paths = {
            'signal': dataset_path / "signal/test",
            'reflec': dataset_path / "reflec/test", 
            'nearir': dataset_path / "nearir/test",
            'range': dataset_path / "range/test",
            'combined_color': dataset_path / "combined_color/test"
        }
        
        self.weights_dir = Path("trained_weights")
        
        # Individual modality configurations
        self.individual_modalities = {
            'signal': {'color': 'green', 'weight': 'train_signal_11n/weights/best.pt'},
            'reflec': {'color': 'purple', 'weight': 'train_reflec_11n/weights/best.pt'}, 
            'nearir': {'color': 'red', 'weight': 'train_narir_11n/weights/best.pt'},
            'range': {'color': 'blue', 'weight': 'train_range_11n/weights/best.pt'},
            'combined_color': {'color': 'orange', 'weight': 'train_combcolor_11n/weights/best.pt'}
        }
        if version_config:
            self.individual_modalities = version_config
        
        self.models = {}
        self._load_models()
        
    def _load_models(self):        
        # Load individual modality models
        for modality, mod_info in self.individual_modalities.items():
            weight_path = self.weights_dir / mod_info['weight']
            model = YOLO(str(weight_path))
            self.models[f'single_{modality}'] = model

    def load_modality_image(self, modality, image_name):
        """Load image from modality directory"""
        for ext in ['.png', '.jpg', '.jpeg']:
            img_name = image_name.replace('.png', ext).replace('.jpg', ext)
            img_path = self.modality_paths[modality] / img_name
            if img_path.exists():
                if modality == 'combined_color':
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        return img
        return None
    
    def create_rgb_combination(self, image_name, combo_info):
        """Create RGB from modality combination"""
        r_img = self.load_modality_image(combo_info['R'], image_name)
        g_img = self.load_modality_image(combo_info['G'], image_name)  
        b_img = self.load_modality_image(combo_info['B'], image_name)
        
        # Ensure same size
        h, w = r_img.shape
        g_img = cv2.resize(g_img, (w, h))
        b_img = cv2.resize(b_img, (w, h))
        
        return np.stack([r_img, g_img, b_img], axis=2)
    
    def load_ground_truth_labels(self, image_name):
        label_name = image_name.replace('.png', '.txt').replace('.jpg', '.txt')
        
        label_paths = [
            self.base_dir / "main_images/SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions/SnowPole_Detection_Dataset/labels" / label_name,
            self.base_dir / "labels" / label_name,
            self.base_dir / "RGB_Combinations/Combination1/test" / label_name
        ]
        
        for label_path in label_paths:
            if label_path.exists():
                labels = []
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id, x_center, y_center, width, height = map(float, parts[:5])
                            labels.append([class_id, x_center, y_center, width, height])
                return labels
        return []
    
    def run_inference(self, model_key, image, conf_threshold=0.1):
        
            
        start_time = time.time()
        
        try:
            # Ensure image is in correct format
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB
                pass  # Already correct
            else:
                return [], 0.0
            
            # Run inference
            results = self.models[model_key](image, conf=conf_threshold, verbose=False)
            inference_time = (time.time() - start_time) * 1000
            
            detections = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    detections.append([x1, y1, x2, y2, conf, cls])
            
            return detections, inference_time
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return [], 0.0
    
    def draw_detections(self, ax, detections, img_height, img_width, modality=None):
        """Draw detection boxes with non-overlapping labels."""
        
        # For NEARIR modality, show all detections without filtering
        if modality == 'nearir':
            # Round confidences to 1 decimal place for display
            print(f"All NEARIR detections: {[(round(d[4], 1), int(d[0])) for d in detections]}")  # Debug: show conf and x1
        
        for i, (x1, y1, x2, y2, conf, cls) in enumerate(detections):
            # Draw bounding box
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=1, edgecolor='lime', facecolor='none', linestyle='-')
            ax.add_patch(rect)
            
            # Simple label placement with vertical staggering
            label_text = f'pole {conf:.1f}' if modality == 'nearir' else f'pole {conf:.2f}'
            
            # Stagger labels vertically to avoid overlap
            label_y = y1 - 15 - (i * 18)  # Each label 18 pixels higher than the previous
            
            ax.text(x1, label_y, label_text, color='white', fontsize=8, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='blue', alpha=0.9, 
                           edgecolor='white', linewidth=0.5))

    def create_comprehensive_visualization(self, image_name="image_0.png", conf_threshold=0.1, output_filename="single_modality_pole_detection.png"):
        """Create visualization with clean detection labels"""
        
        fig = plt.figure(figsize=(20, 10))
        if 'combinations' in output_filename:
            fig.suptitle('Snow Pole Detection: RGB Combinations with Confidence Scores', 
                     fontsize=16, fontweight='bold')
        else:
            fig.suptitle('Snow Pole Detection: Individual Modalities with Confidence Scores', 
                     fontsize=16, fontweight='bold')
        
        # Create 2x3 grid for 5 modalities
        rows, cols = 2, 3
        plot_idx = 0
        
        total_detections = 0
        
        # Process individual modalities only 
        for modality in ['signal', 'reflec', 'nearir', 'range', 'combined_color']:
            if plot_idx >= rows * cols:
                break
                
            modality_image = self.load_modality_image(modality, image_name)
            
            if modality_image is not None:
                ax = plt.subplot(rows, cols, plot_idx + 1)
                
                # Show as grayscale for single modalities
                if modality != 'combined_color':
                    ax.imshow(modality_image, cmap='gray')
                else:
                    ax.imshow(modality_image)
                
                # Run detection
                model_key = f'single_{modality}'
                detections, inf_time = self.run_inference(model_key, modality_image, conf_threshold)
                total_detections += len(detections)
                
                # Draw clean detection boxes
                h, w = modality_image.shape[:2] if modality != 'combined_color' else modality_image.shape[:2]
                self.draw_detections(ax, detections, h, w, modality)
                
                # Clean title
                title = modality.upper()
                if 'combinations' in output_filename:
                    if modality == 'signal':
                        title = 'Combination 1'
                    elif modality == 'reflec':
                        title = 'Combination 3'
                    elif modality == 'nearir':
                        title = 'Combination 4'
                    elif modality == 'range':
                        title = 'Combination 5'
                    elif modality == 'combined_color':
                        title = 'Combined Color'
                
                ax.set_title(f'{title}\n{len(detections)} detections', 
                           fontsize=12, fontweight='bold')
                ax.axis('off')
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        output_path = output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.show()


if __name__ == '__main__':
    versions = {
        "8": {
            'signal': {'color': 'green', 'weight': 'train_signal_8N/weights/best.pt'},
            'reflec': {'color': 'purple', 'weight': 'train_reflec_8n/weights/best.pt'}, 
            'nearir': {'color': 'red', 'weight': 'train_narir_8n/weights/best.pt'},
            'range': {'color': 'blue', 'weight': 'train_range_8N/weights/best.pt'},
            'combined_color': {'color': 'orange', 'weight': 'train_combcolor_8n/weights/best.pt'}
        }
    }

    # First run individual modalities
    for version, config in versions.items():
        print(f"----- Running individual modalities for version {version} -----")
        detector = WorkingMultiSpectralDetector(base_dir="data", version_config=config)
        output_file = f"single_modality_pole_detection_v{version}.png"
        detector.create_comprehensive_visualization("image_0.png", conf_threshold=0.05, output_filename=output_file)