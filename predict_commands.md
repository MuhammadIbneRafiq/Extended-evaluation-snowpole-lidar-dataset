# YOLO Predict Commands for Different Modalities (v8-v11)

## Signal Modality
```bash
# v11
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_signal_11n/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/signal/test/image_0.png" conf=0.01 name=signal_v11_image_0

# v10
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_signal_10n/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/signal/test/image_0.png" conf=0.01 name=signal_v10_image_0

# v9
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_signal_9t/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/signal/test/image_0.png" conf=0.01 name=signal_v9_image_0

# v8
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_signal_8N/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/signal/test/image_0.png" conf=0.01 name=signal_v8_image_0
```

## Reflec Modality
```bash
# v11
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_reflec_11n/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/reflec/test/image_0.png" conf=0.01 name=reflec_v11_image_0

# v10
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_reflec_10N/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/reflec/test/image_0.png" conf=0.01 name=reflec_v10_image_0

# v9
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_reflec_9t/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/reflec/test/image_0.png" conf=0.01 name=reflec_v9_image_0

# v8
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_reflec_8n/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/reflec/test/image_0.png" conf=0.01 name=reflec_v8_image_0
```

## NearIR Modality
```bash
# v11
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_narir_11n/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/nearir/test/image_0.png" conf=0.01 name=nearir_v11_image_0

# v10
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_narir_10n/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/nearir/test/image_0.png" conf=0.01 name=nearir_v10_image_0

# v9
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_narir_9t/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/nearir/test/image_0.png" conf=0.01 name=nearir_v9_image_0

# v8
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_narir_8n/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/nearir/test/image_0.png" conf=0.01 name=nearir_v8_image_0
```

## Range Modality
```bash
# v11
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_range_11n/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/range/test/image_0.png" conf=0.01 name=range_v11_image_0

# v10
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_range_10n/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/range/test/image_0.png" conf=0.01 name=range_v10_image_0

# v9
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_range_9t/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/range/test/image_0.png" conf=0.01 name=range_v9_image_0

# v8
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_range_8N/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/range/test/image_0.png" conf=0.01 name=range_v8_image_0
```

## Combined Color Modality
```bash
# v11
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_combcolor_11n/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/combined_color/test/image_0.png" conf=0.01 name=combined_color_v11_image_0

# v10
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_combcolor_10n/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/combined_color/test/image_0.png" conf=0.01 name=combined_color_v10_image_0

# v9
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_combcolor_9t2/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/combined_color/test/image_0.png" conf=0.01 name=combined_color_v9_image_0

# v8
yolo predict model="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/trained_weights/train_combcolor_8n/weights/best.pt" source="C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m/data/single_modality/SnowPole_Detection_Dataset/combined_color/test/image_0.png" conf=0.01 name=combined_color_v8_image_0
```

Note: 
1. All commands use a confidence threshold of 0.35. Adjust this value if needed to get better detection results.
2. The output files will be saved in runs/detect/[name] where [name] follows the pattern: modality_version_image_name
3. For example, "signal_v11_image_0" will create a directory "runs/detect/signal_v11_image_0" containing the prediction results

Important: These commands use the YOLOv7 weights from the results directory. The version numbers (v8-v11) in the comments are for reference only, as all models are using YOLOv7 weights. 