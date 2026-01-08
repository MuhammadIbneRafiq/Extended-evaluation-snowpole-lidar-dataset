# YOLO Training & Validation Commands

## Training (YOLOv7)
```bash
# Fresh training (perm1)
python train.py --weights yolov7-tiny.pt --data dataset.yaml --epochs 200 --batch-size 2 --device 0 --img-size 1024 --name yolov7-tiny-perm1

# Resume training (perm5)
python train.py --weights "C:\Users\Muhammad.Rafiq\OneDrive - McDermott, Inc\Documents\yolov7\runs\train\yolov7-tiny-perm5\weights\last.pt" --data dataset.yaml --epochs 200 --batch-size 8 --device 0 --img-size 1024 --name yolov7-tiny-perm5 --resume
```

> Data: permu1 from all permu (relative path) — check `dataset.yaml`.

## Training (Ultralytics v8–v11)
```bash
# v8
yolo train model=yolov8n.pt data=dataset.yaml epochs=250 imgsz=1024 device=0 batch=8 name=v8n_perm6

# v9 (note: any "42" or "32" saved names are actually 9t)
yolo train model=yolov9t.pt data=dataset.yaml epochs=300 imgsz=1024 device=0 batch=8 name=v8n_perm1

# v10
yolo train model=yolov10n.pt data=dataset.yaml epochs=300 imgsz=1024 device=0 batch=8 name=v10N__perm6

# v11
yolo train model=yolo11n.pt data=dataset.yaml epochs=300 imgsz=1024 device=0 batch=8 name=v11N__perm1
```

## Test / Validation
```bash
# YOLOv7 test
python test.py --data dataset.yaml --img 1024 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights runs\train\yolov7-tiny-perm5\weights\best.pt --name test_yolov7_permutation5

# Ultralytics val (v9t weights)
yolo val model="C:\Users\Muhammad.Rafiq\OneDrive - McDermott, Inc\Documents\yolov5\runs\detect\v9t_perm5\weights\best.pt" data=dataset.yaml imgsz=1024 batch=8 conf=0.001 iou=0.65 device=0 name=test_yolov9t_permutation5

# YOLOv5s val (perm14)
python val.py --data dataset.yaml --img 1024 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights runs\train\yolov5s_perm14\weights\best.pt --name test_yolov5s_p1
```

## Example Validation Output (for reference)
```
val: New cache created: C:\Users\Muhammad.Rafiq\OneDrive - McDermott, Inc\Documents\yolov7\ultralytics\ultralytics\data\labels\test.cache
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 25/25 [00:02<00:00, 10.71it/s]
                   all        197        395      0.889      0.829      0.915      0.451
Speed: 0.5ms preprocess, 3.7ms inference, 0.0ms loss, 2.0ms postprocess per image
Results saved to C:\Users\Muhammad.Rafiq\OneDrive - McDermott, Inc\Documents\yolov5\runs\detect\test_yolov8n_combined_color
```
