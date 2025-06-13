1. YOLOV7 tiny since that worked previously as well: python train.py --weights yolov7-tiny.pt --data dataset.yaml --epochs 200 --batch-size 2 --device 0 --img-size 1024 --name yolov7-tiny-perm1
      
the data is permu1 from all permu relative path, check teh dataset yaml

if you wanna resume training, do this

python train.py --weights "C:\Users\Muhammad.Rafiq\OneDrive - McDermott, Inc\Documents\yolov7\runs\train\yolov7-tiny-perm5\weights\last.pt" --data dataset.yaml --epochs 200 --batch-size 8 --device 0 --img-size 1024 --name yolov7-tiny-perm5 --resume

yolo train model=yolov8n.pt data=dataset.yaml epochs=250 imgsz=1024 device=0 batch=8 name=v8n_perm6     

 yolo train model=yolov9t.pt data=dataset.yaml epochs=300 imgsz=1024 device=0 batch=8 name=v8n_perm1

 all the 42 or 32 are actually 9t... in the saved file

yolo train model=yolov10n.pt data=dataset.yaml epochs=300 imgsz=1024 device=0 batch=8 name=v10N__perm6 

 yolo train model=yolo11n.pt data=dataset.yaml epochs=300 imgsz=1024 device=0 batch=8 name=v11N__perm1 

# cite papers, yt video for architecture etc...

# TEST COMMANDS

python test.py --data dataset.yaml --img 1024 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights runs\train\yolov7-tiny-perm5\weights\best.pt --name test_yolov7_permutation5      

## testing for ultralytics versions of yolo
yolo val model="C:\Users\Muhammad.Rafiq\OneDrive - McDermott, Inc\Documents\yolov5\runs\detect\v9t_perm5\weights\best.pt" data=dataset.yaml imgsz=1024 batch=8 conf=0.001 iou=0.65 device=0 name=test_yolov9t_permutation5  

python val.py --data dataset.yaml --img 1024 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights runs\train\yolov5s_perm14\weights\best.pt --name test_yolov5s_p1   



val: New cache created: C:\Users\Muhammad.Rafiq\OneDrive - McDermott, Inc\Documents\yolov7\ultralytics\ultralytics\data\labels\test.cache
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 25/25 [00:02<00:00, 10.71it/s]
                   all        197        395      0.889      0.829      0.915      0.451
Speed: 0.5ms preprocess, 3.7ms inference, 0.0ms loss, 2.0ms postprocess per image
Results saved to C:\Users\Muhammad.Rafiq\OneDrive - McDermott, Inc\Documents\yolov5\runs\detect\test_yolov8n_combined_color

