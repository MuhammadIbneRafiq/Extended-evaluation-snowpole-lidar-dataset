import torch
from ultralytics import YOLO

model = YOLO("yolov9t.pt")           # Step 2: load pretrained RGB weights
model.model.eval()                   # get the underlying nn.Module graph

first_conv = model.model.model[0]    # YOLOv9 stem (Conv → BN → SILU)
old_conv = first_conv.conv
new_conv = torch.nn.Conv2d(
    in_channels=4,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=old_conv.bias is not None,
)

with torch.no_grad():
    new_conv.weight[:, :3] = old_conv.weight            # copy RGB kernels
    new_conv.weight[:, 3:] = old_conv.weight[:, :1] * 0 # or torch.randn_like(...)*1e-3
    if old_conv.bias is not None:
        new_conv.bias = old_conv.bias

first_conv.conv = new_conv            # swap into the module tree
model.model.model[0] = first_conv     # ensure the model graph is updated

model.save("yolov9t_rgba.pt")