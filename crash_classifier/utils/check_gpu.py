import torch, torchvision, platform
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
