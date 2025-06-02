import torch
import pynvml

def get_free_gpu():
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    free_gpus = []
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_gpus.append((i, mem_info.free))
    pynvml.nvmlShutdown()
    # Sort GPUs by free memory descending
    free_gpus.sort(key=lambda x: x[1], reverse=True)
    return free_gpus[0][0]  # return index of GPU with most free mem

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cpu")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device