import torch

def get_device():
    """
    Returns the most efficient device available: CUDA (Windows/Linux), MPS (Mac), or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def move_to_device(data, device=None):
    """
    Recursively moves tensors in a list, dict, or single tensor to the specified device.
    """
    if device is None:
        device = get_device()
        
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    else:
        return data
