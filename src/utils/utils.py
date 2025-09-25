import torch
import os
import numpy as np
import random


def save_accuracy_history(history, filepath="logs/accuracy_history.pt"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(history, filepath)


def load_accuracy_history(filepath="logs/accuracy_history.pt"):
    return torch.load(filepath)


def save_model(model, filepath="models/lora_model.pt"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)


def load_model(model_class, filepath="models/lora_model.pt", device="cpu", **kwargs):
    model = model_class(**kwargs).to(device)
    state = torch.load(filepath, map_location=device)
    model.load_state_dict(state)
    return model


def setup_seed(seed):
    print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def choose_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"device {device} will be used")
    return device
