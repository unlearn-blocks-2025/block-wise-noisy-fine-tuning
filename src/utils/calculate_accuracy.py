# Comments are in English as requested.
import os
import json
import time
from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader
from src.training.training_base import choose_device


def evaluate(model, dataloader, device="cpu"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total

    model.train()
    return accuracy


def evaluate_and_save_metrics(
    model: torch.nn.Module,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    test_loader: DataLoader,
    modelname: str,
) -> str:

    device = choose_device()
    # Evaluate
    retain_stats = evaluate(model, retain_loader, device)
    forget_stats = evaluate(model, forget_loader, device)
    test_stats = evaluate(model, test_loader, device)

    # Build payload
    ts = int(time.time())
    accuracy = {
        "modelname": modelname,
        "timestamp_unix": ts,
        "metrics": {
            "retain": retain_stats,
            "forget": forget_stats,
            "test": test_stats,
        },
    }

    filepath = f"experiments_results/final_accuracy/{modelname}.json"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(accuracy, f, ensure_ascii=False, indent=2)

    return filepath
