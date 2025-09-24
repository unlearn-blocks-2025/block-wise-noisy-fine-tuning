import torch.nn as nn
import torch.nn.functional as F
import torch
from src.models.ResNet import resnet18


class LinearNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BigMNISTMLP(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Architecture: 784 -> 2048 -> 1024 -> 512 -> 256 -> 10
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Forward through a deep MLP
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def get_resnet(num_classes=10):
    model = resnet18(num_classes=num_classes, imagenet=False)
    return model


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
