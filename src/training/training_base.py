import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models.tinymodel import LinearNet
from src.utils.calculate_accuracy import evaluate
from src.utils.utils import save_accuracy_history, save_model
import time


def choose_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"device {device} will be used")
    return device


# can be used for fully trained model and for retrain
def train_base_model(
    full_loader,
    test_loader,
    model=None,
    lr=0.01,
    num_epochs=5,
    momentum=0.9,
    weight_decay=1e-4,
    log_every=100,
    save_model_weights=True,
    modelname="full",
    model_type=LinearNet,
):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    device = choose_device()

    # Initialize model, loss function and optimizer
    if model is None:
        model = model_type(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    if "resnet" in modelname:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )

    # Training loop
    loss_history = []
    accuracy_history = []

    # log every log_every steps
    step = 0

    test_acc = evaluate(model, test_loader, device=device)
    accuracy_history.append((step, test_acc))

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in full_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            step += 1

            if step % log_every == 0:
                loss_history.append((step, loss.item()))
                test_acc = evaluate(model, test_loader, device=device)
                accuracy_history.append((step, test_acc))
                model.train()
                print(f"Step [{step}] - Accuracy: {test_acc:.2f}%")

        if "resnet" in modelname:
            scheduler.step()

        epoch_loss = running_loss / total

        epoch_acc = evaluate(model, test_loader, device=device)
        model.train()

        print(
            f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%"
        )

    accuracy_filepath = f"experiments_results/accuracy/{modelname}_lr_{lr}_epochs_{num_epochs}_{timestamp}.pt"
    save_accuracy_history(accuracy_history, filepath=accuracy_filepath)

    if save_model_weights:
        save_model(
            model, filepath=f"experiments_results/models/{modelname}_{timestamp}.pt"
        )
    return model, accuracy_history
