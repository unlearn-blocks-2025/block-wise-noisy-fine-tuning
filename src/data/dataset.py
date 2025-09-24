from torchvision import datasets, transforms
from torch.utils.data import Subset, TensorDataset, DataLoader
import random
from torchvision import datasets, transforms
import torch
import copy


def create_train_dataset(
    dataset="mnist", forget_percent=None, forget_class=None, path_to_data=None
):
    # Load the MNIST training dataset

    if dataset == "mnist":
        transform = transforms.ToTensor()
        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    elif dataset == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=train_transform
        )

    else:
        raise ValueError(f"Unknown dataset type='{dataset}'. Use 'cifar10' or 'mnist'.")

    if path_to_data is not None:
        return create_forget_by_path(train_dataset, path_to_data)

    elif forget_percent is not None:
        return create_forget_by_percent_dataset(train_dataset, forget_percent)

    elif forget_class is not None:
        return create_forget_by_class_dataset(train_dataset, forget_class)

    else:
        raise ValueError(f"both forget class and forget percent are None.")


def create_forget_by_percent_dataset(train_dataset, forget_percent, seed=42):

    random.seed(seed)

    # Total number of training samples
    num_samples = len(train_dataset)

    # Determine the number of samples to forget and retain
    num_forget = int(forget_percent / 100.0 * num_samples)

    # Shuffle all indices and split into forget and retain sets
    all_indices = list(range(num_samples))
    random.shuffle(all_indices)
    forget_indices = all_indices[:num_forget]
    retain_indices = all_indices[num_forget:]

    # Create subsets for forget and retain
    forget_dataset = Subset(train_dataset, forget_indices)
    retain_dataset = Subset(train_dataset, retain_indices)

    return forget_dataset, retain_dataset, train_dataset


def create_forget_by_class_dataset(train_dataset, forget_class):
    """
    Split dataset into forget set (all samples of given class)
    and retain set (all other samples).

    Args:
        train_dataset: full dataset (MNIST, CIFAR, etc.)
        forget_class: int, label of the class to forget
    """
    forget_indices = [i for i, (_, y) in enumerate(train_dataset) if y == forget_class]
    retain_indices = [i for i, (_, y) in enumerate(train_dataset) if y != forget_class]

    forget_dataset = Subset(train_dataset, forget_indices)
    retain_dataset = Subset(train_dataset, retain_indices)

    return forget_dataset, retain_dataset, train_dataset


def create_forget_by_path(train_dataset, path_to_data):

    retain_path = path_to_data["retain"]
    forget_path = path_to_data["forget"]

    forget = torch.load(forget_path)
    forget_dataset = copy.deepcopy(train_dataset)
    forget_dataset.data = forget["data"].cpu().numpy()
    forget_dataset.targets = forget["targets"].cpu().numpy()

    retain = torch.load(retain_path)
    retain_dataset = copy.deepcopy(train_dataset)
    retain_dataset.data = retain["data"].cpu().numpy()
    retain_dataset.targets = retain["targets"].cpu().numpy()

    print(
        f"datasets of len {len(retain_dataset)} and {len(forget_dataset)} are uploaded"
    )

    return forget_dataset, retain_dataset, train_dataset


def create_test_dataloader(
    dataset="mnist",
    exclude_class=None,
    include_class=None,
    batchsize=1000,
):
    if dataset == "mnist":
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transforms.ToTensor()
        )

    elif dataset == "cifar10":
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=test_transform
        )
    else:
        raise ValueError(f"Unknown dataset type='{dataset}'. Use 'cifar10' or 'mnist'.")

    if exclude_class is not None:
        test_indices = [
            i for i, (_, y) in enumerate(test_dataset) if y != exclude_class
        ]
        test_dataset = Subset(test_dataset, test_indices)

    elif include_class is not None:
        test_indices = [
            i for i, (_, y) in enumerate(test_dataset) if y == include_class
        ]

        test_dataset = Subset(test_dataset, test_indices)

    print(f"test dataset contains {len(test_dataset)} samples")

    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

    return test_loader, test_dataset
