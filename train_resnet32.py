import argparse
import os
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def download_dataset(url: str, dest: Path):
    """Download a dataset from a URL if not already present."""
    if dest.exists():
        return
    raise RuntimeError(
        "Dataset download requires internet access which is not available in this environment."
    )


def prepare_data(root: Path):
    """Prepare dataset by filtering classes and splitting."""
    # Placeholder implementation. User must manually download and extract the
    # datasets into `root/original` before running this function.

    train_dir = root / "train"
    valid_dir = root / "valid"
    test_dir = root / "test"
    for d in (train_dir, valid_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)
    # The real implementation should copy images from the original dataset and
    # perform class balancing with augmentation to reach 5000 images per class
    # in the training set.
    raise NotImplementedError(
        "Data preparation not implemented. Please provide preprocessed dataset."
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def make_layer(in_planes, planes, blocks, stride=1):
    layers = [BasicBlock(in_planes, planes, stride)]
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes))
    return nn.Sequential(*layers)


class ResNet32(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = make_layer(16, 16, 5)
        self.layer2 = make_layer(16, 32, 5, stride=2)
        self.layer3 = make_layer(32, 64, 5, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_loaders(data_dir: Path, batch_size: int = 32):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = datasets.ImageFolder(data_dir / "train", transform=transform_train)
    valid_set = datasets.ImageFolder(data_dir / "valid", transform=transform_test)
    test_set = datasets.ImageFolder(data_dir / "test", transform=transform_test)

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2),
        len(train_set.classes),
    )


def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    preds = []
    gts = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            preds.extend(predicted.cpu().tolist())
            gts.extend(targets.cpu().tolist())
    acc = correct / len(loader.dataset)
    return running_loss / len(loader.dataset), acc, preds, gts


def main(args):
    data_dir = Path(args.data_dir)
    if args.prepare_data:
        prepare_data(data_dir)

    train_loader, valid_loader, test_loader, num_classes = get_loaders(data_dir, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet32(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, valid_loader, criterion, device)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    test_loss, test_acc, preds, gts = evaluate(model, test_loader, criterion, device)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

    try:
        from sklearn.metrics import confusion_matrix, recall_score, precision_score
        cm = confusion_matrix(gts, preds)
        recall = recall_score(gts, preds, average='macro')
        precision = precision_score(gts, preds, average='macro')
        print("Confusion Matrix:\n", cm)
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Accuracy: {test_acc:.4f}")
    except Exception as e:
        print("Failed to compute metrics:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet32 on plant datasets")
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prepare-data", action="store_true", help="Run data preparation step")
    args = parser.parse_args()
    main(args)
