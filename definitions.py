import csv
from io import TextIOWrapper
from zipfile import ZipFile

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ConvNet import ConvNet

class ImageClassificationDataset(Dataset):
    """Load labeled images directly from the downloaded ZIP archive."""

    def __init__(self, archive_path, records, transform):
        self.archive_path = archive_path
        self.records = records
        self.transform = transform

    def __len__(self):
        # DataLoader uses this to determine the dataset size and batch count.
        return len(self.records)

    def __getitem__(self, index):
        # Each record contains the ZIP member path and its numeric class label.
        image_path, label = self.records[index]
        with ZipFile(self.archive_path) as archive:
            with archive.open(image_path) as image_file:
                with Image.open(image_file) as image:
                    image = image.convert("RGB")

        # Return a transformed image tensor and an integer target for CrossEntropyLoss.
        inputs = self.transform(image)
        return inputs, torch.tensor(label, dtype=torch.long)


def read_split(archive, split):
    """Read image filenames and class labels from a split's CSV file."""
    split_path = "SortScrews/" if split == "train" else f"SortScrews/{split}/"
    with archive.open(f"{split_path}labels.csv") as csv_file:
        rows = csv.DictReader(TextIOWrapper(csv_file, encoding="utf-8"))
        return [
            (f"{split_path}images/{row['filename']}", int(row["class"]))
            for row in rows
        ]


def stratified_train_test_split(records, test_fraction=0.1, seed=42):
    """Split each class independently so train and test retain class balance."""
    generator = torch.Generator().manual_seed(seed)
    training_records = []
    test_records = []

    for label in sorted({label for _, label in records}):
        class_records = [record for record in records if record[1] == label]
        order = torch.randperm(len(class_records), generator=generator).tolist()
        test_size = max(1, int(len(class_records) * test_fraction))
        test_records.extend(class_records[index] for index in order[:test_size])
        training_records.extend(class_records[index] for index in order[test_size:])

    return training_records, test_records


def evaluate(model, data_loader, criterion, device):
    """Measure average loss and accuracy without updating model parameters."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient tracking because evaluation does not perform backpropagation.
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total