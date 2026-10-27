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
from definitions import ImageClassificationDataset
from definitions import read_split
from definitions import stratified_train_test_split
from definitions import evaluate

import matplotlib.pyplot as plt


def main():
    device = torch.device("cpu")
    # Match the resize, crop, and channel normalization expected by pretrained ResNet-18.
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    # Download the dataset archive, using Hugging Face's local cache when available.
    archive_path = hf_hub_download(
        repo_id="ProjectNeura/SortScrews",
        filename="SortScrews.zip",
        repo_type="dataset",
    )
    with ZipFile(archive_path) as archive:
        train_records = read_split(archive, "train")
        validation_records = read_split(archive, "validation")

    # Keep the held-out test subset class-balanced and repeatable.
    training_records, test_records = stratified_train_test_split(train_records)

    # Determine the classifier output width from the labels in the available splits.
    labels_in_data = [
        label
        for records in (train_records, validation_records)
        for _, label in records
    ]
    num_classes = max(labels_in_data) + 1
    model = ConvNet(num_classes=num_classes, pretrained=True).to(device)

    # Start by training only the task-specific layers; the dataset is small,
    # so keep pretrained ResNet features and their BatchNorm statistics fixed.
    for parameter in model.feature_extractor.parameters():
        parameter.requires_grad = False

    # Wrap each record collection so the loaders retrieve image tensors, not paths.
    training_dataset = ImageClassificationDataset(
        archive_path, training_records, transform
    )
    validation_dataset = ImageClassificationDataset(
        archive_path, validation_records, transform
    )
    test_dataset = ImageClassificationDataset(archive_path, test_records, transform)

    train_loader = DataLoader(training_dataset, batch_size=16, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Cross-entropy expects raw class logits and integer class IDs.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.001,
        weight_decay=0.0001,
    )
    epochs = 60
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min", 
        factor=0.5,
        patience=5,
    ) # for every 5 epochs, reduce the LR by a factor of 0.5 if loss has not decreased. 

    x = []
    y = []
    best_validation_loss = float("inf")
    best_model_state = None
    for epoch in range(epochs):
        model.train()
        # model.train() enables training mode recursively, so keep the frozen
        # backbone's BatchNorm layers in inference mode.
        model.feature_extractor.eval()
        running_loss = 0.0
        training_correct = 0
        training_total = 0
        # Train on complete batches: predict, compute loss, backpropagate, then step.
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            training_correct += (outputs.argmax(dim=1) == labels).sum().item()
            training_total += labels.size(0)

        # Check generalization on the official validation split after each epoch.
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, criterion, device
        )
        scheduler.step(validation_loss)
        x.append(epoch)
        y.append(validation_loss)
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train loss {running_loss / len(train_loader):.4f}, "
            f"train accuracy {training_correct / training_total:.2%}, "
            f"validation loss {validation_loss:.4f}, "
            f"validation accuracy {validation_accuracy:.2%}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }

    # Report final performance on the held-out test subset and save learned weights.
    if best_model_state is None:
        raise RuntimeError("Training completed without recording a best model.")
    model.load_state_dict(best_model_state)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.2%}")
    torch.save(model.state_dict(), "model.pth")
    fig, ax = plt.subplots()
    ax.scatter(x,y)

    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel('Val Loss', fontsize=15)
    ax.set_title('Val Loss Over Time')

    ax.grid(True)
    fig.tight_layout()

    plt.show()

    


if __name__ == "__main__":
    # Avoid downloading data or training when this module is imported elsewhere.
    main()
