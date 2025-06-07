import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from model import LeNet


def get_train_val_loader():
    dataset = FashionMNIST(
        root="data",
        train=True,
        transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ]),
        download=True,
    )

    train_set, val_set = random_split(
        dataset, lengths=[round(len(dataset) * 0.8), round(len(dataset) * 0.2)]
    )

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)

    val_loader = DataLoader(val_set, batch_size=64, shuffle=True, num_workers=2)

    return train_loader, val_loader


def train(model, train_loader, val_loader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.to(device)

    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []

    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        train_corrects = 0
        val_corrects = 0
        train_num = 0
        val_num = 0
        since = time.time()

        model.train()
        for b_x, b_y in train_loader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)
            pred = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_corrects += torch.sum(pred == b_y)
            train_num += len(b_x)

        model.eval()
        with torch.no_grad():
            for b_x, b_y in val_loader:
                b_x = b_x.to(device)
                b_y = b_y.to(device)
                output = model(b_x)
                pred = torch.argmax(output, dim=1)
                loss = criterion(output, b_y)
                val_loss += loss.item()
                val_corrects += torch.sum(pred == b_y)
                val_num += len(b_x)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print("-" * 50)
        print(f"Epoch {epoch + 1} / {num_epochs}")
        print(f"Train loss: {train_loss_all[-1]:.4f} | Val loss: {val_loss_all[-1]:.4f}")
        print(f"Train acc : {train_acc_all[-1]:.4f} | Val acc : {val_acc_all[-1]:.4f}")

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            torch.save(model.state_dict(), "models/best_model.pth")

        time_use = time.time() - since
        print(f"Epoch time: {time_use//60:.0f}m{time_use%60:.0f}s")

    train_record = {
        "epoch": range(num_epochs),
        "train_loss_all": train_loss_all,
        "train_acc_all": train_acc_all,
        "val_loss_all": val_loss_all,
        "val_acc_all": val_acc_all,
    }

    return train_record


def plot_learning_curve(train_record):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    x = train_record["epoch"]
    y1 = train_record["train_loss_all"]
    y2 = train_record["val_loss_all"]
    plt.plot(x, y1, "ro-", label="Train Loss")
    plt.plot(x, y2, "bs-", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    y3 = train_record["train_acc_all"]
    y4 = train_record["val_acc_all"]
    plt.plot(x, y3, "ro-", label="Train Acc")
    plt.plot(x, y4, "bs-", label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Curve")
    plt.legend()

    plt.savefig("images/loss_acc_curve.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    Path("models").mkdir(parents=True, exist_ok=True)

    model = LeNet()

    train_loader, val_loader = get_train_val_loader()

    train_record = train(model, train_loader, val_loader, 40)

    plot_learning_curve(train_record)
