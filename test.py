import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import FashionMNIST

from model import LeNet


def get_test_loader():
    dataset = FashionMNIST(
        root="data",
        train=False,
        transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ]),
        download=True,
    )

    test_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    return test_loader


def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_corrects = 0
    test_num = 0

    model.eval()
    with torch.no_grad():
        for b_x, b_y in test_loader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)
            pred = torch.argmax(output, dim=1)
            test_corrects += torch.sum(pred == b_y)
            test_num += len(b_x)

    test_acc = test_corrects.double().item() / test_num
    print(f"Test accuracy: {test_acc * 100:.1f}%")


def plot_pred_batch(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class_label = datasets.FashionMNIST.classes

    model.eval()
    with torch.no_grad():
        for b_x, b_y in test_loader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)
            pred = torch.argmax(output, dim=1)

            plt.figure(figsize=(12, 6))
            for i in range(len(b_x)):
                plt.subplot(4, 8, i + 1)
                plt.imshow(b_x.squeeze()[i], cmap="gray")

                predict_label = class_label[pred[i]]
                target_label = class_label[b_y[i]]
                color = "r" if pred[i] != b_y[i] else "k"
                
                plt.title(f"predict:{predict_label}\ntarget:{target_label}", size=6, color=color)
                plt.axis("off")

            batch_num = len(b_x)
            batch_corrects = torch.sum(pred == b_y)
            accuracy = batch_corrects.double().item() / batch_num

            # plt.tight_layout()
            plt.subplots_adjust(hspace=0.4)
            plt.suptitle(f"Batch accuracy: {accuracy * 100:.1f}%")
            plt.savefig("images/batch_acc.png", dpi=300)
            plt.show()

            break


if __name__ == "__main__":
    model = LeNet()

    model.load_state_dict(torch.load("models/best_model.pth"))

    test_loader = get_test_loader()

    test(model, test_loader)

    plot_pred_batch(model, test_loader)
