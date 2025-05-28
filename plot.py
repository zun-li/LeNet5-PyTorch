import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def visualize_image_batch(images, labels, class_names, rows=4, cols=16):
    """可视化一个批次的图像数据及其标签

    Args:
        images (numpy.ndarray): 图像数据数组，形状应为[N, H, W]
        labels (numpy.ndarray): 对应标签数组，形状为[N]
        class_names (list): 类别名称列表，索引对应标签值
        rows (int): 子图行数，默认4
        cols (int): 子图列数，默认16

    Note:
        - 总图像数应为rows*cols
        - 图像会自动调整为灰度显示
    """

    plt.figure(figsize=(12, 5))
    for i in range(len(labels)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(class_names[labels[i]], size=6)
        plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
    plt.show()


def plot_training_metrics(train_record):
    """绘制训练过程中的损失和准确率曲线

    Args:
        train_history (dict): 包含训练过程的字典，应有以下键：
            - 'epoch': 迭代轮次列表
            - 'train_loss_all': 训练损失列表
            - 'val_loss_all': 验证损失列表
            - 'train_acc_all': 训练准确率列表
            - 'val_acc_all': 验证准确率列表

    Output:
        显示包含两个子图的图像：
        - 左图: 训练和验证损失曲线
        - 右图: 训练和验证准确率曲线
    """

    plt.figure(figsize=(12, 4))

    # 左图绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(
        train_record["epoch"], train_record["train_loss_all"], "ro-", label="Train loss"
    )
    plt.plot(
        train_record["epoch"], train_record["val_loss_all"], "bs-", label="Val loss"
    )
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    # 右图绘制准确曲线
    plt.subplot(1, 2, 2)
    plt.plot(
        train_record["epoch"], train_record["train_acc_all"], "ro-", label="Train acc"
    )
    plt.plot(train_record["epoch"], train_record["val_acc_all"], "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig('metrics.png', dpi=300)
    plt.show()


def visualize_pred_batch(test_loader, model):
    """可视化模型预测结果

    在测试集上运行模型并可视化预测结果，错误预测会标红显示

    Args:
        test_loader (DataLoader): 测试数据加载器
        model (nn.Module): 训练好的模型

    Note:
        - 每次只显示一个batch的预测结果
        - 正确预测显示黑色，错误预测显示红色
        - 每张图片上方显示模型输出和真实标签
        - 顶部显示当前batch的准确率
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # 设置模型为评估模式

    with torch.no_grad():  # 禁用梯度计算
        for b_x, b_y in test_loader:
            # 将数据移至设备
            b_x, b_y = b_x.to(device), b_y.to(device)

            # 获取模型预测
            output = model(b_x)
            pred = torch.argmax(output, dim=1)  # 取概率最高的类别

            # 创建可视化图像
            plt.rcParams["font.sans-serif"] = ["Hei"]
            plt.rcParams["axes.unicode_minus"] = False

            plt.figure(figsize=(12, 5))
            for i in range(len(b_x)):
                plt.subplot(4, 16, i + 1)  # 4行16列布局
                plt.imshow(b_x.squeeze()[i], cmap="gray")  # 显示灰度图像

                # 错误预测标红，正确标黑
                c = "r" if pred[i] != b_y[i] else "k"
                plt.title(f"output:{pred[i]}\ntarget:{b_y[i]}", size=6, color=c)
                plt.axis("off")

            # 计算并显示batch准确率
            batch_num = len(b_x)
            batch_corrects = torch.sum(pred == b_y)
            accuracy = batch_corrects / batch_num * 100
            # 在标题上写出准确率
            plt.suptitle(f"该批次数据上的准确率: {accuracy:.1f}%")
            plt.subplots_adjust(wspace=0.1)
            plt.savefig("predict.png")
            plt.show()

            break  # 只显示第一个batch


if __name__ == "__main__":
    # 加载MNIST训练数据集
    mnist_dataset = MNIST(
        root="./data",
        train=True,
        transform=transforms.Compose(
            [transforms.Resize(size=32), transforms.ToTensor()]
        ),
        download=True,
    )

    # 创建DataLoader数据加载器，实现数据集的分批处理
    train_loader = DataLoader(
        dataset=mnist_dataset, batch_size=64, shuffle=True, num_workers=0
    )

    # 从数据加载器中获取一个批次的数据
    for b_x, b_y in train_loader:
        print(b_x.shape)
        print(b_y.shape)
        break

    # 去除单通道维度，并转为numpy数组 [64, 1, 32, 32]->[64, 32, 32]
    b_x = b_x.squeeze().numpy()
    b_y = b_y.numpy()

    # 获取类别名称，并打印
    class_label = mnist_dataset.classes
    print(class_label)

    # 绘制批次中的所有图像（共64张）
    visualize_image_batch(images=b_x, labels=b_y, class_names=class_label)
