import time

import torch
import torch.nn as nn

from data import get_train_val_loaders
from model import LeNet5
from plot import plot_training_metrics


def train(model, train_dataloader, val_dataloader, num_epochs):
    """训练并验证模型

    Args:
        model: 要训练的模型
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        num_epochs: 训练轮数

    Returns:
        dict: 包含训练过程的字典，记录损失和准确率
    """

    # 初始化超参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    # 初始化记录变量
    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    since = time.time()

    # 训练循环
    for epoch in range(num_epochs):
        # 初始化每轮训练的记录变量
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0

        # 训练阶段
        model.train()
        for b_x, b_y in train_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 前向传播
            output = model(b_x)
            pred = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录mini-batch的训练损失和准确率
            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pred == b_y.data)
            train_num += b_x.size(0)

        # 验证阶段
        model.eval()
        with torch.no_grad():
            for b_x, b_y in val_dataloader:
                b_x = b_x.to(device)
                b_y = b_y.to(device)
                # 前向传播
                output = model(b_x)
                pred = torch.argmax(output, dim=1)
                loss = criterion(output, b_y)
                # 记录mini-batch的验证损失和准确率
                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pred == b_y.data)
                val_num += b_x.size(0)

        # 记录epoch的训练损失和准确率
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)

        # 记录epoch的训练损失和准确率
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print("-" * 50)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"train loss: {train_loss_all[-1]:.4f} | val loss: {val_loss_all[-1]:.4f}")
        print(f"train acc : {train_acc_all[-1]:.4f} | val acc : {val_acc_all[-1]:.4f}")

        # 保存最佳模型
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            torch.save(model.state_dict(), "./best_model.pth")

        # 计算训练和验证的耗时
        time_use = time.time() - since
        print(f"epoch time: {time_use//60:.0f}m{time_use%60:.0f}s")

    # 记录训练中的训练轮次、训练损失、验证损失、训练精度、验证精度
    train_record = {
        "epoch": range(num_epochs),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all,
    }

    return train_record


if __name__ == "__main__":
    # 初始化模型
    LeNet = LeNet5()

    # 加载数据
    train_data, val_data = get_train_val_loaders()

    # 训练数据
    train_process = train(LeNet, train_data, val_data, num_epochs=20)

    # 绘制损失和精度曲线
    plot_training_metrics(train_process)
