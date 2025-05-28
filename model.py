import torch
from torch import nn
from torchsummary import summary


class LeNet5(nn.Module):
    """LeNet-5 卷积神经网络模型。

    原始论文: Gradient-Based Learning Applied to Document Recognition (Yann LeCun, 1998)

    Args:
        in_channels (int): 输入通道数 (默认为1，灰度图像).
        num_classes (int): 输出类别数 (默认为10，对应MNIST数据集).

    网络结构:
        - 输入: 1x32x32 灰度图像
        - C1: 卷积层 (6个5x5卷积核)
        - S2: 平均池化层 (2x2窗口，步长2)
        - C3: 卷积层 (16个5x5卷积核)
        - S4: 平均池化层 (2x2窗口，步长2)
        - 展平层
        - F5: 全连接层 (400->120)
        - F6: 全连接层 (120->84)
        - F7: 输出层 (84->10)

    注: 本实现使用Sigmoid激活函数替代原始论文中的tanh函数。
    """
    
    def __init__(self, in_channels=1, num_classes=10):
        super(LeNet5, self).__init__()

        # 特征提取层
        self.c1 = nn.Conv2d(in_channels, out_channels=6, kernel_size=5)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.flatten = nn.Flatten()
        self.f5 = nn.Linear(400, 120)
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, num_classes)

        # 激活函数使用Sigmoid
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (batch_size, in_channels, 32, 32)
        x = self.sig(self.c1(x))    # C1: [B, 6, 28, 28]
        x = self.s2(x)              # S2: [B, 6, 14, 14]
        x = self.sig(self.c3(x))    # C3: [B, 16, 10, 10]
        x = self.s4(x)              # S4: [B, 16, 5, 5]
        
        x = self.flatten(x)         # Flatten to [B, 16*5*5] 
        x = self.f5(x)              # F5: [B, 120]
        x = self.f6(x)              # F6: [B, 84]
        x = self.f7(x)              # Output: [B, num_classes]
        return x


if __name__ == "__main__":
    # 有CUDA用CUDA，没有CUDA则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化LeNet-5模型并移动到指定设备（GPU/CPU）
    model = LeNet5().to(device)
    
    # 打印模型结构信息，(1, 32, 32)表示单通道32x32图像（MNIST标准尺寸）
    print(summary(model, (1, 32, 32)))
