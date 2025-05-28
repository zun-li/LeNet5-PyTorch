from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


def get_train_val_loaders():
    """获取MNIST数据集的训练集和验证集数据加载器(DataLoader)
    
    完成以下功能:
    1. 下载MNIST数据集(如不存在)
    2. 对图像进行预处理: 调整尺寸为32x32并转为Tensor
    3. 将数据集划分为训练集(80%)和验证集(20%)
    4. 创建对应的DataLoader对象
    
    Returns:
        tuple: 包含两个元素的元组，按顺序返回:
            - train_loader (DataLoader): 训练集数据加载器
            - val_loader (DataLoader): 验证集数据加载器
            
    Note:
        - 数据集会自动下载到"./data"目录下
        - 图像尺寸调整为32x32是为了匹配LeNet-5的原始输入尺寸
        - 训练集DataLoader会打乱数据顺序(shuffle=True)
        - 使用2个工作进程(num_workers=2)加速数据加载
    """
    
    # 加载MNIST数据集，调整尺寸为32x32并转为Tensor
    mnist_dataset = MNIST(
        root="./data",  # 数据集存储路径
        train=True,     # 加载训练集(而非测试集)
        transform=transforms.Compose(
            [
                transforms.Resize(size=32),  # 调整图像尺寸为32x32
                transforms.ToTensor()        # 转为Tensor并归一化到[0,1]
            ]
        ),
        download=True  # 如果数据集不存在则自动下载
    )

    # 随机划分训练集(80%)和验证集(20%)
    train_set, val_set = random_split(
        mnist_dataset, 
        lengths=[
            round(0.8 * len(mnist_dataset)),  # 训练集样本数(80%)
            round(0.2 * len(mnist_dataset))   # 验证集样本数(20%)
        ]
    )

    # 创建训练数据加载器
    train_loader = DataLoader(
        dataset=train_set,  # 训练数据集
        batch_size=32,      # 批量大小
        shuffle=True,       # 每个epoch打乱数据顺序
        num_workers=2       # 使用2个子进程加载数据
    )
    
    # 创建验证数据加载器
    val_loader = DataLoader(
        dataset=val_set,    # 验证数据集
        batch_size=32,      # 批量大小(与训练集一致)
        shuffle=True,       # 打乱顺序(虽然通常验证集不需要)
        num_workers=2       # 使用2个子进程加载数据
    )

    return train_loader, val_loader


def get_test_loader():
    """获取MNIST数据集的测试集数据加载器(DataLoader)
    
    完成以下功能:
    1. 下载MNIST数据集(如不存在)
    2. 对图像进行预处理: 调整尺寸为32x32并转为Tensor
    4. 创建对应的DataLoader对象
    
    Returns:
        test_loader (DataLoader): 训练集数据加载器
            
    Note:
        - 数据集会自动下载到"./data"目录下
        - 图像尺寸调整为32x32是为了匹配LeNet-5的原始输入尺寸
        - 训练集DataLoader会打乱数据顺序(shuffle=True)
        - 使用2个工作进程(num_workers=2)加速数据加载
    """

    # 加载MNIST数据集，调整尺寸为32x32并转为Tensor
    test_data = MNIST(
        root="./data",
        train=False,
        transform=transforms.Compose(
            [transforms.Resize(size=32), transforms.ToTensor()]
        ),
        download=True,
    )

    # 创建测试数据加载器
    test_dataloader = DataLoader(
        dataset=test_data, batch_size=64, shuffle=True, num_workers=0
    )
    return test_dataloader