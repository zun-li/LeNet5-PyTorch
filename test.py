import torch

from data import get_test_loader
from model import LeNet5
from plot import visualize_pred_batch


def test(model, test_dataloader):
    """测试模型性能
    
    Args:
        model: 待测试的模型
        test_dataloader: 测试数据加载器

    Output:
        输出模型在测试集上的准确率
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 测试阶段
    model.eval()
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            output = model(test_data_x)
            pred = torch.argmax(output, dim=1)
            test_corrects += torch.sum(pred == test_data_y)
            test_num += test_data_x.size(0)

    # 计算测试准确率
    test_acc = test_corrects.double().item() / test_num
    print(f"全部测试集上的准确率：{test_acc * 100:.1f}%")


if __name__ == "__main__":
    # 加载模型
    model = LeNet5()
    model.load_state_dict(torch.load("best_model.pth"))

    # 加载测试数据
    test_loader = get_test_loader()

    # 加载模型测试的函数;
    test(model, test_loader)

    # 可视化预测结果
    visualize_pred_batch(test_loader, model)
