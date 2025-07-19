import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

# 下载训练集
training_data = datasets.MNIST(
    root="data",             # 数据集存放的路径
    train=True,              # 是否为训练集
    download=True,           # 是否下载
    transform=ToTensor(),    # 数据转换
)

# 下载测试集
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True, 
    transform=ToTensor(),
)

# 显示样本
def show_samples(dataset, model, n_samples=10):
    fig, axes = plt.subplots(1, n_samples, figsize=(15,3))
    indices = np.random.choice(len(dataset), n_samples)
    
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device))
            pred_label = pred.argmax().item()
        
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title(f"True: {label}\nPred: {pred_label}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# 图片打包
train_dataloader = DataLoader(training_data, batch_size=64)    #训练集
test_dataloader = DataLoader(test_data, batch_size=64)         #测试集

for x,y in train_dataloader:
    print(f"shape of x[N,C,H,W]:{x.shape}")    #图像形状
    print(f"shape of y:{y.shape,y.dtype}")     #标签形状和数据类型
    break

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# 构建CNN神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.conv3 = nn.Conv2d(32, 32, 5, 1, 2)
        self.conv4 = nn.Conv2d(32, 64, 5, 1, 2)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.out = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
    
model = CNN().to(device)
print(model)

def train(dataloader, model, loss_fn, optimizer):
    batch_size_num = 1
    for x,y in dataloader:
        x,y = x.to(device),y.to(device)
        pred = model(x)    # 前向传播
        loss = loss_fn(pred,y)

        optimizer.zero_grad()    # 梯度清零
        loss.backward()    # 反向传播
        optimizer.step()    # 更新参数

        loss_value = loss.item()    # 损失值
        if batch_size_num % 100 == 0:    # 每100个batch输出一次
            print(f"loss:{loss_value:>7f}    [number:{batch_size_num}]")
        batch_size_num += 1

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)    # 测试集大小
    num_batches = len(dataloader)    # 测试集batch数量
    model.eval()    # 设置为评估模式
    test_loss,correct = 0,0    # 损失值和正确数
    with torch.no_grad():    # 不计算梯度
        for x,y in dataloader:
            x,y = x.to(device),y.to(device)
            pred = model.forward(x)    # 前向传播
            test_loss += loss_fn(pred,y).item()    #损 失值
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()    # 正确数
    test_loss /= num_batches
    correct /= size
    print(f"\nTest Error:\n")
    print(f"Accuracy: {(100*correct)}%")
    print(f"Avg loss: {test_loss}")
    show_samples(test_data, model, n_samples=10)
    return test_loss, correct

loss_fn = nn.CrossEntropyLoss()    # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)    # 定义优化器

if __name__ == "__main__":
    print(len(training_data))    # 训练集大小
    print(len(test_data))        # 测试集大小

    epochs = 8
    for i in range(epochs):
        print(f"\nEpoch {i+1}")
        train(train_dataloader, model, loss_fn, optimizer)

    test(test_dataloader, model, loss_fn)
