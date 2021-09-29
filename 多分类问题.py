from torch.nn import functional as F
import torch
from torch import optim
from torch import nn
import torchvision.transforms

batch_size = 200
epochs = 10
learning_rate = 1e-3


def forward(x):
    x = x @ w1.t() + b1
    x = F.relu(x)
    x = x @ w2.t() + b2
    x = F.relu(x)
    x = x @ w3.t() + b3
    x = F.relu(x)
    return x


##加载数据集
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("../data/", train=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True
)

w1, b1 = torch.randn(200, 784, requires_grad=True), \
         torch.zeros(200, 200, requires_grad=True)
w2, b2 = torch.randn(200, 200, requires_grad=True), \
         torch.zeros(200, requires_grad=True)
w3, b3 = torch.randn(10, 200, requires_grad=True), \
         torch.zeros(10, requires_grad=True)

##初始化
torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)

optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
criterion = nn.CrossEntropyLoss()  ##F.cross_entropy()

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        logits = forward(data)
        loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print("Train Epoch:{} [{}/{}({:.0f}%)]\tLoss:{:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()
            ))

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        logits = forward(data)
        test_loss += criterion(logits, target).item()
        ##显示精度的区别，item()返回的是一个浮点型数据，在求loss或者accuracy时，一般使用item()，而不是直接取。
        pred = logits.data.max(1)[1]#------------------------------------------------------------------------->①
        correct += pred.eq(target.data).sum()----------------------------------------------------------------->②
    test_loss /= len(test_loader.dataset)
    print("\nTest set:Average loss:{:.4f},Accuracy:{}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))
    
'''最后运行报错：E:\Pycharm\venv\lib\site-packages\torchvision\datasets\mnist.py:498: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ..\torch\csrc\utils\tensor_numpy.cpp:180.)
  return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)'''
