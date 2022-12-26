import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

# 定義超參數
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# 下載 MNIST 資料集並加載至記憶體
train_dataset = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size, shuffle=True)

# 定義 CNN 模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = ConvNet()

# 定義損失函數和
# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 開始訓練
for epoch in range(num_epochs):
    for data, target in train_dataset:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# 評估模型效能
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_dataset:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print('Accuracy of the model on the test data: {} %'.format(100 * correct / total))

# 保存模型
torch.save(model.state_dict(), 'model.ckpt')

import onnx
import torch.onnx

# 載入保存的模型
model = ConvNet()
model.load_state_dict(torch.load('model.ckpt'))
model.eval()

# 將模型轉換為 ONNX 格式
dummy_input = torch.randn(1, 1, 28, 28, device='cpu')
torch.onnx.export(model, dummy_input, "model.onnx", verbose=True)