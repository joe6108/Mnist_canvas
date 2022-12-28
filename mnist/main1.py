import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.onnx

# 下載並讀取MNIST數據集
train_dataset = datasets.MNIST(root='./data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data', 
                              train=False, 
                              transform=transforms.ToTensor())

# 定義訓練參數
batch_size = 64
learning_rate = 0.001
num_epochs = 20

# 將數據集轉換為dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# 定義網絡
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定義模型
model = ConvNet()

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

torch.set_printoptions(profile="full")
device = torch.device('cpu')
# 定義訓練計數器
total = 0
correct = 0

# 開始訓練
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 將圖像和標籤轉換為張量
        images = images.to(device)
        labels = labels.to(device)

        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向傳播並優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 計算準確度
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # 輸出訓練準確度
    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
          .format(epoch+1, num_epochs, loss.item(), 
                  100 * correct / total))
    
    # 重置訓練計數器
    total = 0
    correct = 0

# 將模型轉換為ONNX格式
dummy_input = torch.randn(1, 1, 28, 28).to(device)
torch.onnx.export(model, dummy_input, "onnx_model.onnx", verbose=True)