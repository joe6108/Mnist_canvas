import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

# 定義轉換
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, )),  # Common used parameters for mnist normalization.
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, )),
])

# 加載 MNIST 數據集
train_dataset = datasets.MNIST(
    "data",                        # The path to store data
    train=True,                    # True to get training set, false to get validation set.
    download=True,                 
    transform=train_transform      # Apply data preprocessing defined.
)
test_dataset = datasets.MNIST(
    "data",
    train=False,
    transform=test_transform
)

# 建立數據加載器
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=True
)

# 定義模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.4)
        self.fc1 = nn.Linear(1024, 128)
        self.dropout2 = nn.Dropout2d(0.4)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net().to(device)
optimizer = optim.Adam(model.parameters())

model.train()
best_acc = 0.

for epoch in range(20):
    train_loss = 0
    correct = 0
    print(f'Epoch: {epoch}')
    tepoch = tqdm(train_dataloader, total=int(len(train_dataloader)))
    for data, target in tepoch:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Feed forward
        output = model(data)

        # Accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        # Loss
        loss = F.cross_entropy(output, target)
        loss.backward()

        # Step optimizer
        optimizer.step()

        tepoch.set_postfix(loss=loss.item())

    train_acc = correct / len(train_dataloader.dataset)
    train_loss = loss.item()

    print('Train Epoch: {} \t Loss: {:.4f} Accuracy: {:.4f}'.format(
        str(epoch+1), train_loss, train_acc))
    
        # Put validate function here.

torch.save(model.state_dict(), "model.pt")