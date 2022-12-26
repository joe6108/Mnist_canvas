import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

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

