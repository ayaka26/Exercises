# packageのimport
from typing import Any, Union, Callable, Type, TypeVar
from tqdm.std import trange,tqdm
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from PIL import Image
import cv2
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse


def parse_args()->dict[str,Any]:
    parser = argparse.ArgumentParser("説明文")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--SEED", type=int, default=2023_06_27)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    return args
args = parse_args()


def set_seed(SEED = args.SEED):
    # numpyに関係する乱数シードの設定
    np.random.seed(SEED)

    # pytorch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
set_seed()


def CIFAR10 (batch=128):
    #CIFAR-10のデータセットのインポート
    # ToTensor：画像のグレースケール化（RGBの0~255を0~1の範囲に正規化）、Normalize：Z値化（RGBの平均と標準偏差を0.5で決め打ちして正規化）
    transform = transforms.Compose([
        transforms.ToTensor(), 
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_set = torchvision.datasets.CIFAR10(root='./data', 
                                             train=True, 
                                             download=True, 
                                             transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=4, 
                                               shuffle=True, 
                                               num_workers=2)
    # テストデータをダウンロード
    val_set = torchvision.datasets.CIFAR10(root='./data', 
                                           train=False, 
                                           download=True, 
                                           transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=True, num_workers=2)

    # classesはよく利用するので別途保持しておく
    classes = train_set.classes

    return {"train":train_loader, "validation":val_loader}


cnn = nn.Sequential(
    nn.Conv2d(1,32, 3,1,0),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(32,64, 3,1,0),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Dropout2d(0.25),
    nn.Flatten(),
    nn.Linear(12*12*64,128),
    nn.ReLU(),
    nn.Linear(128,10)
)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1((x)))
        x = F.relu(self.fc2((x)))
        x = self.fc3(x)
        return x


def main(lr, batch_size, SEED, max_epochs, device):

    #学習結果の保存
    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_acc": []
    }

    #データのロード
    data_loder = CIFAR10(batch=batch_size)
        
    #ネットワーク構造の構築
    net = CNN().to(device)

    #最適化方法の設定
    optimizer = torch.optim.Adam(params=net.parameters(), lr = args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(max_epochs):
        """ 学習部分 """
        loss = None
        train_loss = 0.0
        net.train() #学習モード

        for i,(data,target) in enumerate(data_loder["train"]):
            data,target = data.to(device),target.to(device)

            #勾配の初期化
            optimizer.zero_grad()
            #順伝搬 -> 逆伝搬 -> 最適化
            output = net(data)
            #loss = F.nll_loss(output,target)
            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i % 100 == 99:
                print("Training: {} epoch. {} iteration. Loss:{}".format(epoch+1,i+1,loss.item()))

            train_loss /= len(data_loder["train"])
            print("Training loss (ave.): {}".format(train_loss))
            history["train_loss"].append(train_loss)

            """検証部分"""
            print("\nValidation start")
            net.eval() #検証モード(Validation)
            val_loss = 0.0
            val_acc = 0.0

            with torch.no_grad():
                for data,target in data_loder["validation"]:
                    data,target = data.to(device),target.to(device)

                    #順伝搬の計算
                    output = net(data)
                    loss = criterion(output,target)
                    #loss = F.nll_loss(output,target).item()
                    val_loss += F.nll_loss(output,target,reduction='sum').item()
                    predict = output.argmax(dim=1,keepdim=True)
                    val_acc += predict.eq(target.view_as(predict)).sum().item()

            val_loss /= len(data_loder["validation"].dataset)   # 平均
            val_acc /= len(data_loder["validation"].dataset)    # 平均

            print("Validation loss: {}, Validation Accuracy: {}\n".format(val_loss,val_acc))
            history["validation_loss"].append(val_loss)
            history["validation_acc"].append(val_acc)

            train_loss = history["train_loss"]
            val_loss = history["validation_acc"]
            val_acc = history["validation_acc"]


if __name__ == '__main__':
    
    #ハイパーパラメータ各種
    lr =args.learning_rate
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    SEED = args.SEED
    device = args.device

    main(lr, batch_size, SEED, max_epochs, device)

    plt.plot(range(1, max_epochs+1), train_loss, label='train_loss', c="#1f77b4")
    plt.plot(range(1, max_epochs+1), val_loss, marker="o", lw=0, label='val_loss', c="#1f77b4")
    plt.plot(range(1, max_epochs+1), val_acc, label='val_acc', c="#ff7f0e")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy & Loss")

    plt.legend()
    plt.show()
