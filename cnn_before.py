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
    parser.add_argument("--max_epochs", type=int, default=5)
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


def load_MNIST(batch=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,)),
        ])

    train_set = torchvision.datasets.MNIST(root="./data",
                                           train=True,
                                           download=True,
                                           transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch,
                                               shuffle=True,
                                               num_workers=2)

    val_set = torchvision.datasets.MNIST(root="./data",
                                         train=False,
                                         download=True,
                                         transform=transform)
    val_loader =torch.utils.data.DataLoader(val_set,
                                            batch_size=batch,
                                            shuffle=True,
                                            num_workers=2)

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
        self.conv1 = nn.Conv2d(1,32,3,1,0)# in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.pool = nn.MaxPool2d(2,2,) # kernel_size, stride
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12*12*64,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        #x = self.dropout1(x)
        x = x.view(-1,12*12*64)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)

        return x # F.log_softmax(x, dim=1)


def main(lr, batch_size, SEED, max_epochs, device):

    #学習結果の保存
    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_acc": []
    }

    #データのロード
    data_loder = load_MNIST(batch=batch_size)
        
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
            accuracy = 0.0

            with torch.no_grad():
                for data,target in data_loder["validation"]:
                    data,target = data.to(device),target.to(device)

                    #順伝搬の計算
                    output = net(data)
                    loss = criterion(output,target)
                    #loss = F.nll_loss(output,target).item()
                    val_loss += F.nll_loss(output,target,reduction='sum').item()
                    predict = output.argmax(dim=1,keepdim=True)
                    accuracy += predict.eq(target.view_as(predict)).sum().item()

            val_loss /= len(data_loder["validation"].dataset)
            accuracy /= len(data_loder["validation"].dataset)

            print("Validation loss: {}, Accuracy: {}\n".format(val_loss,accuracy))
            history["validation_loss"].append(val_loss)
            history["validation_acc"].append(accuracy)


if __name__ == '__main__':
    
    #ハイパーパラメータ各種
    lr =args.learning_rate
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    SEED = args.SEED
    device = args.device

    main(lr, batch_size, SEED, max_epochs, device)

    plt.plot(val_loss, accuracy)
    