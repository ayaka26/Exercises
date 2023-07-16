import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
from typing import Any

#argparser
#モジュールとしてインポートして時に勝手に実行されてしまうと面倒なので関数に入れる
def parse_args()->dict[str,Any]:
    parser = argparse.ArgumentParser("説明文")
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--test_size",type=float,default=0.3)
    args = parser.parse_args()
    return args
args = parse_args()

def get_batch(X, Y,batch_size=16, shuffle=True):
    # GPUに対応するために半精度(float32？)にキャスト
    X = X.astype(np.float32)
    # Xとyをシャッフル
    index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(index)
        X = X[index]
        Y = Y[index]
    #batch_sizeずつ切り出してタプルとして返すループ
    for i in range(0, Y.shape[0], batch_size):
      x = X[i:i+batch_size]
      y = Y[i:i+batch_size]
      x = torch.from_numpy(x)
      y = torch.from_numpy(y)
      yield(x, y)

def main(lr, batch_size, seed, max_epochs, device, test_size):
    
    iris = load_iris(as_frame=True)["frame"]
    train, test = train_test_split(iris, test_size=test_size)
    X_train = train.iloc[:,:4].to_numpy()
    y_train = train.iloc[:,-1].to_numpy()
    X_test = test.iloc[:,:4].to_numpy()
    y_test = test.iloc[:,-1].to_numpy()
    n_labels = len(np.unique(y_train))  # 分類クラスの数 = 3
    y_train = np.eye(n_labels)[y_train]
    y_test = np.eye(n_labels)[y_test]
    
    net = nn.Sequential(
        nn.Linear(4,20),  # 4つの要素で20次元
        nn.ReLU(),
        nn.Linear(20,3),) #20次元で3つ(アヤメの種類)に分類
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    
    net.to(device)

    training_loss = []
    train_acc = []

    for epoch in range(max_epochs):
        net.train()
        train_loss = []
        for batch in get_batch(X_train,y_train):
            optimizer.zero_grad()
            x,y = batch
            x = x.to(device)
            y = y.to(device)
            output = net(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()#更新
            train_loss.append(float(loss))
            accuracy = (torch.argmax(output, dim=1) == torch.argmax(y, dim=1)).sum().numpy() / x.size(0)
            train_acc.append(float(accuracy))
        training_loss.append(np.mean(train_loss))

    test_acc  =[]
    with torch.no_grad():
        for batch in get_batch(X_test, y_test):
            x,y = batch
            x = x.to(device)
            y = y.to(device)
            output = net(x)
            accuracy = (torch.argmax(output, dim=1) == torch.argmax(y, dim=1)).sum().numpy() / x.size(0)
            test_acc.append(float(accuracy))

    train_Accuracy = np.mean(train_acc)
    test_Accuracy = np.mean(test_acc)

    return train_Accuracy , test_Accuracy

if __name__ == "__main__":
    lr = args.learning_rate
    batch_size = args.batch_size
    seed = args.seed
    max_epochs = args.max_epochs
    device = args.device
    test_size = args.test_size

    train_Accuracy,test_Accuracy=main(lr,batch_size,seed,max_epochs,device,test_size)
    print("教師データのACC:",train_Accuracy)
    print("テストデータのACC:",test_Accuracy) 
    
