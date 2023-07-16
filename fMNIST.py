import math
from datetime import datetime
from typing import Any, Union, Callable, Type, TypeVar
from tqdm.auto import trange,tqdm
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
plt.style.use("bmh")
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import skorch
from skorch import NeuralNet, NeuralNetClassifier, NeuralNetRegressor
from matplotlib.animation import ArtistAnimation
from IPython.display import display, Markdown, HTML
import argparse


def parse_args()->dict[str,Any]:
    parser = argparse.ArgumentParser("説明文")
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--latent_dtm", type=str, default=20)
    args = parser.parse_args()
    return args
args = parse_args()

#ハイパーパラメータ
DEVICE = args.device
LATENT_DIM = args.latent_dtm
LEARNING_RATE = args.learning_rate
MAX_EPOCHS = args.max_epochs
BATCH_SIZE = args.batch_size


class SimpleAE(nn.Module):
    def __init__(self, in_features:int, n_components:int):
        super().__init__()
        self.in_features = in_features
        self.n_components = n_components
        # build layers
        self.encoder = nn.Sequential(
            nn.Linear(self.in_features, self.n_components),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_components, self.in_features),
            nn.Sigmoid(),
        )

    def forward(self, x:torch.Tensor):
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)

ae = SimpleAE(10,2)


#実装
class WeightTyingLinear(nn.Module):
    """ほぼnn.Linearと同じで,結合重みだけ別のnn.Linearインスタンスを利用するクラス"""
    def __init__(self, shared_weights:torch.Tensor,bias:bool=True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.weight = shared_weights.T
        self.out_features,self.in_features = self.weight.size()
        if bias:
            self.bias = nn.Parameter(torch.empty((self.out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class WeightTyingAE(nn.Module):
    def __init__(self, in_features:int, n_components:int):
        super().__init__()
        self.in_features = in_features
        self.n_components = n_components
        # build layers
        self.encoder = nn.Sequential(
            nn.Linear(self.in_features, self.n_components),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            WeightTyingLinear(self.encoder[0].weight),
            nn.Sigmoid(),
        )

    def forward(self, x:torch.Tensor):
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)


#データの準備
def load_fMNIST_skorch():
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x)),
        ])

    train_set = torchvision.datasets.FashionMNIST(root="./data",
                                           train=True,
                                           download=True,
                                           transform=transform)

    test_set = torchvision.datasets.FashionMNIST(root="./data",
                                         train=False,
                                         download=True,
                                         transform=transform)
    return {"train":train_set, "test": test_set}

#訓練
class NeuralNetAE(NeuralNetRegressor):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        encoded, decoded = y_pred
        recons_loss = super().get_loss(decoded, kwargs["X"], *args, **kwargs)
        return recons_loss
    

class ReconsImage(skorch.callbacks.Callback):
    @staticmethod
    def get_sample():
        """0~9の数字の手書き文字画像をランダムに一枚ずつ適当に選ぶ"""
        transform = transforms.Compose([
            transforms.ToTensor(),])
        test_set = torchvision.datasets.FashionMNIST(root="./data",
                                            train=False,
                                            download=True,
                                            transform=transform)
        sample = []
        i = 0
        for x,y in iter(test_set):
            if y == i:
                sample.append(x)
                i+=1
            if len(sample) == 10:
                break
        return sample

    def initialize(self):
        """最初に一度だけ呼ばれる"""
        self.sample_imgs = self.get_sample()
        self.fig_anime = plt.figure(figsize=[15,5])
        self.axs = []
        for i in range(20):
            self.axs.append(self.fig_anime.add_subplot(2,10,i+1))

        self.frames = []

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs)->None:
        """epoch終了時に毎回呼ばれる"""
        net.module_.eval()
        frame = []
        with torch.no_grad():
            for i in range(10):
                art1 = self.axs[i].imshow(self.sample_imgs[i].view(-1,28).detach().cpu().numpy(),
                            cmap='gray')
                _, sample_hat = net.module_(self.sample_imgs[i].view([1,-1]).to(net.device))
                frame.append(art1)
                art2 = self.axs[i+10].imshow(sample_hat.view(-1,28).detach().cpu().numpy(),
                                cmap="gray")
                frame.append(art2)
        self.frames.append(frame)

    def on_train_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        """学習終了時に一回だけ呼ばれる"""
        plt.close(self.fig_anime)

recons_image = ReconsImage()

trainer = NeuralNetAE(
    # models
    SimpleAE(28**2, LATENT_DIM),
    optimizer= optim.Adam,
    criterion = nn.MSELoss,

    # hyper-params
    lr = LEARNING_RATE,
    batch_size=BATCH_SIZE,
    max_epochs = MAX_EPOCHS,
    device = DEVICE,
    train_split=skorch.dataset.ValidSplit(5), # 0.2 を valid setとして利用

    # callbacks
    callbacks=[("recons_image", recons_image)]
)
# prepare datasets
dataset = load_fMNIST_skorch()
train_dataset = dataset["train"]
y_train = np.array([y for x, y in iter(train_dataset)]) # y_trainはダミーです．np.zeros(len(train_dataset))でもOK．

# training start!
trainer.fit(train_dataset, y_train)