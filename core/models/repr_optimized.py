import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np


class EncoderOptimized(nn.Module):
    """
    最適化版Encoder - デバイス移動を最小化して高速化
    元のEncoderと同じ動作を保証しつつ、パフォーマンスを向上
    """

    def __init__(self, layers_dims=(60, 40, 20)):
        super(EncoderOptimized, self).__init__()

        self.layers_dims = layers_dims
        self.input_shape = None
        self.layers = []
        self.net = None
        self.outputs = None
        self._device = None  # デバイス情報をキャッシュ
        self._initialized = False  # 初期化済みフラグ

        sinWeights = []
        sinCoeffs = []

        for i in range(1, len(self.layers_dims)):
            self.layers.append(nn.Linear(self.layers_dims[i-1], self.layers_dims[i], bias=True))

            t = torch.randn(self.layers_dims[i - 1], 1, requires_grad=True)
            sinWeights.append(nn.Parameter(Variable(t, requires_grad=True)))

            t = torch.randn(1, 1, requires_grad=True)
            sinCoeffs.append(nn.Parameter(Variable(t, requires_grad=True)))

        self.sinWeights = sinWeights
        self.sinCoeffs = sinCoeffs

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x, save=True):
        # 入力xのデバイスを取得
        device = x.device

        if self.input_shape is None:
            # 初回のみ初期化
            self.input_shape = x.shape
            
            # 既存のレイヤーとパラメータをデバイスに移動（一度だけ）
            for layer in self.layers:
                layer.to(device)
            for param in self.sinWeights:
                param.data = param.data.to(device)
            for param in self.sinCoeffs:
                param.data = param.data.to(device)
            
            # 新しい入力レイヤーを作成してデバイスに移動
            new_layer = nn.Linear(self.input_shape[-1], self.layers_dims[0], bias=True)
            new_layer = new_layer.to(device)
            self.layers = [new_layer] + self.layers
            
            # すべてのレイヤーをデバイスに移動してからSequentialに追加
            for layer in self.layers:
                layer.to(device)
            self.net = nn.Sequential(*self.layers)
            # ネットワークをselfに登録
            if 'net' not in self._modules:
                self.add_module('net', self.net)
            # ネットワーク全体をデバイスに移動
            self.net = self.net.to(device)
            self.to(device)

            t = torch.randn(self.input_shape[-1], 1, requires_grad=True, device=device)
            self.sinWeights = [nn.Parameter(Variable(t, requires_grad=True))] + self.sinWeights

            t = torch.randn(1, 1, requires_grad=True, device=device)
            self.sinCoeffs = [nn.Parameter(Variable(t, requires_grad=True))] + self.sinCoeffs

            self.sinWeights = nn.ParameterList(self.sinWeights)
            self.sinCoeffs = nn.ParameterList(self.sinCoeffs)
            
            self._initialized = True
            self._device = device
        else:
            # 2回目以降は、デバイスが変わった場合のみ移動（通常は発生しない）
            if self._device is None or self._device != device:
                # デバイスが変わった場合のみ移動
                if self.net is not None:
                    self.net = self.net.to(device)
                if isinstance(self.sinWeights, nn.ParameterList):
                    for param in self.sinWeights:
                        param.data = param.data.to(device)
                if isinstance(self.sinCoeffs, nn.ParameterList):
                    for param in self.sinCoeffs:
                        param.data = param.data.to(device)
                self._device = device

        # forward実行（デバイス移動は不要）
        for i in range(len(self.net) - 1):
            x = self.net[i](x)
            x = self.sinCoeffs[i+1] * torch.sin(2 * np.pi * x * self.sinWeights[i+1].view(1, -1))
        x = self.net[-1](x)
        x = x.view(-1, self.layers_dims[-1])

        if save:
            self.outputs = x

        return x

