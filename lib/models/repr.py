"""
高速化版Encoderモジュール
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class Encoder(nn.Module):
    """
    高速化版Encoderクラス
    
    最適化:
    - Variableの削除（PyTorch 0.4.0以降では不要）
    - ループ内の操作を最適化
    - 不要なview()操作の削減
    """

    def __init__(self, layers_dims: Tuple[int, ...] = (60, 40, 20)):
        super(Encoder, self).__init__()

        self.layers_dims = layers_dims
        self.input_shape: Optional[torch.Size] = None
        self.layers: list = []
        self.net: Optional[nn.Sequential] = None

        self.outputs: Optional[torch.Tensor] = None

        sinWeights: list = []
        sinCoeffs: list = []

        for i in range(1, len(self.layers_dims)):
            self.layers.append(nn.Linear(self.layers_dims[i-1], self.layers_dims[i], bias=True))

            # Variableの代わりに直接Parameterを作成
            t = torch.randn(self.layers_dims[i - 1], 1, requires_grad=True)
            sinWeights.append(nn.Parameter(t))

            t = torch.randn(1, 1, requires_grad=True)
            sinCoeffs.append(nn.Parameter(t))

        self.sinWeights = sinWeights
        self.sinCoeffs = sinCoeffs

    def freeze(self) -> None:
        """パラメータを凍結"""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """パラメータの凍結を解除"""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor, save: bool = True) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            x: 入力テンソル
            save: 出力を保存するかどうか
            
        Returns:
            エンコードされたテンソル
        """
        if self.input_shape is None:
            self.input_shape = x.shape
            device = x.device  # 入力テンソルのデバイスを取得
            
            # 新しいレイヤーを作成
            new_layer = nn.Linear(self.input_shape[-1], self.layers_dims[0], bias=True)
            self.layers = [new_layer] + self.layers
            
            # Sequentialを作成してからデバイスに移動（すべてのレイヤーが自動的に移動される）
            self.net = nn.Sequential(*self.layers)
            self.net = self.net.to(device)

            # Variableの代わりに直接Parameterを作成（デバイスに合わせる）
            t = torch.randn(self.input_shape[-1], 1, requires_grad=True, device=device)
            self.sinWeights = [nn.Parameter(t)] + self.sinWeights

            t = torch.randn(1, 1, requires_grad=True, device=device)
            self.sinCoeffs = [nn.Parameter(t)] + self.sinCoeffs

            # 既存のsinWeightsとsinCoeffsもデバイスに移動
            for i, param in enumerate(self.sinWeights):
                if param.device != device:
                    self.sinWeights[i] = nn.Parameter(param.data.to(device))
            for i, param in enumerate(self.sinCoeffs):
                if param.device != device:
                    self.sinCoeffs[i] = nn.Parameter(param.data.to(device))
            
            self.sinWeights = nn.ParameterList(self.sinWeights)
            self.sinCoeffs = nn.ParameterList(self.sinCoeffs)
            
            # ParameterListもデバイスに移動
            self.sinWeights = self.sinWeights.to(device)
            self.sinCoeffs = self.sinCoeffs.to(device)

        # ループを最適化：view()操作を削減
        for i in range(len(self.net) - 1):
            x = self.net[i](x)
            # view(1, -1)の代わりに、直接ブロードキャスト可能な形状で計算
            sin_weight = self.sinWeights[i+1]  # shape: (dim, 1)
            sin_coeff = self.sinCoeffs[i+1]    # shape: (1, 1)
            # x * sin_weight.T でブロードキャスト（より効率的）
            x = sin_coeff * torch.sin(2 * np.pi * x * sin_weight.T)
        
        x = self.net[-1](x)
        x = x.view(-1, self.layers_dims[-1])

        if save:
            self.outputs = x

        return x
