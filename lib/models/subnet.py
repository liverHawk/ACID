"""
高速化版SubNetモジュール
"""
import torch
import torch.nn as nn
from typing import List, Optional


def get_rand(w: int, h: int, grad: bool = True) -> nn.Parameter:
    """
    ランダムパラメータを生成（Variableの代わりに直接Parameterを作成）
    
    Args:
        w: 幅
        h: 高さ
        grad: 勾配を計算するかどうか
        
    Returns:
        ランダムパラメータ
    """
    t = torch.randn(w, h, requires_grad=grad)
    return nn.Parameter(t)


class SubNet(nn.Module):
    """
    高速化版SubNetクラス
    
    最適化:
    - Variableの削除
    - ループの最適化
    """
    
    def __init__(self, input_size: int, layers_dims: List[int] = [50, 30]):
        super(SubNet, self).__init__()

        self.input_size = input_size
        self.layers_dims = [input_size + 0] + list(layers_dims) + [1]
        self.layers: List[nn.Module] = []

        for i in range(1, len(self.layers_dims)):
            self.layers.append(nn.Linear(self.layers_dims[i-1], self.layers_dims[i]))
        self.net = nn.Sequential(*self.layers)

        self.kernel_weights = get_rand(1, input_size)

        self.encoder: Optional[nn.Module] = None
        self.outputs: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            x: 入力テンソル
            
        Returns:
            出力テンソル
        """
        # ループを最適化：Sequentialを使用してより効率的に
        for i in range(len(self.net) - 1):
            x = self.net[i](x)
            x = torch.tanh(x)
        x = self.net[-1](x)
        x = torch.sigmoid(x)

        self.outputs = x

        return x
