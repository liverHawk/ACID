"""
高速化版Datasetモジュール（型ヒント追加）
"""
from torch.utils.data import Dataset as TorchDataset
from typing import Any
import torch


class Dataset(TorchDataset):
    """
    PyTorch用データセットクラス（型ヒント追加）
    """
    
    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        """
        初期化
        
        Args:
            data: データテンソル
            labels: ラベルテンソル
        """
        self.labels = labels
        self.data = data

    def __len__(self) -> int:
        """
        データセットのサイズを返す
        
        Returns:
            データセットのサイズ
        """
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        サンプルを取得
        
        Args:
            index: インデックス
            
        Returns:
            (データ, ラベル)のタプル
        """
        X = self.data[index]
        y = self.labels[index]

        return X, y
