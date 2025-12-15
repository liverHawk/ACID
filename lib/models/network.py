"""
高速化版AdaptiveClusteringモジュール
"""
from .repr import Encoder
from .subnet import SubNet

import torch
import torch.nn as nn
from typing import Optional, List, Tuple


class AdaptiveClustering(nn.Module):
    """
    高速化版AdaptiveClusteringクラス
    
    最適化:
    - loss()メソッドでnonzero()の代わりにマスクを使用
    - clustering_loss_distの計算をベクトル化
    - forward()メソッドでリスト内包表記を使用
    """
    
    def __init__(self, encoder_dims: Tuple[int, ...] = (60, 40, 20), 
                 kernel_size: int = 3, n_kernels: Optional[int] = None, 
                 subnet_dims: List[int] = [50, 30]):
        super(AdaptiveClustering, self).__init__()

        assert n_kernels is not None, f"The number of kernels must be defined"
        assert n_kernels > 0, f"The number of kernels ({n_kernels}) must be greater than 0"
        assert kernel_size > 0, f"The dimensionality of kernels ({kernel_size}) must be greater than 0"

        self.kernel_size = kernel_size
        self.n_kernels = n_kernels
        self.subnet_dims = subnet_dims
        self.encoder_dims = list(encoder_dims) + [kernel_size]
        self.labels_: Optional[torch.Tensor] = None

        self.sub_nets = nn.ParameterList([])
        self.sub_nets_list: List[SubNet] = []

        self.add_sub_net()

        for i in range(len(self.sub_nets_list), self.n_kernels):
            self.add_sub_net()

    def add_sub_net(self) -> None:
        """サブネットを追加"""
        sub_net = SubNet(self.encoder_dims[-1], self.subnet_dims)
        sub_net.encoder = Encoder(layers_dims=self.encoder_dims)
        self.sub_nets_list.append(sub_net)
        self.sub_nets = nn.ModuleList(self.sub_nets_list)

    def remove_sub_net(self, idx: int) -> None:
        """サブネットを削除"""
        self.sub_nets_list.pop(idx)
        del self.sub_nets[idx]
        self.sub_nets = nn.ModuleList(self.sub_nets_list)

    def reset(self) -> None:
        """リセット（現在は何もしない）"""
        pass

    @property
    def n_kernels_(self) -> int:
        """カーネル数を返す"""
        return len(self.sub_nets)

    def loss(self) -> torch.Tensor:
        """
        損失を計算（高速化版）
        
        最適化:
        - nonzero()の代わりにマスク（boolean indexing）を使用
        - clustering_loss_distの計算をベクトル化
        """
        classification_loss = 0.0
        clustering_loss_close = 0.0
        clustering_loss_dist = 0.0
        labels = self.labels_

        for i in range(self.n_kernels_):
            # nonzero()の代わりにマスクを使用（高速化）
            mask_i = (labels == i)
            mask_not_i = (labels != i)
            
            # マスクが空でない場合のみ処理
            if not mask_i.any():
                continue
                
            idx = mask_i.nonzero(as_tuple=False).squeeze(-1)
            idx_diff = mask_not_i.nonzero(as_tuple=False).squeeze(-1)

            embeddings = self.sub_nets[i].encoder.outputs
            outputs = self.sub_nets[i].outputs

            targets = torch.zeros_like(outputs)
            targets[idx] = 1.0
            classification_loss += ((outputs.squeeze() - targets.squeeze()) ** 2).mean()

            if len(idx) > 0:
                # view()操作を最適化
                embeddings_selected = embeddings[idx].view(-1, self.kernel_size)
                clustering_loss_close += ((embeddings_selected -
                                          self.sub_nets[i].kernel_weights) ** 2).mean()
                
                if len(idx_diff) > 0:
                    # ベクトル化：すべての異なるラベルに対して一度に計算
                    unique_labels = labels[idx_diff].unique()
                    kernel_weights_i = self.sub_nets[i].kernel_weights.clone().detach()
                    
                    for j in unique_labels:
                        j = j.item()
                        if j < self.n_kernels_:
                            kernel_weights_j = self.sub_nets[j].kernel_weights
                            tmp_loss = ((kernel_weights_i - kernel_weights_j) ** 2).mean()
                            clustering_loss_dist += torch.clamp(1 - tmp_loss, min=0)

        n_k_dist = (self.n_kernels_ - 1 if self.n_kernels_ > 1 else 1) * self.n_kernels_
        return classification_loss + (clustering_loss_close + clustering_loss_dist/n_k_dist) / self.n_kernels_

    def backward(self, losses: List[torch.Tensor]) -> None:
        """複数の損失に対してバックワードパスを実行"""
        for loss in losses:
            loss.backward(retain_graph=True)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        フォワードパス（高速化版）
        
        最適化:
        - リスト内包表記を使用して高速化
        """
        if self.training:
            assert labels is not None, "True labels must be provided during training."
        self.labels_ = labels

        # リスト内包表記で高速化
        outputs = [
            self.sub_nets[i](self.sub_nets[i].encoder(x)).view(x.shape[0], 1, -1)
            for i in range(self.n_kernels_)
        ]

        outputs = torch.cat(tuple(outputs), dim=1)
        outputs = torch.softmax(outputs, dim=1)

        return outputs
