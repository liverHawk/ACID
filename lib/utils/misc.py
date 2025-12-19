"""
高速化版ユーティリティ関数
"""
import numpy as np
import torch
import pandas as pd
from typing import List, Tuple, Any
import os
import sys
from datetime import datetime
from tqdm import tqdm

root_path = os.path.dirname(os.path.realpath(__file__)) + '/../../logs'
sys.path.append(root_path)


class Dot(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def log(msg: Any, filename: str = 'output.txt') -> None:
    """ログを記録"""
    if type(msg) in (list, tuple):
        for message in msg:
            log(message, filename)
    else:
        os.makedirs(root_path, exist_ok=True)
        with open(os.path.join(root_path, filename), mode='a+') as f:
            f.write(f"[{str(datetime.now())}] {msg}\n")


def log_error(msg: Any, filename: str = 'error.txt') -> None:
    """エラーログを記録"""
    log(msg, filename)


def dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    距離を計算
    
    Args:
        x: テンソル1
        y: テンソル2
        
    Returns:
        距離テンソル
    """
    if len(x.shape) == 1:
        x = x.view(-1, 1)
    if len(y.shape) == 1:
        y = y.view(-1, 1)

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    d = torch.pow(x - y, 2).sum(2)
    return d


def extend_dataset(model: Any, df: pd.DataFrame, cats: Any, label_tag: str, 
                  batch_size: int = 1024) -> Tuple[List[List[float]], List[str]]:
    """
    拡張データセットを作成（バッチ処理で高速化、さらに最適化）
    
    Args:
        model: 訓練済みAdaptiveClusteringモデル
        df: データフレーム
        cats: カテゴリラベル
        label_tag: ラベルタグ名
        batch_size: バッチサイズ（デフォルト: 1024）
        
    Returns:
        (特徴量リスト, カラム名リスト)のタプル
    """
    cols = df.columns.values.tolist()

    for i in range(model.kernel_size):
        cols.append(f'kernel_feature_{i}')
    cols.append(label_tag)

    print(f"Creating extended dataset...")
    
    # デバイスを取得（CUDA、MPS、CPUのいずれか）
    device = next(model.parameters()).device
    
    # データをnumpy配列に変換（iterrowsより高速）
    data_array = df.values.astype(np.float32)
    n_samples = len(df)
    rf_features = []
    
    # バッチ処理で高速化
    model.eval()
    with torch.no_grad():
        # カーネル重みを事前に取得してCPUに移動（ベクトル化の準備）
        # detach()を呼んで勾配計算から切り離す
        kernel_weights_list = [
            model.sub_nets[i].kernel_weights.squeeze().detach().cpu().numpy()
            for i in range(model.n_kernels_)
        ]
    
    with torch.no_grad():
        # バッチ数の計算
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for batch_start in tqdm(range(0, n_samples, batch_size), 
                                desc="Processing batches", 
                                total=n_batches,
                                unit="batch"):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_data = data_array[batch_start:batch_end]
            
            # バッチをTensorに変換し、モデルと同じデバイスに移動
            x_batch = torch.FloatTensor(batch_data).to(device)
            
            # バッチ全体で推論
            outputs = model(x_batch)
            labels_batch = outputs.max(dim=1).indices.cpu().numpy()
            
            # 各サンプルに対して特徴量を抽出（ベクトル化）
            for idx_in_batch, (row_data, label_idx, global_idx) in enumerate(
                zip(batch_data, labels_batch, range(batch_start, batch_end))
            ):
                label_idx = int(label_idx)
                attack_type = cats[global_idx]
                
                # 元の特徴量
                rf_features_tmp = row_data.tolist()
                
                # カーネル特徴量を取得（事前に取得したリストから）
                kernel_features = kernel_weights_list[label_idx].tolist()
                
                # 特徴量を結合
                rf_features_tmp.extend(kernel_features)
                rf_features_tmp.append(attack_type)
                
                rf_features.append(rf_features_tmp)
    
    print(f"Done creating extended dataset")

    return rf_features, cols
