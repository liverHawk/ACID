import numpy as np
import torch

import os
import sys
from datetime import datetime

root_path = os.path.dirname(os.path.realpath(__file__)) + '/../../logs'
sys.path.append(root_path)


class Dot(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def log(msg, filename='output.txt'):
    if type(msg) in (list, tuple):
        for message in msg:
            log(message, filename)
    else:
        with open(root_path + '/' + filename, mode='a+') as f:
            f.write(f"[{str(datetime.now())}] {msg}\n")


def log_error(msg, filename='error.txt'):
    log(msg, filename)


def dist(x, y):
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


def extend_dataset(model, df, cats, label_tag, batch_size=1024):
    """
    拡張データセットを作成（バッチ処理で高速化）
    
    Args:
        model: 訓練済みAdaptiveClusteringモデル
        df: データフレーム
        cats: カテゴリラベル
        label_tag: ラベルタグ名
        batch_size: バッチサイズ（デフォルト: 1024）
    """
    cols = df.columns.values.tolist()

    for i in range(model.kernel_size):
        cols.append(f'kernel_feature_{i}')
    cols.append(label_tag)

    print(f"Creating extended dataset...")
    
    # GPU使用可能かチェック
    device = next(model.parameters()).device
    is_cuda = device.type == 'cuda'
    
    # データをnumpy配列に変換（iterrowsより高速）
    data_array = df.values.astype(np.float32)
    n_samples = len(df)
    rf_features = []
    
    # バッチ処理で高速化
    model.eval()
    with torch.no_grad():
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_data = data_array[batch_start:batch_end]
            
            # バッチをTensorに変換
            x_batch = torch.FloatTensor(batch_data)
            if is_cuda:
                x_batch = x_batch.cuda()
            
            # バッチ全体で推論
            outputs = model(x_batch)
            labels_batch = outputs.max(dim=1).indices
            
            # 各サンプルに対して特徴量を抽出
            for idx_in_batch, (row_data, label_idx, global_idx) in enumerate(
                zip(batch_data, labels_batch, range(batch_start, batch_end))
            ):
                label_idx = label_idx.item()
                attack_type = cats[global_idx]
                
                # 元の特徴量
                rf_features_tmp = row_data.tolist()
                
                # カーネル特徴量を取得
                features = model.sub_nets[label_idx].kernel_weights
                kernel_features = features.squeeze().cpu().tolist()
                
                # 特徴量を結合
                rf_features_tmp.extend(kernel_features)
                rf_features_tmp.append(attack_type)
                
                rf_features.append(rf_features_tmp)
            
            # 進捗表示
            if (batch_start // batch_size + 1) % 10 == 0 or batch_end == n_samples:
                print(f"  Processed {batch_end}/{n_samples} samples ({100*batch_end/n_samples:.1f}%)")
    
    print(f"Done creating extended dataset")

    return rf_features, cols
