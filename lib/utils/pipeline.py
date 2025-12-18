"""
エンドツーエンドのパイプライン関数
"""
import os
import json
import logging
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Any
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.network import AdaptiveClustering
from ..utils.dataset import Dataset


class Metrics:
    """評価メトリクスクラス"""
    
    def __init__(self, tp: float = None, tn: float = None, fp: float = None, fn: float = None):
        self.tp, self.tn, self.fp, self.fn = tp, tn, fp, fn
        self.metrics = {}

    def accuracy(self) -> float:
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def detection_rate(self) -> float:
        return self.tp / (self.tp + self.fn)

    def false_alarm_rate(self) -> float:
        return self.fp / (self.fp + self.tn)

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp)

    def f1(self) -> float:
        prec = self.precision()
        rec = self.detection_rate()
        return 0 if (prec + rec) == 0 else 2 * (prec * rec) / (prec + rec)

    def get_metrics(self) -> dict:
        self.metrics = {
            "Acc": self.accuracy(),
            "DR/Recall": self.detection_rate(),
            "FAR": self.false_alarm_rate(),
            "PRECISION": self.precision(),
            "F1 SCORE": self.f1()
        }
        return self.metrics


def train_adaptive_clustering(
    X: torch.Tensor,
    y: torch.Tensor,
    categories: List[str],
    lr: float = 1e-4,
    n_epochs: int = 100,
    early_stop_threshold: float = 1.0,
    batch_size: Optional[int] = None,
    timestamp: Optional[str] = None,
    loss_log_dir: str = "logs",
    device_preference: str = "auto",
) -> Tuple[AdaptiveClustering, List[dict], List[dict]]:
    """
    Adaptive Clusteringモデルを訓練
    
    Args:
        X: 訓練データ
        y: 訓練ラベル
        categories: カテゴリリスト
        lr: 学習率
        n_epochs: エポック数
        early_stop_threshold: 早期停止の閾値
        batch_size: バッチサイズ（Noneの場合は自動決定）
        
    Returns:
        (訓練済みモデル, エポックごとのloss情報, バッチごとのloss情報)のタプル
    """
    # タイムスタンプとログディレクトリを設定
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(loss_log_dir, exist_ok=True)
    nan_log_path = os.path.join(loss_log_dir, f"nan_debug_{timestamp}.json")

    # デバイスを決定
    pref = (device_preference or "auto").lower()
    if pref == "cpu":
        device = torch.device("cpu")
        default_batch_size = 1024
    elif pref == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            default_batch_size = 2048
        else:
            logging.warning("CUDA is not available. Falling back to CPU.")
            device = torch.device("cpu")
            default_batch_size = 1024
    elif pref == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            default_batch_size = 1024
        else:
            logging.warning("MPS is not available. Falling back to CPU.")
            device = torch.device("cpu")
            default_batch_size = 1024
    else:
        # auto: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            default_batch_size = 2048
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            default_batch_size = 1024
        else:
            device = torch.device("cpu")
            default_batch_size = 1024
    
    if batch_size is None:
        batch_size = default_batch_size

    model_ = AdaptiveClustering(
        encoder_dims=[500, 200, 50], 
        n_kernels=len(categories), 
        kernel_size=10
    )
    model_.train()
    
    # モデルをデバイスに移動
    model_ = model_.to(device)
    
    # DataLoaderの設定
    if device.type == 'cuda':
        pin_memory = True
        num_workers = 0  # GPU使用時はnum_workers=0
    else:
        pin_memory = False
        num_workers = 2  # CPU使用時のみマルチプロセッシング

    ds = Dataset(X, y)
    dl = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )

    # optimizerを事前に初期化
    optimizer = torch.optim.Adam(model_.parameters(), lr=lr)

    # loss履歴を保存するリスト
    epoch_losses = []  # 各エポックの平均loss
    batch_losses = []  # 各バッチのloss（詳細）
    for i in range(n_epochs):
        iteration_losses = []
        
        pbar = tqdm(dl, total=len(dl), desc=f"Epoch {i+1}/{n_epochs}")
        for x, labels_ in pbar:
            # バッチをデバイスに移動
            x = x.to(device)
            labels_ = labels_.to(device)
            
            optimizer.zero_grad()
            _ = model_(x, labels_)
            
            loss = model_.loss()
            loss_value = loss.item()
            
            iteration_losses.append(loss_value)
            batch_losses.append({
                'epoch': i,
                'batch': len(iteration_losses) - 1,
                'loss': loss_value
            })
            # NaN / Inf を検知したら詳細をファイルに保存して即エラー
            if not np.isfinite(loss_value):
                try:
                    x_stats = {
                        "min": float(x.min().detach().cpu().item()),
                        "max": float(x.max().detach().cpu().item()),
                        "mean": float(x.mean().detach().cpu().item()),
                    }
                except Exception:
                    x_stats = "unavailable"
                try:
                    labels_stats = {
                        "min": float(labels_.min().detach().cpu().item()),
                        "max": float(labels_.max().detach().cpu().item()),
                        "unique": [float(v) for v in labels_.unique().detach().cpu().tolist()],
                    }
                except Exception:
                    labels_stats = "unavailable"
                loss_debug = {
                    "epoch": i,
                    "batch": len(iteration_losses) - 1,
                    "loss": loss_value,
                    "x_stats": x_stats,
                    "labels_stats": labels_stats,
                }
                try:
                    with open(nan_log_path, "w") as f:
                        json.dump(loss_debug, f, indent=2)
                    logging.error(f"NaN/Inf loss detected. Details saved to {nan_log_path}")
                except Exception as e:
                    logging.error(f"Failed to write NaN debug log: {e}")
                raise ValueError(
                    f"Loss is not finite (nan/inf). epoch={i}, batch={len(iteration_losses) - 1}, loss={loss_value}"
                )
            
            loss.backward()
            optimizer.step()

            # プログレスバーにlossを表示
            pbar.set_postfix({'loss': f'{loss_value:.4f}'})

        avg_loss = np.mean(iteration_losses)
        epoch_losses.append({
            'epoch': i,
            'avg_loss': avg_loss,
            'min_loss': np.min(iteration_losses),
            'max_loss': np.max(iteration_losses),
            'std_loss': np.std(iteration_losses)
        })
        print(f"Iteration {i} | Loss {avg_loss:.6f}")
        if avg_loss < early_stop_threshold:
            print(f"Early stop triggered at: Iteration {i}")
            break
        pbar.close()

    model_.eval()
    return model_, epoch_losses, batch_losses


def evaluate_model(model: Any, test_df: pd.DataFrame, categories: List[str],
                   y_actual: np.ndarray, y_hat: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, dict, List]:
    """
    モデルを評価
    
    Args:
        model: 訓練済みモデル
        test_df: テストデータフレーム
        categories: カテゴリリスト
        y_actual: 実際のラベル
        y_hat: 予測ラベル
        
    Returns:
        (正規化混同行列, 生の数値混同行列, メトリクス, 特徴量重要度)のタプル
    """
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support as prf, accuracy_score
    
    # ラベル名に変換
    y_actual_labels = [categories[i] for i in y_actual]
    y_pred_labels = [categories[i] for i in y_hat]
    
    y_actu = pd.Series(y_actual_labels, name='Actual')
    y_pred = pd.Series(y_pred_labels, name='Predicted')
    
    # 正規化された混同行列（パーセンテージ）を作成
    df_confusion_normalized = pd.crosstab(
        y_actu, y_pred, 
        rownames=['Actual'], 
        colnames=['Predicted'],
        dropna=False, 
        margins=False, 
        normalize='index'
    ).round(4) * 100
    
    # 生の数値（カウント）の混同行列を作成
    df_confusion_counts = pd.crosstab(
        y_actu, y_pred, 
        rownames=['Actual'], 
        colnames=['Predicted'],
        dropna=False, 
        margins=True
    )
    
    # メトリクスを計算
    cnf_matrix = confusion_matrix(y_actual, y_hat)
    _fp = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    _fn = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    _tp = np.diag(cnf_matrix)
    _tn = cnf_matrix.sum() - (_fp + _fn + _tp)
    
    _fp = _fp.astype(float)
    _fn = _fn.astype(float)
    _tp = _tp.astype(float)
    _tn = _tn.astype(float)
    
    _fp = np.mean(_fp)
    _fn = np.mean(_fn)
    _tp = np.mean(_tp)
    _tn = np.mean(_tn)
    
    metrics_handler = Metrics(tp=_tp, tn=_tn, fp=_fp, fn=_fn)
    metrics_data = metrics_handler.get_metrics()
    
    accuracy = accuracy_score(y_actual, y_hat)
    precision, recall, f_score, support = prf(y_actual, y_hat, average='weighted')
    
    metrics_data.update({
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f_score': float(f_score)
    })
    
    # 特徴量重要度を取得
    feature_importance = None
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'cls'):
        importances = model.classifier.cls.feature_importances_
        indices = np.argsort(importances)[-50:][::-1]
        cols = test_df.columns.values.tolist()
        
        feature_importance = []
        for f in range(len(indices)):
            feature_importance.append([
                int(indices[f]),
                cols[indices[f]],
                float(importances[indices[f]])
            ])
    
    return df_confusion_normalized, df_confusion_counts, metrics_data, feature_importance
