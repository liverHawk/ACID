#!/usr/bin/env python
"""
Adaptive Clusteringモデルの可視化スクリプト

クラスタ中心、埋め込み表現、クラスタ割り当てなどを可視化します
"""
import os
import argparse
import logging
from datetime import datetime
import json
import pickle
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # バックエンドを設定（GUI不要）
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

# libモジュールをインポート
from lib.models.network import AdaptiveClustering
from lib.utils.preprocessing import load_cicids2017_dataset, preprocess_cicids2017, split_dataset


def setup_logging(log_dir: str = 'logs') -> str:
    """
    ログ設定
    
    Args:
        log_dir: ログディレクトリ
        
    Returns:
        タイムスタンプ文字列
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'visualize_output_{timestamp}.txt')
    error_file = os.path.join(log_dir, f'visualize_error_{timestamp}.txt')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    error_handler = logging.FileHandler(error_file)
    error_handler.setLevel(logging.ERROR)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)
    
    return timestamp


def load_trained_model(model_path: str, categories_path: str):
    """
    訓練済みモデルを読み込む
    
    Args:
        model_path: Adaptive Clusteringモデルのパス
        categories_path: カテゴリ情報のパス
        
    Returns:
        (model, categories)のタプル
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logging.info(f"Model loaded: {model_path}")
    
    with open(categories_path, 'r') as f:
        categories = json.load(f)
    logging.info(f"Categories loaded: {categories_path}")
    logging.info(f"Categories: {categories}")
    
    return model, categories


def visualize_cluster_centers(model: AdaptiveClustering, categories: list,
                              results_dir: str = 'results', 
                              timestamp: Optional[str] = None,
                              use_pca: bool = True,
                              use_tsne: bool = True) -> None:
    """
    クラスタ中心（kernel_weights）を可視化
    
    Args:
        model: 訓練済みAdaptive Clusteringモデル
        categories: カテゴリリスト
        results_dir: 結果保存ディレクトリ
        timestamp: タイムスタンプ
        use_pca: PCAを使用するか
        use_tsne: t-SNEを使用するか
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs(results_dir, exist_ok=True)
    
    # クラスタ中心を取得
    cluster_centers = []
    for i in range(model.n_kernels_):
        try:
            # kernel_weightsの形状を確認
            kernel_weights = model.sub_nets[i].kernel_weights
            logging.info(f"SubNet {i} kernel_weights shape: {kernel_weights.shape}, dtype: {kernel_weights.dtype}")
            
            # デバイスからCPUに移動してnumpyに変換
            center = kernel_weights.detach().cpu().numpy()
            
            # 形状を確認
            logging.info(f"SubNet {i} center shape before squeeze: {center.shape}")
            
            # squeezeして1次元に
            center = center.squeeze()
            
            # 1次元配列に変換
            if center.ndim == 0:
                center = np.array([center])
            elif center.ndim > 1:
                center = center.flatten()
            
            logging.info(f"SubNet {i} center shape after processing: {center.shape}, "
                        f"min={center.min():.4f}, max={center.max():.4f}, "
                        f"has_nan={np.isnan(center).any()}, has_inf={np.isinf(center).any()}")
            
            cluster_centers.append(center)
        except Exception as e:
            logging.error(f"Error getting kernel_weights for SubNet {i}: {e}")
            # エラーが発生した場合は、kernel_size分のゼロ配列を作成
            center = np.zeros(model.kernel_size)
            cluster_centers.append(center)
    
    cluster_centers = np.array(cluster_centers)
    
    # NaNや無限大の値をチェック・処理
    nan_mask = np.isnan(cluster_centers)
    inf_mask = np.isinf(cluster_centers)
    if nan_mask.any() or inf_mask.any():
        nan_count = nan_mask.sum()
        inf_count = inf_mask.sum()
        logging.warning(f"Cluster centers contain {nan_count} NaN and {inf_count} inf values. "
                       f"Replacing with 0.")
        cluster_centers = np.nan_to_num(cluster_centers, nan=0.0, posinf=0.0, neginf=0.0)
        
        # すべて0になっている場合は警告
        if np.allclose(cluster_centers, 0):
            logging.warning("All cluster centers are zero after NaN/inf replacement. "
                          "This may indicate a problem with the model or kernel_weights. "
                          "Continuing with zero values for visualization.")
    
    logging.info(f"Cluster centers shape: {cluster_centers.shape}")
    logging.info(f"Cluster centers stats: min={cluster_centers.min():.4f}, max={cluster_centers.max():.4f}, mean={cluster_centers.mean():.4f}")
    
    # クラスタ中心の距離行列を計算
    try:
        distance_matrix = pairwise_distances(cluster_centers)
    except ValueError as e:
        logging.error(f"Error computing distance matrix: {e}")
        logging.info("Skipping distance matrix visualization")
        distance_matrix = None
    
    # 距離行列を可視化
    if distance_matrix is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(distance_matrix, annot=True, fmt='.2f', cmap='viridis',
                    xticklabels=categories, yticklabels=categories,
                    cbar_kws={'label': 'Distance'})
        plt.title('Cluster Centers Distance Matrix', fontsize=16, pad=20)
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Cluster', fontsize=12)
        plt.tight_layout()
        
        distance_plot_path = os.path.join(results_dir, f'cluster_centers_distance_{timestamp}.png')
        plt.savefig(distance_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Cluster centers distance matrix saved: {distance_plot_path}")
    
    # 次元削減して可視化
    if len(cluster_centers.shape) == 1:
        # 1次元の場合は次元削減不要
        logging.warning("Cluster centers are 1-dimensional, skipping dimensionality reduction")
        return
    elif cluster_centers.shape[1] > 2:
        if use_pca:
            # PCAで2次元に削減
            pca = PCA(n_components=2, random_state=42)
            centers_2d_pca = pca.fit_transform(cluster_centers)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(centers_2d_pca[:, 0], centers_2d_pca[:, 1], 
                                s=200, c=range(len(categories)), 
                                cmap='tab10', alpha=0.7, edgecolors='black', linewidths=2)
            for i, category in enumerate(categories):
                plt.annotate(category, (centers_2d_pca[i, 0], centers_2d_pca[i, 1]),
                           fontsize=10, ha='center', va='center', fontweight='bold')
            plt.colorbar(scatter, label='Cluster Index')
            plt.title(f'Cluster Centers (PCA)\nExplained Variance: {pca.explained_variance_ratio_.sum():.2%}', 
                     fontsize=16, pad=20)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            pca_plot_path = os.path.join(results_dir, f'cluster_centers_pca_{timestamp}.png')
            plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Cluster centers PCA plot saved: {pca_plot_path}")
        
        if use_tsne and cluster_centers.shape[0] >= 4:  # t-SNEは少なくとも4サンプル必要
            # t-SNEで2次元に削減
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, cluster_centers.shape[0]-1))
            centers_2d_tsne = tsne.fit_transform(cluster_centers)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(centers_2d_tsne[:, 0], centers_2d_tsne[:, 1], 
                                s=200, c=range(len(categories)), 
                                cmap='tab10', alpha=0.7, edgecolors='black', linewidths=2)
            for i, category in enumerate(categories):
                plt.annotate(category, (centers_2d_tsne[i, 0], centers_2d_tsne[i, 1]),
                           fontsize=10, ha='center', va='center', fontweight='bold')
            plt.colorbar(scatter, label='Cluster Index')
            plt.title('Cluster Centers (t-SNE)', fontsize=16, pad=20)
            plt.xlabel('t-SNE 1', fontsize=12)
            plt.ylabel('t-SNE 2', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            tsne_plot_path = os.path.join(results_dir, f'cluster_centers_tsne_{timestamp}.png')
            plt.savefig(tsne_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Cluster centers t-SNE plot saved: {tsne_plot_path}")
    else:
        # 既に2次元以下の場合は直接プロット
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
                            s=200, c=range(len(categories)), 
                            cmap='tab10', alpha=0.7, edgecolors='black', linewidths=2)
        for i, category in enumerate(categories):
            plt.annotate(category, (cluster_centers[i, 0], cluster_centers[i, 1]),
                       fontsize=10, ha='center', va='center', fontweight='bold')
        plt.colorbar(scatter, label='Cluster Index')
        plt.title('Cluster Centers', fontsize=16, pad=20)
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        centers_plot_path = os.path.join(results_dir, f'cluster_centers_{timestamp}.png')
        plt.savefig(centers_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Cluster centers plot saved: {centers_plot_path}")


def visualize_embeddings(model: AdaptiveClustering, test_data: torch.Tensor,
                         test_labels: np.ndarray, categories: list,
                         results_dir: str = 'results',
                         timestamp: Optional[str] = None,
                         n_samples: int = 5000,
                         use_pca: bool = True,
                         use_tsne: bool = True) -> None:
    """
    埋め込み表現を可視化
    
    Args:
        model: 訓練済みAdaptive Clusteringモデル
        test_data: テストデータ
        test_labels: テストラベル
        categories: カテゴリリスト
        results_dir: 結果保存ディレクトリ
        timestamp: タイムスタンプ
        n_samples: 可視化するサンプル数（多い場合はサンプリング）
        use_pca: PCAを使用するか
        use_tsne: t-SNEを使用するか
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs(results_dir, exist_ok=True)
    
    # サンプリング（データが多い場合）
    if len(test_data) > n_samples:
        indices = np.random.choice(len(test_data), n_samples, replace=False)
        test_data_sampled = test_data[indices]
        test_labels_sampled = test_labels[indices]
        logging.info(f"Sampling {n_samples} samples from {len(test_data)} total samples")
    else:
        test_data_sampled = test_data
        test_labels_sampled = test_labels
    
    # デバイスを取得
    device = next(model.parameters()).device
    test_data_sampled = test_data_sampled.to(device)
    
    # 埋め込み表現を取得
    model.eval()
    embeddings_list = []
    predicted_clusters = []
    
    with torch.no_grad():
        outputs = model(test_data_sampled)
        predicted_clusters = outputs.max(dim=1).indices.cpu().numpy()
        
        # 各サンプルの埋め込み表現を取得
        for i, cluster_idx in enumerate(predicted_clusters):
            embedding = model.sub_nets[cluster_idx].encoder(test_data_sampled[i:i+1])
            embeddings_list.append(embedding.squeeze().cpu().numpy())
    
    embeddings = np.array(embeddings_list)
    
    # NaNや無限大の値をチェック・処理
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        logging.warning("Embeddings contain NaN or inf values. Replacing with 0.")
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    
    logging.info(f"Embeddings shape: {embeddings.shape}")
    
    # 次元削減して可視化
    if len(embeddings.shape) == 1:
        # 1次元の場合は次元削減不要
        logging.warning("Embeddings are 1-dimensional, skipping dimensionality reduction")
        return
    elif embeddings.shape[1] > 2:
        if use_pca:
            # PCAで2次元に削減
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d_pca = pca.fit_transform(embeddings)
            
            plt.figure(figsize=(12, 10))
            scatter = plt.scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1], 
                                c=test_labels_sampled, cmap='tab10', alpha=0.6, s=20)
            plt.colorbar(scatter, label='True Label')
            plt.title(f'Embeddings Visualization (PCA)\nExplained Variance: {pca.explained_variance_ratio_.sum():.2%}', 
                     fontsize=16, pad=20)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            pca_plot_path = os.path.join(results_dir, f'embeddings_pca_{timestamp}.png')
            plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Embeddings PCA plot saved: {pca_plot_path}")
        
        if use_tsne:
            # t-SNEで2次元に削減
            logging.info("Computing t-SNE (this may take a while)...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_2d_tsne = tsne.fit_transform(embeddings)
            
            plt.figure(figsize=(12, 10))
            scatter = plt.scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1], 
                                c=test_labels_sampled, cmap='tab10', alpha=0.6, s=20)
            plt.colorbar(scatter, label='True Label')
            plt.title('Embeddings Visualization (t-SNE)', fontsize=16, pad=20)
            plt.xlabel('t-SNE 1', fontsize=12)
            plt.ylabel('t-SNE 2', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            tsne_plot_path = os.path.join(results_dir, f'embeddings_tsne_{timestamp}.png')
            plt.savefig(tsne_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Embeddings t-SNE plot saved: {tsne_plot_path}")
    else:
        # 既に2次元以下の場合は直接プロット
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                            c=test_labels_sampled, cmap='tab10', alpha=0.6, s=20)
        plt.colorbar(scatter, label='True Label')
        plt.title('Embeddings Visualization', fontsize=16, pad=20)
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        embeddings_plot_path = os.path.join(results_dir, f'embeddings_{timestamp}.png')
        plt.savefig(embeddings_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Embeddings plot saved: {embeddings_plot_path}")


def visualize_cluster_assignments(model: AdaptiveClustering, test_data: torch.Tensor,
                                  test_labels: np.ndarray, categories: list,
                                  results_dir: str = 'results',
                                  timestamp: Optional[str] = None) -> None:
    """
    クラスタ割り当てを可視化
    
    Args:
        model: 訓練済みAdaptive Clusteringモデル
        test_data: テストデータ
        test_labels: テストラベル
        categories: カテゴリリスト
        results_dir: 結果保存ディレクトリ
        timestamp: タイムスタンプ
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs(results_dir, exist_ok=True)
    
    # デバイスを取得
    device = next(model.parameters()).device
    test_data = test_data.to(device)
    
    # クラスタ割り当てを取得
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        predicted_clusters = outputs.max(dim=1).indices.cpu().numpy()
        cluster_probs = outputs.cpu().numpy()
    
    # クラスタ割り当ての混同行列を作成
    assignment_matrix = np.zeros((len(categories), model.n_kernels_))
    for true_label, pred_cluster in zip(test_labels, predicted_clusters):
        assignment_matrix[true_label, pred_cluster] += 1
    
    # 正規化（行方向）
    assignment_matrix_normalized = assignment_matrix / (assignment_matrix.sum(axis=1, keepdims=True) + 1e-10) * 100
    
    # 可視化
    plt.figure(figsize=(12, 8))
    sns.heatmap(assignment_matrix_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=[f'Cluster {i}' for i in range(model.n_kernels_)],
                yticklabels=categories,
                cbar_kws={'label': 'Percentage (%)'})
    plt.title('Cluster Assignment Matrix\n(True Label → Predicted Cluster)', fontsize=16, pad=20)
    plt.xlabel('Predicted Cluster', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    assignment_plot_path = os.path.join(results_dir, f'cluster_assignment_matrix_{timestamp}.png')
    plt.savefig(assignment_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Cluster assignment matrix saved: {assignment_plot_path}")
    
    # 各クラスタの確信度分布を可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i in range(min(4, model.n_kernels_)):  # 最大4つのクラスタを表示
        cluster_probs_i = cluster_probs[:, i]
        axes[i].hist(cluster_probs_i, bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Cluster {i} ({categories[i] if i < len(categories) else "N/A"})', fontsize=12)
        axes[i].set_xlabel('Probability', fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Cluster Assignment Probability Distributions', fontsize=16, y=0.995)
    plt.tight_layout()
    
    prob_dist_plot_path = os.path.join(results_dir, f'cluster_probability_distributions_{timestamp}.png')
    plt.savefig(prob_dist_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Cluster probability distributions saved: {prob_dist_plot_path}")


def visualize_clustering_scatter(model: AdaptiveClustering, test_data: torch.Tensor,
                                 test_labels: np.ndarray, categories: list,
                                 results_dir: str = 'results',
                                 timestamp: Optional[str] = None,
                                 n_samples: int = 5000,
                                 use_pca: bool = True,
                                 use_tsne: bool = True) -> None:
    """
    クラスタリング後の散布図を可視化
    
    Args:
        model: 訓練済みAdaptive Clusteringモデル
        test_data: テストデータ
        test_labels: テストラベル（真のラベル、比較用）
        categories: カテゴリリスト
        results_dir: 結果保存ディレクトリ
        timestamp: タイムスタンプ
        n_samples: 可視化するサンプル数（多い場合はサンプリング）
        use_pca: PCAを使用するか
        use_tsne: t-SNEを使用するか
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs(results_dir, exist_ok=True)
    
    # サンプリング（データが多い場合）
    if len(test_data) > n_samples:
        indices = np.random.choice(len(test_data), n_samples, replace=False)
        test_data_sampled = test_data[indices]
        test_labels_sampled = test_labels[indices]
        logging.info(f"Sampling {n_samples} samples from {len(test_data)} total samples")
    else:
        test_data_sampled = test_data
        test_labels_sampled = test_labels
    
    # デバイスを取得
    device = next(model.parameters()).device
    test_data_sampled = test_data_sampled.to(device)
    
    # クラスタリング結果を取得
    model.eval()
    embeddings_list = []
    predicted_clusters = []
    
    with torch.no_grad():
        outputs = model(test_data_sampled)
        predicted_clusters = outputs.max(dim=1).indices.cpu().numpy()
        
        # 各サンプルの埋め込み表現を取得
        for i, cluster_idx in enumerate(predicted_clusters):
            embedding = model.sub_nets[cluster_idx].encoder(test_data_sampled[i:i+1])
            embeddings_list.append(embedding.squeeze().cpu().numpy())
    
    embeddings = np.array(embeddings_list)
    
    # NaNや無限大の値をチェック・処理
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        logging.warning("Embeddings contain NaN or inf values. Replacing with 0.")
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    
    logging.info(f"Embeddings shape: {embeddings.shape}")
    logging.info(f"Predicted clusters distribution: {np.bincount(predicted_clusters)}")
    
    # クラスタ中心を取得
    cluster_centers = []
    for i in range(model.n_kernels_):
        try:
            kernel_weights = model.sub_nets[i].kernel_weights
            center = kernel_weights.detach().cpu().numpy().squeeze()
            if center.ndim == 0:
                center = np.array([center])
            elif center.ndim > 1:
                center = center.flatten()
            cluster_centers.append(center)
        except Exception as e:
            logging.warning(f"Error getting kernel_weights for SubNet {i}: {e}")
            cluster_centers.append(np.zeros(model.kernel_size))
    
    cluster_centers = np.array(cluster_centers)
    if np.isnan(cluster_centers).any() or np.isinf(cluster_centers).any():
        cluster_centers = np.nan_to_num(cluster_centers, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 次元削減して可視化
    if len(embeddings.shape) == 1 or embeddings.shape[1] <= 2:
        logging.warning("Embeddings are already low-dimensional, skipping dimensionality reduction")
        return
    
    # PCAで2次元に削減
    if use_pca:
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d_pca = pca.fit_transform(embeddings)
        centers_2d_pca = pca.transform(cluster_centers) if cluster_centers.shape[1] == embeddings.shape[1] else None
        
        # 予測クラスタで色分けした散布図
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # 左：予測クラスタで色分け
        scatter1 = axes[0].scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1], 
                                  c=predicted_clusters, cmap='tab10', alpha=0.6, s=20, edgecolors='black', linewidths=0.5)
        if centers_2d_pca is not None:
            axes[0].scatter(centers_2d_pca[:, 0], centers_2d_pca[:, 1], 
                          s=300, c=range(len(categories)), cmap='tab10', 
                          marker='X', edgecolors='black', linewidths=2, 
                          label='Cluster Centers', zorder=5)
            for i, category in enumerate(categories):
                axes[0].annotate(f'C{i}', (centers_2d_pca[i, 0], centers_2d_pca[i, 1]),
                               fontsize=12, ha='center', va='center', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        plt.colorbar(scatter1, ax=axes[0], label='Predicted Cluster')
        axes[0].set_title(f'Clustering Result (PCA)\nExplained Variance: {pca.explained_variance_ratio_.sum():.2%}', 
                         fontsize=14, pad=15)
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # 右：真のラベルで色分け（比較用）
        scatter2 = axes[1].scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1], 
                                  c=test_labels_sampled, cmap='tab10', alpha=0.6, s=20, edgecolors='black', linewidths=0.5)
        plt.colorbar(scatter2, ax=axes[1], label='True Label')
        axes[1].set_title('True Labels (PCA)', fontsize=14, pad=15)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        pca_scatter_path = os.path.join(results_dir, f'clustering_scatter_pca_{timestamp}.png')
        plt.savefig(pca_scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Clustering scatter plot (PCA) saved: {pca_scatter_path}")
    
    # t-SNEで2次元に削減
    if use_tsne:
        logging.info("Computing t-SNE for scatter plot (this may take a while)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d_tsne = tsne.fit_transform(embeddings)
        centers_2d_tsne = tsne.fit_transform(cluster_centers) if cluster_centers.shape[1] == embeddings.shape[1] else None
        
        # 予測クラスタで色分けした散布図
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # 左：予測クラスタで色分け
        scatter1 = axes[0].scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1], 
                                  c=predicted_clusters, cmap='tab10', alpha=0.6, s=20, edgecolors='black', linewidths=0.5)
        if centers_2d_tsne is not None:
            axes[0].scatter(centers_2d_tsne[:, 0], centers_2d_tsne[:, 1], 
                          s=300, c=range(len(categories)), cmap='tab10', 
                          marker='X', edgecolors='black', linewidths=2, 
                          label='Cluster Centers', zorder=5)
            for i, category in enumerate(categories):
                axes[0].annotate(f'C{i}', (centers_2d_tsne[i, 0], centers_2d_tsne[i, 1]),
                               fontsize=12, ha='center', va='center', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        plt.colorbar(scatter1, ax=axes[0], label='Predicted Cluster')
        axes[0].set_title('Clustering Result (t-SNE)', fontsize=14, pad=15)
        axes[0].set_xlabel('t-SNE 1', fontsize=12)
        axes[0].set_ylabel('t-SNE 2', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # 右：真のラベルで色分け（比較用）
        scatter2 = axes[1].scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1], 
                                  c=test_labels_sampled, cmap='tab10', alpha=0.6, s=20, edgecolors='black', linewidths=0.5)
        plt.colorbar(scatter2, ax=axes[1], label='True Label')
        axes[1].set_title('True Labels (t-SNE)', fontsize=14, pad=15)
        axes[1].set_xlabel('t-SNE 1', fontsize=12)
        axes[1].set_ylabel('t-SNE 2', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        tsne_scatter_path = os.path.join(results_dir, f'clustering_scatter_tsne_{timestamp}.png')
        plt.savefig(tsne_scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Clustering scatter plot (t-SNE) saved: {tsne_scatter_path}")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='Visualize Adaptive Clustering model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model (.pkl file)')
    parser.add_argument('--categories_path', type=str, required=True,
                        help='Path to categories file (.categories file)')
    parser.add_argument('--dataset_path', type=str, default='dataset/CICIDS2017_improved',
                        help='Dataset path (default: dataset/CICIDS2017_improved)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Results directory (default: results/)')
    parser.add_argument('--logs_dir', type=str, default='logs',
                        help='Logs directory (default: logs/)')
    parser.add_argument('--train_test_split', type=float, default=0.7,
                        help='Train/test split ratio (default: 0.7)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='Number of samples for embedding visualization (default: 5000)')
    parser.add_argument('--no_pca', action='store_true',
                        help='Disable PCA visualization')
    parser.add_argument('--no_tsne', action='store_true',
                        help='Disable t-SNE visualization')
    
    args = parser.parse_args()
    
    # ログ設定
    timestamp = setup_logging(args.logs_dir)
    logging.info("=" * 50)
    logging.info("Starting Adaptive Clustering visualization")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Categories path: {args.categories_path}")
    logging.info(f"Dataset path: {args.dataset_path}")
    logging.info("=" * 50)
    
    try:
        # 1. 訓練済みモデルの読み込み
        logging.info("Loading trained model...")
        model, categories = load_trained_model(args.model_path, args.categories_path)
        logging.info("Model loaded successfully")
        
        # 2. データ読み込み
        logging.info("Loading dataset...")
        df = load_cicids2017_dataset(args.dataset_path)
        logging.info(f"Dataset loaded: {len(df)} samples")
        
        # 3. 前処理
        logging.info("Preprocessing dataset...")
        df = preprocess_cicids2017(df)
        logging.info("Preprocessing completed")
        
        # 4. データ分割
        logging.info("Splitting dataset...")
        train_df, test_df = split_dataset(df, train_test_split=args.train_test_split, random_seed=args.random_seed)
        logging.info(f"Test samples: {len(test_df)}")
        
        # 5. ラベルのインデックス変換
        label_to_idx = {label: idx for idx, label in enumerate(categories)}
        test_df_ = test_df.drop(['Label'], axis=1).reset_index(drop=True)
        test_labels = test_df['Label'].map(label_to_idx).values
        test_data = torch.FloatTensor(test_df_.values)
        
        # 6. 可視化の実行
        logging.info("=" * 50)
        logging.info("Visualizing cluster centers...")
        visualize_cluster_centers(model, categories, args.results_dir, timestamp,
                                 use_pca=not args.no_pca, use_tsne=not args.no_tsne)
        
        logging.info("Visualizing embeddings...")
        visualize_embeddings(model, test_data, test_labels, categories, 
                           args.results_dir, timestamp, n_samples=args.n_samples,
                           use_pca=not args.no_pca, use_tsne=not args.no_tsne)
        
        logging.info("Visualizing cluster assignments...")
        visualize_cluster_assignments(model, test_data, test_labels, categories,
                                     args.results_dir, timestamp)
        
        logging.info("Visualizing clustering scatter plots...")
        visualize_clustering_scatter(model, test_data, test_labels, categories,
                                    args.results_dir, timestamp, n_samples=args.n_samples,
                                    use_pca=not args.no_pca, use_tsne=not args.no_tsne)
        
        logging.info("=" * 50)
        logging.info("Visualization completed successfully")
        logging.info("=" * 50)
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

