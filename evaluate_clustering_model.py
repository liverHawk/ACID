#!/usr/bin/env python
"""
IDS_cicids2017.pyで訓練されたAdaptiveClusteringモデルを評価するスクリプト

クラスター数、クラスタ内データ数、その他の評価指標を計算し、結果を可視化します。
"""
import os
import argparse
import logging
from datetime import datetime
import json
import pickle
from typing import Optional

import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # バックエンドを設定（GUI不要）
import matplotlib.pyplot as plt
import seaborn as sns

# libモジュールをインポート
from lib.models.network import AdaptiveClustering
from lib.utils.preprocessing import load_cicids2017_dataset, preprocess_cicids2017, split_dataset
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix


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
    
    log_file = os.path.join(log_dir, f'evaluate_clustering_output_{timestamp}.txt')
    error_file = os.path.join(log_dir, f'evaluate_clustering_error_{timestamp}.txt')
    
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


def get_device(device_preference: str = "auto") -> torch.device:
    """
    デバイスを取得
    
    Args:
        device_preference: デバイス指定（auto/cpu/cuda/mps）
        
    Returns:
        torch.device
    """
    pref = (device_preference or "auto").lower()
    if pref == "cpu":
        device = torch.device("cpu")
    elif pref == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logging.warning("CUDA is not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif pref == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            logging.warning("MPS is not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        # auto: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    logging.info(f"Using device: {device}")
    return device


def load_model(model_path: str, categories_path: str, device: torch.device) -> tuple:
    """
    訓練済みモデルを読み込む
    
    Args:
        model_path: モデルファイルのパス
        categories_path: カテゴリ情報のパス
        device: デバイス
        
    Returns:
        (model, categories)のタプル
    """
    # Adaptive Clusteringモデルを読み込み
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logging.info(f"Model loaded: {model_path}")
    
    # カテゴリ情報を読み込み
    with open(categories_path, 'r') as f:
        categories = json.load(f)
    logging.info(f"Categories loaded: {categories_path}")
    logging.info(f"Categories: {categories}")
    
    # モデルをデバイスに移動
    model = model.to(device)
    model.eval()
    
    return model, categories


def evaluate_clustering(model: AdaptiveClustering, test_data: torch.Tensor,
                        test_labels: np.ndarray, categories: list, device: torch.device,
                        batch_size: int = 8192) -> dict:
    """
    クラスタリング結果を評価
    
    Args:
        model: 訓練済みAdaptive Clusteringモデル
        test_data: テストデータ
        test_labels: テストラベル（真のラベル）
        categories: カテゴリリスト
        device: デバイス
        
    Returns:
        評価メトリクスの辞書
    """
    model.eval()
    
    # バッチ処理で推論を実行（メモリ効率のため）
    all_outputs = []
    n_samples = len(test_data)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_data = test_data[i:batch_end].to(device)
            
            batch_outputs = model(batch_data)
            all_outputs.append(batch_outputs.detach().cpu())
        
        # すべてのバッチの出力を結合
        outputs = torch.cat(all_outputs, dim=0)
        
        # デバッグ情報: 出力の統計を確認
        logging.info(f"Output shape: {outputs.shape}")
        
        # 出力の形状を確認して処理
        # outputsは [batch, n_kernels, 1] の形状の可能性がある
        if len(outputs.shape) == 3:
            # [batch, n_kernels, 1] -> [batch, n_kernels] に変換
            outputs = outputs.squeeze(-1)
        
        output_probs = outputs.numpy()
        logging.info(f"Output shape after squeeze: {outputs.shape}")
        logging.info(f"Output min: {output_probs.min():.6f}, max: {output_probs.max():.6f}, mean: {output_probs.mean():.6f}")
        
        # 各サンプルの最大確率クラスタを確認
        max_probs = output_probs.max(axis=1)
        logging.info(f"Max probabilities - min: {max_probs.min():.6f}, max: {max_probs.max():.6f}, mean: {max_probs.mean():.6f}")
        
        # 最初の10サンプルの出力を確認
        logging.info(f"First 10 samples output probabilities:\n{output_probs[:10]}")
        
        # 各クラスタへの確率分布を確認（最初の数サンプル）
        logging.info(f"Probability distribution across clusters (first 5 samples):")
        for sample_idx in range(min(5, len(output_probs))):
            probs = output_probs[sample_idx]
            prob_dict = {i: f'{float(p):.4f}' for i, p in enumerate(probs)}
            logging.info(f"  Sample {sample_idx}: {prob_dict}")
        
        # 最大確率のクラスタを取得
        predicted_clusters = (
            outputs.max(dim=1).indices
            .numpy()
            .astype(int)
            .reshape(-1)
        )
        # 念のため負値を0にクリップ
        predicted_clusters = np.clip(predicted_clusters, 0, len(categories) - 1)
        
        # デバッグ情報: 予測クラスタの分布を確認
        unique_clusters, counts = np.unique(predicted_clusters, return_counts=True)
        logging.info(f"Predicted clusters distribution: {dict(zip(unique_clusters, counts))}")
        
        # 各クラスタに割り当てられた確率の統計を確認
        cluster_probs = output_probs.max(axis=1)
        logging.info(f"Cluster assignment probabilities - min: {cluster_probs.min():.6f}, "
                    f"max: {cluster_probs.max():.6f}, mean: {cluster_probs.mean():.6f}, "
                    f"std: {cluster_probs.std():.6f}")
        
        # 各クラスタへの確率分布を確認
        for cluster_idx in range(min(5, len(categories))):  # 最初の5クラスタのみ
            cluster_mask = predicted_clusters == cluster_idx
            if cluster_mask.sum() > 0:
                probs_for_cluster = cluster_probs[cluster_mask]
                logging.info(f"Cluster {cluster_idx} - samples: {cluster_mask.sum()}, "
                            f"avg prob: {probs_for_cluster.mean():.6f}, "
                            f"min prob: {probs_for_cluster.min():.6f}, "
                            f"max prob: {probs_for_cluster.max():.6f}")
    
    # 埋め込み表現を取得（シルエットスコア用）
    embeddings_list = []
    # バッチ処理で埋め込みを取得（メモリ効率のため）
    with torch.no_grad():
        for i in range(0, len(predicted_clusters), batch_size):
            batch_end = min(i + batch_size, len(predicted_clusters))
            batch_indices = range(i, batch_end)
            batch_data = test_data[i:batch_end].to(device)
            batch_clusters = predicted_clusters[i:batch_end]
            
            for j, (idx, cluster_idx) in enumerate(zip(batch_indices, batch_clusters)):
                # numpy配列の要素をPythonのintに変換
                cluster_idx_int = int(cluster_idx)
                embedding = model.sub_nets[cluster_idx_int].encoder(batch_data[j:j+1])
                embeddings_list.append(embedding.squeeze().detach().cpu().numpy())
    
    embeddings = np.array(embeddings_list)
    
    # NaNや無限大の値をチェック・処理
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        logging.warning("Embeddings contain NaN or inf values. Replacing with 0.")
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 評価メトリクスを計算
    metrics = {}
    
    # クラスター数
    n_clusters = len(np.unique(predicted_clusters))
    metrics['n_clusters'] = int(n_clusters)
    
    # Adjusted Rand Index (ARI)
    ari = adjusted_rand_score(test_labels, predicted_clusters)
    metrics['adjusted_rand_index'] = float(ari)
    
    # Normalized Mutual Information (NMI)
    nmi = normalized_mutual_info_score(test_labels, predicted_clusters)
    metrics['normalized_mutual_info'] = float(nmi)
    
    # シルエットスコア（埋め込み表現が2次元以上の場合）
    if embeddings.shape[1] >= 2:
        try:
            # サンプル数が多い場合はサブサンプリング
            if len(embeddings) > 10000:
                indices = np.random.choice(len(embeddings), 10000, replace=False)
                silhouette = silhouette_score(embeddings[indices], predicted_clusters[indices])
            else:
                silhouette = silhouette_score(embeddings, predicted_clusters)
            metrics['silhouette_score'] = float(silhouette)
        except Exception as e:
            logging.warning(f"Could not compute silhouette score: {e}")
            metrics['silhouette_score'] = None
    
    # クラスタリング精度（各クラスタに最も多く割り当てられた真のラベルを正解とする）
    contingency = contingency_matrix(test_labels, predicted_clusters)
    cluster_to_label = {}
    for cluster_idx in range(len(categories)):
        if cluster_idx < contingency.shape[1]:
            label_idx = np.argmax(contingency[:, cluster_idx])
            cluster_to_label[cluster_idx] = label_idx
    
    correct = 0
    for true_label, pred_cluster in zip(test_labels, predicted_clusters):
        if pred_cluster in cluster_to_label:
            if true_label == cluster_to_label[pred_cluster]:
                correct += 1
    
    clustering_accuracy = correct / len(test_labels)
    metrics['clustering_accuracy'] = float(clustering_accuracy)
    
    # 各クラスタのサイズ（クラスタ内データ数）
    cluster_sizes = np.bincount(predicted_clusters, minlength=len(categories))
    metrics['cluster_sizes'] = cluster_sizes.tolist()
    
    # 各クラスタの純度（最も多いラベルの割合）
    cluster_purities = []
    for cluster_idx in range(len(categories)):
        if cluster_idx < contingency.shape[1]:
            cluster_mask = predicted_clusters == cluster_idx
            if cluster_mask.sum() > 0:
                cluster_labels = test_labels[cluster_mask]
                most_common_count = np.bincount(cluster_labels).max()
                purity = most_common_count / len(cluster_labels)
                cluster_purities.append(purity)
            else:
                cluster_purities.append(0.0)
        else:
            cluster_purities.append(0.0)
    metrics['cluster_purities'] = cluster_purities
    
    # 予測クラスタも返す（可視化用）
    metrics['predicted_clusters'] = predicted_clusters.tolist()
    
    return metrics


def visualize_cluster_sizes(cluster_sizes: list, categories: list,
                             results_dir: str) -> None:
    """
    クラスタサイズを可視化
    
    Args:
        cluster_sizes: 各クラスタのサイズ
        categories: カテゴリリスト
        results_dir: 結果保存ディレクトリ
    """
    plt.figure(figsize=(12, 6))
    cluster_indices = list(range(len(cluster_sizes)))
    bars = plt.bar(cluster_indices, cluster_sizes, color='steelblue', alpha=0.7)
    
    # 各バーに値を表示
    for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
        if size > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cluster_sizes)*0.01,
                    f'{int(size)}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Cluster Index', fontsize=12)
    plt.ylabel('Number of Data Points', fontsize=12)
    plt.title('Cluster Sizes (Number of Data Points per Cluster)', fontsize=14, pad=20)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(cluster_indices, [f'C{i}' for i in cluster_indices], rotation=45)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'cluster_sizes.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Cluster sizes plot saved: {plot_path}")


def visualize_cluster_purities(cluster_purities: list, categories: list,
                                results_dir: str) -> None:
    """
    クラスタ純度を可視化
    
    Args:
        cluster_purities: 各クラスタの純度
        categories: カテゴリリスト
        results_dir: 結果保存ディレクトリ
    """
    plt.figure(figsize=(12, 6))
    cluster_indices = list(range(len(cluster_purities)))
    bars = plt.bar(cluster_indices, cluster_purities, color='coral', alpha=0.7)
    
    # 各バーに値を表示
    for i, (bar, purity) in enumerate(zip(bars, cluster_purities)):
        if purity > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{purity:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Cluster Index', fontsize=12)
    plt.ylabel('Purity', fontsize=12)
    plt.title('Cluster Purities (Dominant Label Ratio per Cluster)', fontsize=14, pad=20)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(cluster_indices, [f'C{i}' for i in cluster_indices], rotation=45)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'cluster_purities.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Cluster purities plot saved: {plot_path}")


def visualize_cluster_assignment_matrix(test_labels: np.ndarray, predicted_clusters: np.ndarray,
                                        categories: list, results_dir: str) -> None:
    """
    クラスタ割り当てマトリックスを可視化
    
    Args:
        test_labels: 真のラベル
        predicted_clusters: 予測クラスタ
        categories: カテゴリリスト
        results_dir: 結果保存ディレクトリ
    """
    # コンティンジェンシーマトリックスを作成
    contingency = contingency_matrix(test_labels, predicted_clusters)
    
    # データフレームに変換
    df_matrix = pd.DataFrame(
        contingency,
        index=[f'{categories[i]}' for i in range(len(categories))],
        columns=[f'Cluster {i}' for i in range(contingency.shape[1])]
    )
    
    # ヒートマップを作成
    plt.figure(figsize=(max(12, len(categories)*0.8), max(8, contingency.shape[1]*0.6)))
    sns.heatmap(df_matrix, annot=True, fmt='d', cmap='Blues', 
                cbar_kws={'label': 'Number of Data Points'})
    plt.title('Cluster Assignment Matrix\n(True Label → Predicted Cluster)', 
              fontsize=16, pad=20)
    plt.xlabel('Predicted Cluster', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, 'cluster_assignment_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Cluster assignment matrix plot saved: {plot_path}")


def save_results(metrics: dict, test_labels: np.ndarray, predicted_clusters: np.ndarray,
                 categories: list, results_dir: str) -> None:
    """
    結果を保存
    
    Args:
        metrics: 評価メトリクス
        test_labels: 真のラベル
        predicted_clusters: 予測クラスタ
        categories: カテゴリリスト
        results_dir: 結果保存ディレクトリ
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # メトリクスを保存（JSON形式）
    metrics_to_save = metrics.copy()
    # predicted_clustersはリストが長いので除外（別途CSVで保存）
    if 'predicted_clusters' in metrics_to_save:
        del metrics_to_save['predicted_clusters']
    
    metrics_json_path = os.path.join(results_dir, 'clustering_metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2, default=str)
    logging.info(f"Clustering metrics (JSON) saved: {metrics_json_path}")
    
    # メトリクスを保存（pickle形式）
    metrics_pkl_path = os.path.join(results_dir, 'clustering_metrics.pkl')
    with open(metrics_pkl_path, 'wb') as f:
        pickle.dump(metrics_to_save, f)
    logging.info(f"Clustering metrics (pickle) saved: {metrics_pkl_path}")
    
    # クラスタ割り当てを保存
    assignments_df = pd.DataFrame({
        'true_label': [categories[label] for label in test_labels],
        'predicted_cluster': predicted_clusters,
        'true_label_idx': test_labels
    })
    assignments_path = os.path.join(results_dir, 'clustering_assignments.csv')
    assignments_df.to_csv(assignments_path, index=False)
    logging.info(f"Clustering assignments saved: {assignments_path}")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description='Evaluate Adaptive Clustering model trained with IDS_cicids2017.py'
    )
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
                        help='Train/test split ratio used during training (default: 0.7)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for data splitting (default: 42)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use: auto / cpu / cuda / mps (default: auto)')
    
    args = parser.parse_args()
    
    # ログ設定
    timestamp = setup_logging(args.logs_dir)
    logging.info("=" * 50)
    logging.info("Starting Clustering Model Evaluation")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Categories path: {args.categories_path}")
    logging.info(f"Dataset path: {args.dataset_path}")
    logging.info(f"Results directory: {args.results_dir}")
    logging.info(f"Device preference: {args.device}")
    logging.info("=" * 50)
    
    try:
        # 1. デバイスを決定
        device = get_device(args.device)
        
        # 2. 訓練済みモデルの読み込み
        logging.info("Loading trained model...")
        model, categories = load_model(args.model_path, args.categories_path, device)
        logging.info("Model loaded successfully")
        
        # デバッグ情報: モデルの状態を確認
        logging.info(f"Model training mode: {model.training}")
        logging.info(f"Number of kernels: {model.n_kernels_}")
        logging.info(f"Kernel size: {model.kernel_size}")
        logging.info(f"Number of categories: {len(categories)}")
        
        # カーネル重みの統計を確認（最初のカーネルのみ）
        if model.n_kernels_ > 0:
            kernel_weights = model.sub_nets[0].kernel_weights.detach().cpu().numpy()
            logging.info(f"First kernel weights - shape: {kernel_weights.shape}, "
                        f"min: {kernel_weights.min():.6f}, max: {kernel_weights.max():.6f}, "
                        f"mean: {kernel_weights.mean():.6f}, std: {kernel_weights.std():.6f}")
        
        # 3. データ読み込み
        logging.info("Loading dataset...")
        df = load_cicids2017_dataset(args.dataset_path)
        logging.info(f"Dataset loaded: {len(df)} samples")
        
        # 4. 前処理
        logging.info("Preprocessing dataset...")
        df = preprocess_cicids2017(df)
        logging.info("Preprocessing completed")
        
        # 5. データ分割（訓練時と同じ分割を使用）
        logging.info("Splitting dataset...")
        train_df, test_df = split_dataset(df, train_test_split=args.train_test_split, 
                                         random_seed=args.random_seed)
        logging.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
        
        # 6. ラベルのインデックス変換マッピングを作成
        label_to_idx = {label: idx for idx, label in enumerate(categories)}
        logging.info(f"Label to index mapping: {label_to_idx}")
        
        # 7. テストデータの準備
        logging.info("Preparing test data...")
        test_df_ = test_df.drop(['Label'], axis=1)
        X_test = torch.FloatTensor(test_df_.values)
        y_test = np.array([label_to_idx[lbl] for lbl in test_df['Label'].values])
        logging.info(f"Test data prepared: {len(X_test)} samples")
        
        # デバッグ情報: テストデータの統計を確認
        X_test_np = X_test.numpy()
        logging.info(f"Test data shape: {X_test_np.shape}")
        logging.info(f"Test data - min: {X_test_np.min():.6f}, max: {X_test_np.max():.6f}, "
                    f"mean: {X_test_np.mean():.6f}, std: {X_test_np.std():.6f}")
        logging.info(f"Test data contains NaN: {np.isnan(X_test_np).any()}")
        logging.info(f"Test data contains Inf: {np.isinf(X_test_np).any()}")
        
        # 真のラベルの分布を確認
        unique_labels, label_counts = np.unique(y_test, return_counts=True)
        logging.info(f"True label distribution: {dict(zip([categories[i] for i in unique_labels], label_counts))}")
        
        # 8. クラスタリング評価
        logging.info("Evaluating clustering performance...")
        # バッチサイズを設定（メモリに応じて調整）
        eval_batch_size = 8192 if device.type == 'cuda' else 4096
        metrics = evaluate_clustering(model, X_test, y_test, categories, device, batch_size=eval_batch_size)
        
        # 9. 結果の表示
        logging.info("=" * 50)
        logging.info("Clustering Evaluation Results:")
        logging.info(f"Number of Clusters: {metrics.get('n_clusters', 0)}")
        logging.info(f"Adjusted Rand Index (ARI): {metrics.get('adjusted_rand_index', 0):.4f}")
        logging.info(f"Normalized Mutual Information (NMI): {metrics.get('normalized_mutual_info', 0):.4f}")
        if metrics.get('silhouette_score') is not None:
            logging.info(f"Silhouette Score: {metrics.get('silhouette_score', 0):.4f}")
        logging.info(f"Clustering Accuracy: {metrics.get('clustering_accuracy', 0):.4f}")
        logging.info(f"Cluster Sizes: {metrics.get('cluster_sizes', [])}")
        logging.info(f"Cluster Purities: {[f'{p:.4f}' for p in metrics.get('cluster_purities', [])]}")
        logging.info("=" * 50)
        
        # 10. タイムスタンプ付きディレクトリを作成
        timestamp_dir = os.path.join(args.results_dir, timestamp)
        os.makedirs(timestamp_dir, exist_ok=True)
        logging.info(f"Results will be saved to: {timestamp_dir}")
        
        # 11. 可視化
        logging.info("Creating visualizations...")
        predicted_clusters = np.array(metrics['predicted_clusters'])
        visualize_cluster_sizes(metrics['cluster_sizes'], categories, timestamp_dir)
        visualize_cluster_purities(metrics['cluster_purities'], categories, timestamp_dir)
        visualize_cluster_assignment_matrix(y_test, predicted_clusters, categories, timestamp_dir)
        logging.info("Visualizations created")
        
        # 12. 結果の保存
        logging.info("Saving results...")
        save_results(metrics, y_test, predicted_clusters, categories, timestamp_dir)
        logging.info("Results saved")
        
        logging.info("=" * 50)
        logging.info("Evaluation completed successfully")
        logging.info("=" * 50)
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
