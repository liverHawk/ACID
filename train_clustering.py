#!/usr/bin/env python
"""
Adaptive Clusteringのみを訓練するスクリプト

Random Forestの分類は行わず、クラスタリングのみに焦点を当てます
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
import yaml

# libモジュールをインポート
from lib.models.network import AdaptiveClustering
from lib.utils.preprocessing import load_cicids2017_dataset, preprocess_cicids2017, split_dataset
from lib.utils.pipeline import train_adaptive_clustering
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
    
    log_file = os.path.join(log_dir, f'clustering_output_{timestamp}.txt')
    error_file = os.path.join(log_dir, f'clustering_error_{timestamp}.txt')
    
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


def load_config(config_path: Optional[str]) -> Optional[dict]:
    if config_path is None:
        return None
    if not os.path.exists(config_path):
        logging.warning(f"Config file not found: {config_path}. Using all labels.")
        return None
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_model(model: AdaptiveClustering, categories: list,
               results_dir: str = 'results', timestamp: Optional[str] = None) -> tuple:
    """
    モデルとカテゴリ情報を保存

    注意:
        categories には「可視化や評価に用いるカテゴリリスト」を渡す。
        そのため、既知ラベルだけでなく全ラベル（評価用カテゴリ）を渡すことで、
        visualize_adaptive_clustering.py 側でデータセット中の全ラベルを
        正しくインデックス変換できるようにする。

    Args:
        model: 訓練済み Adaptive Clustering モデル
        categories: カテゴリリスト（評価に用いる全ラベル）
        results_dir: 結果保存ディレクトリ
        timestamp: タイムスタンプ（None の場合は自動生成）

    Returns:
        (モデルパス, カテゴリパス) のタプル
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Adaptive Clusteringモデルを保存
    # run_ids_cicids2017.py と同じ命名規則（trained_model_...）に合わせる
    model_path = os.path.join(results_dir, f'trained_model_{timestamp}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved: {model_path}")
    
    # カテゴリ情報を保存（こちらも run_ids_cicids2017.py に合わせる）
    categories_path = os.path.join(results_dir, f'trained_model_{timestamp}.categories')
    with open(categories_path, 'w') as f:
        json.dump(categories, f)
    logging.info(f"Categories saved: {categories_path}")
    
    return model_path, categories_path


def evaluate_clustering(model: AdaptiveClustering, test_data: torch.Tensor,
                        test_labels: np.ndarray, categories: list) -> dict:
    """
    クラスタリング結果を評価
    
    Args:
        model: 訓練済みAdaptive Clusteringモデル
        test_data: テストデータ
        test_labels: テストラベル（真のラベル）
        categories: カテゴリリスト
        
    Returns:
        評価メトリクスの辞書
    """
    device = next(model.parameters()).device
    test_data = test_data.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        # squeezeで1次元化し、scikit-leーン用にint配列へ
        predicted_clusters = (
            outputs.max(dim=1).indices
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
            .astype(int)
            .reshape(-1)
        )
        # 念のため負値を0にクリップ
        predicted_clusters = np.clip(predicted_clusters, 0, len(categories) - 1)
    
    # 埋め込み表現を取得（シルエットスコア用）
    embeddings_list = []
    for i, cluster_idx in enumerate(predicted_clusters):
        # numpy配列の要素をPythonのintに変換（.item()を使用）
        cluster_idx_int = cluster_idx.item() if hasattr(cluster_idx, 'item') else int(cluster_idx)
        embedding = model.sub_nets[cluster_idx_int].encoder(test_data[i:i+1])
        embeddings_list.append(embedding.squeeze().detach().cpu().numpy())
    
    embeddings = np.array(embeddings_list)
    
    # NaNや無限大の値をチェック・処理
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        logging.warning("Embeddings contain NaN or inf values. Replacing with 0.")
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 評価メトリクスを計算
    metrics = {}
    
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
    
    # 各クラスタのサイズ
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
    
    return metrics


def save_training_losses(epoch_losses: list, batch_losses: list,
                        results_dir: str = 'results', timestamp: Optional[str] = None) -> None:
    """
    訓練中のlossを保存
    
    Args:
        epoch_losses: エポックごとのloss情報
        batch_losses: バッチごとのloss情報
        results_dir: 結果保存ディレクトリ
        timestamp: タイムスタンプ（Noneの場合は自動生成）
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs(results_dir, exist_ok=True)
    
    # エポックごとのlossをCSV形式で保存
    epoch_df = pd.DataFrame(epoch_losses)
    epoch_loss_path = os.path.join(results_dir, f'clustering_loss_epoch_{timestamp}.csv')
    epoch_df.to_csv(epoch_loss_path, index=False)
    logging.info(f"Epoch losses saved: {epoch_loss_path}")
    
    # バッチごとのlossをCSV形式で保存
    batch_df = pd.DataFrame(batch_losses)
    batch_loss_path = os.path.join(results_dir, f'clustering_loss_batch_{timestamp}.csv')
    batch_df.to_csv(batch_loss_path, index=False)
    logging.info(f"Batch losses saved: {batch_loss_path}")
    
    # JSON形式でも保存（可読性のため）
    loss_data = {
        'epoch_losses': epoch_losses,
        'batch_losses': batch_losses
    }
    loss_json_path = os.path.join(results_dir, f'clustering_loss_{timestamp}.json')
    with open(loss_json_path, 'w') as f:
        json.dump(loss_data, f, indent=2, default=str)
    logging.info(f"Training losses (JSON) saved: {loss_json_path}")


def save_clustering_results(metrics: dict, predicted_clusters: np.ndarray,
                           test_labels: np.ndarray, categories: list,
                           results_dir: str = 'results',
                           timestamp: Optional[str] = None) -> None:
    """
    クラスタリング結果を保存
    
    Args:
        metrics: 評価メトリクス
        predicted_clusters: 予測されたクラスタ
        test_labels: 真のラベル
        categories: カテゴリリスト
        results_dir: 結果保存ディレクトリ
        timestamp: タイムスタンプ
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs(results_dir, exist_ok=True)
    
    # メトリクスを保存（JSON形式）
    metrics_json_path = os.path.join(results_dir, f'clustering_metrics_{timestamp}.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logging.info(f"Clustering metrics saved: {metrics_json_path}")
    
    # メトリクスを保存（pickle形式）
    metrics_pkl_path = os.path.join(results_dir, f'clustering_metrics_{timestamp}.pkl')
    with open(metrics_pkl_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    # クラスタ割り当てを保存
    assignments_df = pd.DataFrame({
        'true_label': [categories[label] for label in test_labels],
        'predicted_cluster': predicted_clusters,
        'true_label_idx': test_labels
    })
    assignments_path = os.path.join(results_dir, f'clustering_assignments_{timestamp}.csv')
    assignments_df.to_csv(assignments_path, index=False)
    logging.info(f"Clustering assignments saved: {assignments_path}")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='Train Adaptive Clustering model only')
    parser.add_argument('--dataset_path', type=str, default='dataset/CICIDS2017_improved',
                        help='Dataset path (default: dataset/CICIDS2017_improved)')
    parser.add_argument('--config', type=str, default='dataset_config.yaml',
                        help='Path to dataset_config.yaml (known/unknown labels). If missing, all labels are used for training.')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Results directory (default: results/)')
    parser.add_argument('--logs_dir', type=str, default='logs',
                        help='Logs directory (default: logs/)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: auto)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--early_stop_threshold', type=float, default=1.0,
                        help='Early stop threshold (default: 1.0)')
    parser.add_argument('--train_test_split', type=float, default=0.7,
                        help='Train/test split ratio (default: 0.7)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # ログ設定
    timestamp = setup_logging(args.logs_dir)
    logging.info("=" * 50)
    logging.info("Starting Adaptive Clustering training")
    logging.info(f"Dataset path: {args.dataset_path}")
    logging.info(f"Results directory: {args.results_dir}")
    logging.info(f"Logs directory: {args.logs_dir}")
    logging.info(f"Learning rate: {args.learning_rate}")
    logging.info(f"Number of epochs: {args.n_epochs}")
    logging.info(f"Early stop threshold: {args.early_stop_threshold}")
    logging.info("=" * 50)
    
    try:
        # 1. データ読み込み
        logging.info("Loading dataset...")
        df = load_cicids2017_dataset(args.dataset_path)
        logging.info(f"Dataset loaded: {len(df)} samples")
        
        # 2. 前処理
        logging.info("Preprocessing dataset...")
        df = preprocess_cicids2017(df)
        logging.info("Preprocessing completed")
        
        # 3. データ分割
        logging.info("Splitting dataset...")
        train_df, test_df = split_dataset(df, train_test_split=args.train_test_split, random_seed=args.random_seed)
        logging.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
        
        # 4. 既知/未知ラベルの決定（configがあれば適用）
        cfg = load_config(args.config)
        all_labels = set(df['Label'].unique().tolist())
        if cfg:
            known_labels = cfg.get('known_attacks', [])
            unknown_labels_cfg = cfg.get('unknown_attacks', [])
            known_set = set(known_labels) if known_labels else all_labels
            if unknown_labels_cfg:
                unknown_set = set(unknown_labels_cfg)
            else:
                unknown_set = all_labels - known_set
            logging.info(f"Known labels (config): {sorted(list(known_set))}")
            logging.info(f"Unknown labels (config or auto): {sorted(list(unknown_set))}")
        else:
            known_set = all_labels
            unknown_set = set()
            logging.info("Config not found or not provided. Using all labels as known.")
        
        # 既知データのみで学習
        train_known_df = train_df[train_df['Label'].isin(known_set)].copy()
        if train_known_df.empty:
            raise ValueError("No training samples for known labels. Check config.")
        
        categories_known = sorted(list(known_set))
        label_to_idx_known = {label: idx for idx, label in enumerate(categories_known)}
        
        # 評価用は全ラベルを整数マップ（未知も別ラベルとして扱う）
        categories_eval = sorted(list(all_labels))
        label_to_idx_eval = {label: idx for idx, label in enumerate(categories_eval)}
        
        # 5. データ準備（学習:既知のみ／評価:全て）
        logging.info("Preparing data for training...")
        train_df_ = train_known_df.drop(['Label'], axis=1)
        X_train = torch.FloatTensor(train_df_.values)
        y_train = torch.LongTensor(train_known_df['Label'].map(label_to_idx_known).values)
        
        test_df_ = test_df.drop(['Label'], axis=1)
        X_test = torch.FloatTensor(test_df_.values)
        y_test = np.array([label_to_idx_eval[lbl] for lbl in test_df['Label'].values])
        
        logging.info("Data preparation completed")
        
        # 6. Adaptive Clusteringモデルの訓練
        logging.info("Training Adaptive Clustering model...")
        model, epoch_losses, batch_losses = train_adaptive_clustering(
            X_train, y_train, categories_known,
            lr=args.learning_rate,
            n_epochs=args.n_epochs,
            early_stop_threshold=args.early_stop_threshold,
            batch_size=args.batch_size
        )
        logging.info("Adaptive Clustering training completed")
        
        # 訓練lossを保存
        logging.info("Saving training losses...")
        save_training_losses(epoch_losses, batch_losses, args.results_dir, timestamp)
        logging.info("Training losses saved")
        
        # 7. モデルの保存
        #    モデル自体は既知ラベルで学習しているが、
        #    可視化時にデータセット中の全ラベルを扱えるようにするため、
        #    カテゴリ情報としては評価用の全ラベル（categories_eval）を保存する。
        logging.info("Saving model...")
        save_model(model, categories_eval, args.results_dir, timestamp)
        logging.info("Model saved")
        
        # 8. クラスタリング評価
        logging.info("Evaluating clustering performance...")
        metrics = evaluate_clustering(model, X_test, y_test, categories_eval)
        
        # 結果の表示
        logging.info("=" * 50)
        logging.info("Clustering Evaluation Results:")
        logging.info(f"Adjusted Rand Index (ARI): {metrics.get('adjusted_rand_index', 0):.4f}")
        logging.info(f"Normalized Mutual Information (NMI): {metrics.get('normalized_mutual_info', 0):.4f}")
        if metrics.get('silhouette_score') is not None:
            logging.info(f"Silhouette Score: {metrics.get('silhouette_score', 0):.4f}")
        logging.info(f"Clustering Accuracy: {metrics.get('clustering_accuracy', 0):.4f}")
        logging.info(f"Cluster Sizes: {metrics.get('cluster_sizes', [])}")
        logging.info(f"Cluster Purities: {[f'{p:.4f}' for p in metrics.get('cluster_purities', [])]}")
        logging.info("=" * 50)
        
        # 9. 予測クラスタを取得（保存用）
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            outputs = model(X_test.to(device))
            predicted_clusters = (
                outputs.max(dim=1).indices
                .squeeze(-1)
                .detach()
                .cpu()
                .numpy()
                .astype(int)
                .reshape(-1)
            )
            predicted_clusters = np.clip(predicted_clusters, 0, len(categories_eval) - 1)
        
        # 10. 結果の保存
        logging.info("Saving clustering results...")
        save_clustering_results(metrics, predicted_clusters, y_test, categories_eval,
                               args.results_dir, timestamp)
        logging.info("Results saved")
        
        logging.info("=" * 50)
        logging.info("Completed successfully")
        logging.info("=" * 50)
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

