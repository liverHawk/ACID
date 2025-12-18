#!/usr/bin/env python
"""
CICIDS2017_improved データセットを用いた
既知 / 未知攻撃クラスタリング評価スクリプト

- dataset_config.yaml で既知 / 未知攻撃ラベルを定義
- 既知攻撃のみでクラスタリングモデルを学習
- 評価は既知 + 未知を含む全ラベルで実施
- 結果はタイムスタンプ付きディレクトリに保存
"""
import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from lib.utils.preprocessing import load_cicids2017_dataset, preprocess_cicids2017, split_dataset
from lib.utils.pipeline import train_adaptive_clustering


def setup_logging(log_dir: str = "logs") -> str:
    """
    ログ設定

    Args:
        log_dir: ログディレクトリ

    Returns:
        タイムスタンプ文字列
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"known_unknown_output_{timestamp}.txt")
    error_file = os.path.join(log_dir, f"known_unknown_error_{timestamp}.txt")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    error_handler = logging.FileHandler(error_file)
    error_handler.setLevel(logging.ERROR)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    return timestamp


def load_config(config_path: str) -> Tuple[List[str], List[str]]:
    """
    既知 / 未知攻撃ラベル設定を読み込む

    Args:
        config_path: dataset_config.yaml へのパス

    Returns:
        (known_attacks, unknown_attacks) のタプル
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    known = cfg.get("known_attacks", [])
    unknown = cfg.get("unknown_attacks", [])
    return known, unknown


def save_metrics_and_assignments(
    metrics: Dict,
    predicted_clusters: np.ndarray,
    y_true_labels: List[str],
    categories_eval: List[str],
    results_dir: str,
    timestamp: str,
) -> None:
    """
    クラスタリング評価結果および割り当て情報を保存

    Args:
        metrics: 評価指標の辞書
        predicted_clusters: 予測クラスタインデックス配列
        y_true_labels: 真のラベル（文字列）のリスト
        categories_eval: 評価時に用いるカテゴリ一覧（UNKNOWN を含む）
        results_dir: 結果保存ディレクトリ（タイムスタンプディレクトリ）
        timestamp: タイムスタンプ文字列
    """
    os.makedirs(results_dir, exist_ok=True)

    metrics_json_path = os.path.join(results_dir, f"known_unknown_metrics_{timestamp}.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logging.info(f"Metrics saved: {metrics_json_path}")

    assignments_df = pd.DataFrame(
        {
            "true_label": y_true_labels,
            "predicted_cluster": predicted_clusters,
        }
    )
    assignments_path = os.path.join(results_dir, f"known_unknown_assignments_{timestamp}.csv")
    assignments_df.to_csv(assignments_path, index=False)
    logging.info(f"Assignments saved: {assignments_path}")

    # 混同行列（真のラベル vs 予測クラスタ）
    y_series = pd.Series(y_true_labels, name="Actual")
    pred_series = pd.Series(predicted_clusters, name="PredictedCluster")
    cm_counts = pd.crosstab(y_series, pred_series, rownames=["Actual"], colnames=["PredictedCluster"], dropna=False)
    cm_path = os.path.join(results_dir, f"known_unknown_confusion_counts_{timestamp}.csv")
    cm_counts.to_csv(cm_path)
    logging.info(f"Confusion matrix saved: {cm_path}")


def save_training_losses(
    epoch_losses: List[dict],
    batch_losses: List[dict],
    results_dir: str,
    timestamp: str,
) -> None:
    """
    学習中の loss を CSV / JSON 形式で保存

    Args:
        epoch_losses: エポックごとの loss 情報
        batch_losses: バッチごとの loss 情報
        results_dir: 結果保存ディレクトリ（タイムスタンプディレクトリ）
        timestamp: タイムスタンプ文字列
    """
    os.makedirs(results_dir, exist_ok=True)
    epoch_df = pd.DataFrame(epoch_losses)
    batch_df = pd.DataFrame(batch_losses)
    epoch_path = os.path.join(results_dir, f"training_loss_epoch_{timestamp}.csv")
    batch_path = os.path.join(results_dir, f"training_loss_batch_{timestamp}.csv")
    epoch_df.to_csv(epoch_path, index=False)
    batch_df.to_csv(batch_path, index=False)
    loss_json_path = os.path.join(results_dir, f"training_loss_{timestamp}.json")
    with open(loss_json_path, "w") as f:
        json.dump({"epoch_losses": epoch_losses, "batch_losses": batch_losses}, f, indent=2, default=str)
    logging.info(f"Losses saved: {epoch_path}, {batch_path}, {loss_json_path}")


def plot_overall_scatter(
    X: torch.Tensor,
    results_dir: str,
    timestamp: str,
    max_points: int = 5000,
    use_pca: bool = True,
    use_tsne: bool = True,
    metrics: Dict = None,
) -> None:
    """
    全体散布図（ラベル色分けなし）を保存（PCA / t-SNE）
    評価指標をテキストボックスで表示
    """
    os.makedirs(results_dir, exist_ok=True)
    X_np = X.cpu().numpy()
    if len(X_np) > max_points:
        idx = np.random.choice(len(X_np), max_points, replace=False)
        X_np = X_np[idx]

    # 評価指標のテキストを生成
    metrics_text = ""
    if metrics:
        metrics_text = "Evaluation Metrics:\n"
        if metrics.get("adjusted_rand_index") is not None:
            metrics_text += f"ARI: {metrics['adjusted_rand_index']:.4f}\n"
        if metrics.get("normalized_mutual_info") is not None:
            metrics_text += f"NMI: {metrics['normalized_mutual_info']:.4f}\n"
        if metrics.get("silhouette_score") is not None:
            metrics_text += f"Silhouette: {metrics['silhouette_score']:.4f}\n"
        if metrics.get("cluster_purities"):
            avg_purity = np.mean(metrics["cluster_purities"])
            metrics_text += f"Avg Purity: {avg_purity:.4f}\n"

    # PCA
    if use_pca and X_np.shape[1] >= 2:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        emb_pca = pca.fit_transform(X_np)
        df_pca = pd.DataFrame({"pc1": emb_pca[:, 0], "pc2": emb_pca[:, 1]})

        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df_pca, x="pc1", y="pc2", s=12, alpha=0.6, edgecolor="none")
        plt.title(f"Overall Scatter (PCA) – Var: {pca.explained_variance_ratio_.sum():.2%}")
        if metrics_text:
            plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        out_path = os.path.join(results_dir, f"overall_scatter_pca_{timestamp}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Overall PCA scatter saved: {out_path}")

    # t-SNE
    if use_tsne and X_np.shape[1] >= 2:
        from sklearn.manifold import TSNE

        logging.info("Computing t-SNE for overall scatter (may take time)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_np) - 1))
        emb_tsne = tsne.fit_transform(X_np)
        df_tsne = pd.DataFrame({"x1": emb_tsne[:, 0], "x2": emb_tsne[:, 1]})

        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df_tsne, x="x1", y="x2", s=12, alpha=0.6, edgecolor="none")
        plt.title("Overall Scatter (t-SNE)")
        if metrics_text:
            plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        out_path = os.path.join(results_dir, f"overall_scatter_tsne_{timestamp}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Overall t-SNE scatter saved: {out_path}")


def plot_true_label_scatter(
    X: torch.Tensor,
    y_true_labels: List[str],
    results_dir: str,
    timestamp: str,
    max_points: int = 5000,
    use_pca: bool = True,
    use_tsne: bool = True,
    metrics: Dict = None,
) -> None:
    """
    真のラベルごとに色分けした散布図を保存（PCA / t-SNE）
    未知攻撃ラベルも個別に凡例に出す
    評価指標をテキストボックスで表示
    """
    os.makedirs(results_dir, exist_ok=True)
    X_np = X.cpu().numpy()
    labels = np.array(y_true_labels)

    # サンプリング（多すぎる場合）
    if len(X_np) > max_points:
        idx = np.random.choice(len(X_np), max_points, replace=False)
        X_np = X_np[idx]
        labels = labels[idx]

    # 評価指標のテキストを生成
    metrics_text = ""
    if metrics:
        metrics_text = "Evaluation Metrics:\n"
        if metrics.get("adjusted_rand_index") is not None:
            metrics_text += f"ARI: {metrics['adjusted_rand_index']:.4f}\n"
        if metrics.get("normalized_mutual_info") is not None:
            metrics_text += f"NMI: {metrics['normalized_mutual_info']:.4f}\n"
        if metrics.get("silhouette_score") is not None:
            metrics_text += f"Silhouette: {metrics['silhouette_score']:.4f}\n"
        if metrics.get("cluster_purities"):
            avg_purity = np.mean(metrics["cluster_purities"])
            metrics_text += f"Avg Purity: {avg_purity:.4f}\n"

    # PCA
    if use_pca and X_np.shape[1] >= 2:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        emb_pca = pca.fit_transform(X_np)
        df_pca = pd.DataFrame({"pc1": emb_pca[:, 0], "pc2": emb_pca[:, 1], "label": labels})

        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=df_pca, x="pc1", y="pc2", hue="label", s=25, alpha=0.7, edgecolor="none", palette="tab20")
        plt.title(f"True Label Scatter (PCA) – Var: {pca.explained_variance_ratio_.sum():.2%}")
        if metrics_text:
            plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        out_path = os.path.join(results_dir, f"true_label_scatter_pca_{timestamp}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"PCA scatter saved: {out_path}")

    # t-SNE（特徴次元が高い場合のみ実施）
    if use_tsne and X_np.shape[1] >= 2:
        from sklearn.manifold import TSNE

        logging.info("Computing t-SNE for true-label scatter (may take time)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_np) - 1))
        emb_tsne = tsne.fit_transform(X_np)
        df_tsne = pd.DataFrame({"x1": emb_tsne[:, 0], "x2": emb_tsne[:, 1], "label": labels})

        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=df_tsne, x="x1", y="x2", hue="label", s=25, alpha=0.7, edgecolor="none", palette="tab20")
        plt.title("True Label Scatter (t-SNE)")
        if metrics_text:
            plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        out_path = os.path.join(results_dir, f"true_label_scatter_tsne_{timestamp}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"t-SNE scatter saved: {out_path}")


def plot_predicted_cluster_scatter(
    X: torch.Tensor,
    predicted_clusters: np.ndarray,
    results_dir: str,
    timestamp: str,
    max_points: int = 5000,
    use_pca: bool = True,
    use_tsne: bool = True,
    metrics: Dict = None,
    categories_known: List[str] = None,
) -> None:
    """
    予測クラスタごとに色分けした散布図を保存（PCA / t-SNE）
    評価指標をテキストボックスで表示
    """
    os.makedirs(results_dir, exist_ok=True)
    X_np = X.cpu().numpy()
    clusters = predicted_clusters.copy()

    # サンプリング（多すぎる場合）
    if len(X_np) > max_points:
        idx = np.random.choice(len(X_np), max_points, replace=False)
        X_np = X_np[idx]
        clusters = clusters[idx]

    # クラスタラベルの生成
    if categories_known:
        cluster_labels = [f"Cluster {i} ({categories_known[i]})" if i < len(categories_known) else f"Cluster {i}" 
                         for i in range(len(categories_known))]
    else:
        cluster_labels = [f"Cluster {i}" for i in range(len(np.unique(clusters)))]
    
    df_clusters = pd.DataFrame({"cluster": clusters})
    df_clusters["cluster_label"] = df_clusters["cluster"].apply(lambda x: cluster_labels[x] if x < len(cluster_labels) else f"Cluster {x}")

    # 評価指標のテキストを生成
    metrics_text = ""
    if metrics:
        metrics_text = "Evaluation Metrics:\n"
        if metrics.get("adjusted_rand_index") is not None:
            metrics_text += f"ARI: {metrics['adjusted_rand_index']:.4f}\n"
        if metrics.get("normalized_mutual_info") is not None:
            metrics_text += f"NMI: {metrics['normalized_mutual_info']:.4f}\n"
        if metrics.get("silhouette_score") is not None:
            metrics_text += f"Silhouette: {metrics['silhouette_score']:.4f}\n"
        if metrics.get("cluster_purities"):
            avg_purity = np.mean(metrics["cluster_purities"])
            metrics_text += f"Avg Purity: {avg_purity:.4f}\n"
        if metrics.get("cluster_sizes"):
            metrics_text += f"Cluster Sizes: {metrics['cluster_sizes']}\n"

    # PCA
    if use_pca and X_np.shape[1] >= 2:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        emb_pca = pca.fit_transform(X_np)
        df_pca = pd.DataFrame({"pc1": emb_pca[:, 0], "pc2": emb_pca[:, 1], "cluster": df_clusters["cluster_label"]})

        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=df_pca, x="pc1", y="pc2", hue="cluster", s=25, alpha=0.7, edgecolor="none", palette="tab20")
        plt.title(f"Predicted Cluster Scatter (PCA) – Var: {pca.explained_variance_ratio_.sum():.2%}")
        if metrics_text:
            plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        out_path = os.path.join(results_dir, f"predicted_cluster_scatter_pca_{timestamp}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Predicted cluster PCA scatter saved: {out_path}")

    # t-SNE
    if use_tsne and X_np.shape[1] >= 2:
        from sklearn.manifold import TSNE

        logging.info("Computing t-SNE for predicted cluster scatter (may take time)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_np) - 1))
        emb_tsne = tsne.fit_transform(X_np)
        df_tsne = pd.DataFrame({"x1": emb_tsne[:, 0], "x2": emb_tsne[:, 1], "cluster": df_clusters["cluster_label"]})

        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=df_tsne, x="x1", y="x2", hue="cluster", s=25, alpha=0.7, edgecolor="none", palette="tab20")
        plt.title("Predicted Cluster Scatter (t-SNE)")
        if metrics_text:
            plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        out_path = os.path.join(results_dir, f"predicted_cluster_scatter_tsne_{timestamp}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Predicted cluster t-SNE scatter saved: {out_path}")


def main():
    """メイン実行関数（既知 / 未知クラスタリング実験）"""
    parser = argparse.ArgumentParser(
        description="Known/Unknown clustering experiment on CICIDS2017_improved"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="dataset_config.yaml",
        help="Path to dataset_config.yaml (default: dataset_config.yaml)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset/CICIDS2017_improved",
        help="Dataset path (default: dataset/CICIDS2017_improved)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Results directory (default: results/)",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="logs",
        help="Logs directory (default: logs/)",
    )
    parser.add_argument(
        "--train_test_split",
        type=float,
        default=0.7,
        help="Train/test split ratio (default: 0.7)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)",
    )
    parser.add_argument(
        "--early_stop_threshold",
        type=float,
        default=1.0,
        help="Early stop threshold (default: 1.0)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (default: auto)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug outputs (e.g., training loss files)",
    )
    args = parser.parse_args()

    # ログ設定
    timestamp = setup_logging(args.logs_dir)

    try:
        logging.info("=" * 50)
        logging.info("Starting known/unknown clustering experiment")
        logging.info(f"Config path: {args.config}")
        logging.info(f"Dataset path: {args.dataset_path}")
        logging.info(f"Results directory: {args.results_dir}")
        logging.info(f"Logs directory: {args.logs_dir}")
        logging.info("=" * 50)

        # 1. 設定読み込み
        logging.info("1. Loading config...")
        known_labels, unknown_labels_cfg = load_config(args.config)
        known_set = set(known_labels)
        logging.info(f"Known labels (from config): {known_labels}")
        logging.info(f"Unknown labels (from config): {unknown_labels_cfg}")

        # 2. データ読み込み
        logging.info("2. Loading dataset...")
        df = load_cicids2017_dataset(args.dataset_path)
        logging.info(f"Dataset loaded: {len(df)} samples")

        # 3. 前処理
        logging.info("3. Preprocessing dataset...")
        df = preprocess_cicids2017(df)
        logging.info("Preprocessing completed")

        # 4. データ分割
        logging.info("4. Splitting dataset...")
        train_df, test_df = split_dataset(df, train_test_split=args.train_test_split, random_seed=args.random_seed)
        logging.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

        # 5. 既知 / 未知ラベル集合の確定
        all_labels = set(df["Label"].unique().tolist())
        if not unknown_labels_cfg:
            unknown_set = all_labels - known_set
        else:
            unknown_set = set(unknown_labels_cfg)
        logging.info(f"Auto-detected unknown labels: {sorted(list(unknown_set))}")

        # 6. 既知データのみで学習
        train_known_df = train_df[train_df["Label"].isin(known_set)].copy()
        if train_known_df.empty:
            raise ValueError("No training samples for known labels. Check dataset_config.yaml.")

        categories_known = sorted(list(known_set))
        label_to_idx_known = {label: idx for idx, label in enumerate(categories_known)}

        X_train = torch.FloatTensor(train_known_df.drop(["Label"], axis=1).values)
        y_train = torch.LongTensor(train_known_df["Label"].map(label_to_idx_known).values)

        # 7. テスト用データ（全ラベル）
        X_test = torch.FloatTensor(test_df.drop(["Label"], axis=1).values)
        y_test_labels = test_df["Label"].values

        # 8. 評価用ラベルマップ（UNKNOWN を追加）
        categories_eval = categories_known + ["UNKNOWN"]
        label_to_idx_eval = {label: idx for idx, label in enumerate(categories_known)}
        unknown_idx = len(categories_known)

        def map_label_eval(label: str) -> int:
            return label_to_idx_eval.get(label, unknown_idx)

        y_test_idx = np.array([map_label_eval(lbl) for lbl in y_test_labels])

        # 9. Adaptive Clustering モデルの学習（既知攻撃のみ）
        logging.info("5. Training Adaptive Clustering on known labels...")
        model, epoch_losses, batch_losses = train_adaptive_clustering(
            X_train,
            y_train,
            categories_known,
            lr=args.learning_rate,
            n_epochs=args.n_epochs,
            early_stop_threshold=args.early_stop_threshold,
            batch_size=args.batch_size,
            timestamp=timestamp,
            loss_log_dir=args.logs_dir,
        )
        logging.info("Adaptive Clustering training completed")

        # 10. タイムスタンプごとのディレクトリを作成（loss 保存の前に作成）
        timestamp_dir = os.path.join(args.results_dir, timestamp)
        os.makedirs(timestamp_dir, exist_ok=True)
        logging.info(f"Results will be saved to: {timestamp_dir}")

        # 11. loss 保存（debug 時のみ）
        if args.debug:
            save_training_losses(epoch_losses, batch_losses, timestamp_dir, timestamp)

        # 12. テストデータに対する予測
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            outputs = model(X_test.to(device))
            predicted_clusters = (
                outputs.max(dim=1)
                .indices.squeeze(-1)
                .detach()
                .cpu()
                .numpy()
                .astype(int)
                .reshape(-1)
            )
            predicted_clusters = np.clip(predicted_clusters, 0, len(categories_known) - 1)

        # 13. メトリクス計算（クラスタリング系）
        metrics = {}
        metrics["adjusted_rand_index"] = float(adjusted_rand_score(y_test_idx, predicted_clusters))
        metrics["normalized_mutual_info"] = float(normalized_mutual_info_score(y_test_idx, predicted_clusters))
        if X_test.shape[1] >= 2:
            try:
                # 評価はテストデータ全量を使用（サブサンプリングなし）
                emb = X_test.numpy()
                metrics["silhouette_score"] = float(silhouette_score(emb, predicted_clusters))
            except Exception as e:
                logging.warning(f"Silhouette score failed: {e}")
                metrics["silhouette_score"] = None

        # クラスタサイズ
        cluster_sizes = np.bincount(predicted_clusters, minlength=len(categories_known))
        metrics["cluster_sizes"] = cluster_sizes.tolist()

        # 純度（各クラスタで最多ラベルの割合）
        cluster_purities = []
        for c in range(len(categories_known)):
            mask = predicted_clusters == c
            if mask.sum() == 0:
                cluster_purities.append(0.0)
                continue
            labels_in_cluster = y_test_idx[mask]
            most_common = np.bincount(labels_in_cluster).max()
            cluster_purities.append(most_common / len(labels_in_cluster))
        metrics["cluster_purities"] = cluster_purities

        # 14. メトリクス・割り当ての保存
        save_metrics_and_assignments(
            metrics, predicted_clusters, y_test_labels.tolist(), categories_eval, timestamp_dir, timestamp
        )

        # 15. 散布図の保存: 全体 / 真のラベル別 / 予測クラスタ別
        plot_overall_scatter(
            X_test,
            timestamp_dir,
            timestamp,
            max_points=5000,
            use_pca=True,
            use_tsne=True,
            metrics=metrics,
        )
        plot_true_label_scatter(
            X_test,
            y_test_labels.tolist(),
            timestamp_dir,
            timestamp,
            max_points=5000,
            use_pca=True,
            use_tsne=True,
            metrics=metrics,
        )
        plot_predicted_cluster_scatter(
            X_test,
            predicted_clusters,
            timestamp_dir,
            timestamp,
            max_points=5000,
            use_pca=True,
            use_tsne=True,
            metrics=metrics,
            categories_known=categories_known,
        )

        logging.info("=" * 50)
        logging.info("Known/unknown clustering experiment completed successfully")
        logging.info("=" * 50)

    except Exception as e:
        logging.error(f"Error occurred: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

