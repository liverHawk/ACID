#!/usr/bin/env python
"""
CICIDS2017_improvedデータセットを使用したIDS訓練・評価スクリプト

ログとモデル保存機能付き
"""
import os
import argparse
import logging
from datetime import datetime
import json
import pickle
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
import torch
import yaml

# libモジュールをインポート
from lib.models.network import AdaptiveClustering
from lib.models.RandomForest import RandomForest
from lib.utils.preprocessing import load_cicids2017_dataset, preprocess_cicids2017, split_dataset
from lib.utils.pipeline import train_adaptive_clustering, evaluate_model
from lib.utils.misc import extend_dataset


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


def setup_logging(log_dir: str = 'logs') -> str:
    """
    ログ設定
    
    Args:
        log_dir: ログディレクトリ
        
    Returns:
        タイムスタンプ文字列
    """
    # タイムスタンプ付きログファイル名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'output_{timestamp}.txt')
    error_file = os.path.join(log_dir, f'error_{timestamp}.txt')
    
    # ログ設定
    # FileHandlerはlevel引数を受け取らないので、setLevel()を使用
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    error_handler = logging.FileHandler(error_file)
    error_handler.setLevel(logging.ERROR)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # フォーマッターを設定
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # ルートロガーを設定
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)
    
    return timestamp


def save_model(model: AdaptiveClustering, categories: list, rf_model: Optional[RandomForest],
               results_dir: str = 'results', timestamp: Optional[str] = None) -> tuple:
    """
    モデルとカテゴリ情報を保存
    
    Args:
        model: 訓練済みAdaptive Clusteringモデル
        categories: カテゴリリスト
        rf_model: 訓練済みRandom Forestモデル（オプション）
        results_dir: 結果保存ディレクトリ
        timestamp: タイムスタンプ（Noneの場合は自動生成）
        
    Returns:
        (モデルパス, カテゴリパス, RFモデルパス)のタプル
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Adaptive Clusteringモデルを保存
    model_path = os.path.join(results_dir, f'trained_model_{timestamp}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved: {model_path}")
    
    # カテゴリ情報を保存
    categories_path = os.path.join(results_dir, f'trained_model_{timestamp}.categories')
    with open(categories_path, 'w') as f:
        json.dump(categories, f)
    logging.info(f"Categories saved: {categories_path}")
    
    # Random Forestモデルを保存（存在する場合）
    rf_model_path = None
    if rf_model is not None:
        rf_model_path = os.path.join(results_dir, f'rf_model_{timestamp}.pkl')
        with open(rf_model_path, 'wb') as f:
            pickle.dump(rf_model, f)
        logging.info(f"Random Forest model saved: {rf_model_path}")
    
    return model_path, categories_path, rf_model_path


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
    epoch_loss_path = os.path.join(results_dir, f'training_loss_epoch_{timestamp}.csv')
    epoch_df.to_csv(epoch_loss_path, index=False)
    logging.info(f"Epoch losses saved: {epoch_loss_path}")
    
    # バッチごとのlossをCSV形式で保存
    batch_df = pd.DataFrame(batch_losses)
    batch_loss_path = os.path.join(results_dir, f'training_loss_batch_{timestamp}.csv')
    batch_df.to_csv(batch_loss_path, index=False)
    logging.info(f"Batch losses saved: {batch_loss_path}")
    
    # JSON形式でも保存（可読性のため）
    loss_data = {
        'epoch_losses': epoch_losses,
        'batch_losses': batch_losses
    }
    loss_json_path = os.path.join(results_dir, f'training_loss_{timestamp}.json')
    with open(loss_json_path, 'w') as f:
        json.dump(loss_data, f, indent=2, default=str)
    logging.info(f"Training losses (JSON) saved: {loss_json_path}")


def save_results(confusion_matrix_normalized: pd.DataFrame, confusion_matrix_counts: pd.DataFrame,
                 metrics: dict, feature_importance: Optional[list],
                 results_dir: str = 'results', timestamp: Optional[str] = None) -> None:
    """
    評価結果を保存
    
    Args:
        confusion_matrix_normalized: 正規化された混同行列データフレーム（パーセンテージ）
        confusion_matrix_counts: 生の数値の混同行列データフレーム（カウント）
        metrics: メトリクス辞書
        feature_importance: 特徴量重要度リスト
        results_dir: 結果保存ディレクトリ
        timestamp: タイムスタンプ（Noneの場合は自動生成）
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs(results_dir, exist_ok=True)
    
    # 正規化された混同行列を保存（パーセンテージ）
    cm_normalized_path = os.path.join(results_dir, f'rf_confusion_matrix_normalized_{timestamp}.csv')
    confusion_matrix_normalized.to_csv(cm_normalized_path, index=True)
    logging.info(f"Normalized confusion matrix saved: {cm_normalized_path}")
    
    # 生の数値の混同行列を保存（カウント、ラベル名付き）
    cm_counts_path = os.path.join(results_dir, f'rf_confusion_matrix_counts_{timestamp}.csv')
    confusion_matrix_counts.to_csv(cm_counts_path, index=True)
    logging.info(f"Confusion matrix (counts) saved: {cm_counts_path}")
    
    # メトリクスを保存（pickle形式）
    metrics_pkl_path = os.path.join(results_dir, f'metrics_{timestamp}.pkl')
    with open(metrics_pkl_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    # メトリクスを保存（JSON形式、可読性のため）
    metrics_json_path = os.path.join(results_dir, f'metrics_{timestamp}.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logging.info(f"Metrics saved: {metrics_json_path}")
    
    # 特徴量重要度を保存
    if feature_importance is not None:
        fi_path = os.path.join(results_dir, f'rf_feature_importance_{timestamp}.pkl')
        with open(fi_path, 'wb') as f:
            pickle.dump(feature_importance, f)
        logging.info(f"Feature importance saved: {fi_path}")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='IDS training and evaluation on CICIDS2017_improved')
    parser.add_argument('--config', type=str, default='dataset_config.yaml',
                        help='Path to dataset_config.yaml (default: dataset_config.yaml)')
    parser.add_argument('--dataset_path', type=str, default='dataset/CICIDS2017_improved',
                        help='Dataset path (default: dataset/CICIDS2017_improved)')
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
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use: auto / cpu / cuda / mps (default: auto)')
    
    args = parser.parse_args()
    
    # ログ設定
    timestamp = setup_logging(args.logs_dir)

    # タイムスタンプごとの結果ディレクトリを作成
    timestamp_dir = os.path.join(args.results_dir, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)

    logging.info("=" * 50)
    logging.info("Starting IDS training and evaluation")
    logging.info(f"Config path: {args.config}")
    logging.info(f"Dataset path: {args.dataset_path}")
    logging.info(f"Results base directory: {args.results_dir}")
    logging.info(f"Timestamped results directory: {timestamp_dir}")
    logging.info(f"Logs directory: {args.logs_dir}")
    logging.info(f"Device preference: {args.device}")
    logging.info("=" * 50)
    
    try:
        # 0. 設定読み込み（既知 / 未知ラベル）
        logging.info("Loading known/unknown label config...")
        known_labels, unknown_labels_cfg = load_config(args.config)
        known_set = set(known_labels)
        logging.info(f"Known labels (from config): {known_labels}")
        logging.info(f"Unknown labels (from config): {unknown_labels_cfg}")
        
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
        train_df, test_df = split_dataset(df, train_test_split=0.7, random_seed=42)
        logging.info(f"Train samples (all labels): {len(train_df)}, Test samples (all labels): {len(test_df)}")
        
        # 4. 既知 / 未知ラベル集合の確定
        all_labels = set(df["Label"].unique().tolist())
        if not unknown_labels_cfg:
            unknown_set = all_labels - known_set
        else:
            unknown_set = set(unknown_labels_cfg)
        logging.info(f"Auto-detected unknown labels: {sorted(list(unknown_set))}")
        
        # 5. 既知ラベルのみを用いた訓練データ作成
        train_df_known = train_df[train_df["Label"].isin(known_set)].copy()
        if train_df_known.empty:
            raise ValueError("No training samples for known labels. Check dataset_config.yaml.")
        
        dropped_train = len(train_df) - len(train_df_known)
        logging.info(f"Train samples (known only): {len(train_df_known)} (dropped {dropped_train} unknown samples)")
        
        # 評価対象として既知ラベルのみを使用
        test_df_known = test_df[test_df["Label"].isin(known_set)].copy()
        dropped_test = len(test_df) - len(test_df_known)
        logging.info(f"Test samples (known only): {len(test_df_known)} (dropped {dropped_test} unknown samples)")
        
        # 6. カテゴリの抽出（既知ラベルのみ）
        categories = sorted(list(known_set))
        logging.info(f"Categories (known only): {categories}")
        
        # 7. データ準備（既知ラベルのみ）
        logging.info("Preparing data for training...")
        train_df_ = train_df_known.drop(['Label'], axis=1)
        X = torch.FloatTensor(train_df_.values)
        
        # ラベルのインデックス変換
        label_to_idx = {label: idx for idx, label in enumerate(categories)}
        y = train_df_known['Label'].map(label_to_idx).values
        y = torch.LongTensor(y)
        
        # extend_dataset では全データを入力しつつ、後続で既知ラベルのみに絞る
        cats = df['Label'].copy()
        df_ = df.drop(['Label'], axis=1)
        logging.info("Data preparation completed")
        
        # 6. Adaptive Clusteringモデルの訓練
        logging.info("Training Adaptive Clustering model...")
        model, epoch_losses, batch_losses = train_adaptive_clustering(
            X, y, categories,
            lr=args.learning_rate,
            n_epochs=args.n_epochs,
            early_stop_threshold=args.early_stop_threshold,
            batch_size=args.batch_size,
            device_preference=args.device,
        )
        logging.info("Adaptive Clustering training completed")
        
        # 訓練lossを保存（タイムスタンプディレクトリ配下）
        logging.info("Saving training losses...")
        save_training_losses(epoch_losses, batch_losses, timestamp_dir, timestamp)
        logging.info("Training losses saved")
        
        # 7. 拡張データセットの作成
        logging.info("Creating extended dataset...")
        rf_features, cols = extend_dataset(model, df_, cats, label_tag='Label')
        logging.info("Extended dataset created")
        
        # 8. 拡張データセットの分割（既知ラベルのみを使用）
        new_df = pd.DataFrame(data=rf_features, columns=cols)
        
        # 既知ラベルのみを残す
        if 'Label' not in new_df.columns:
            raise ValueError("Extended dataset does not contain 'Label' column.")
        before_filter = len(new_df)
        new_df = new_df[new_df["Label"].isin(categories)].copy()
        after_filter = len(new_df)
        logging.info(f"Extended dataset size (known only): {after_filter} (dropped {before_filter - after_filter} unknown samples)")
        
        # 無限大やNaNの値をチェック・処理
        # 数値列のみを処理（Label列は除外）
        numeric_cols = new_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = new_df[col].values
            if np.isinf(col_data).any() or np.isnan(col_data).any():
                logging.warning(f"Column {col} contains inf or NaN values. Replacing with 0.")
                new_df[col] = new_df[col].replace([np.inf, -np.inf], 0)
                new_df[col] = new_df[col].fillna(0)
        
        # 再度チェック
        numeric_data = new_df[numeric_cols].values
        if np.isinf(numeric_data).any() or np.isnan(numeric_data).any():
            logging.error("Data still contains inf or NaN values after cleaning")
            raise ValueError("Data contains invalid values (inf or NaN)")
        
        train_test_split = 0.7
        np.random.seed(42)
        msk = np.random.rand(len(new_df)) < train_test_split
        train_df_extended = new_df[msk]
        test_df_extended = new_df[~msk]
        
        df_extended_ = train_df_extended.drop(['Label'], axis=1)
        X_extended = df_extended_.values.astype(np.float32)
        y_extended = train_df_extended['Label'].map(label_to_idx).values
        
        # 9. Random Forestモデルの訓練
        logging.info("Training Random Forest model...")
        rf_model = RandomForest(n_estimators=200)
        rf_model.fit(X_extended, y_extended)
        model.classifier = rf_model
        logging.info("Random Forest training completed")
        
        # 10. モデルの保存（タイムスタンプディレクトリ配下）
        logging.info("Saving models...")
        save_model(model, categories, rf_model, timestamp_dir, timestamp)
        logging.info("Models saved")
        
        # 11. 評価
        logging.info("Evaluating model...")
        test_df_extended_ = test_df_extended.drop(['Label'], axis=1)
        x_test = test_df_extended_.values
        y_hat = rf_model.predict(x_test)
        
        y_actual_ = test_df_extended['Label'].map(label_to_idx).values
        
        # 評価メトリクスの計算
        confusion_matrix_normalized, confusion_matrix_counts, metrics_data, feature_importance = evaluate_model(
            model, test_df_extended_, categories, y_actual_, y_hat
        )
        
        # 結果の表示
        logging.info("=" * 50)
        logging.info("Evaluation Results:")
        logging.info(f"Accuracy: {metrics_data.get('accuracy', 0):.4f}")
        logging.info(f"Precision: {metrics_data.get('precision', 0):.4f}")
        logging.info(f"Recall: {metrics_data.get('DR/Recall', 0):.4f}")
        logging.info(f"F1 Score: {metrics_data.get('F1 SCORE', 0):.4f}")
        logging.info(f"False Alarm Rate: {metrics_data.get('FAR', 0):.4f}")
        logging.info("=" * 50)
        
        # 12. 結果の保存（タイムスタンプディレクトリ配下）
        logging.info("Saving results...")
        save_results(
            confusion_matrix_normalized,
            confusion_matrix_counts,
            metrics_data,
            feature_importance,
            timestamp_dir,
            timestamp,
        )
        logging.info("Results saved")
        
        logging.info("=" * 50)
        logging.info("Completed successfully")
        logging.info("=" * 50)
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
