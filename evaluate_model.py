#!/usr/bin/env python
"""
既存の訓練済みモデルを使用したIDS評価スクリプト

ラベル名付き混同行列を生成します
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
from lib.models.RandomForest import RandomForest
from lib.utils.preprocessing import load_cicids2017_dataset, preprocess_cicids2017, split_dataset
from lib.utils.pipeline import evaluate_model
from lib.utils.misc import extend_dataset


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
    
    log_file = os.path.join(log_dir, f'evaluate_output_{timestamp}.txt')
    error_file = os.path.join(log_dir, f'evaluate_error_{timestamp}.txt')
    
    # ログ設定
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


def load_trained_model(model_path: str, categories_path: str, rf_model_path: Optional[str] = None):
    """
    訓練済みモデルを読み込む
    
    Args:
        model_path: Adaptive Clusteringモデルのパス
        categories_path: カテゴリ情報のパス
        rf_model_path: Random Forestモデルのパス（オプション）
        
    Returns:
        (model, categories, rf_model)のタプル
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
    
    # Random Forestモデルを読み込み（存在する場合）
    rf_model = None
    if rf_model_path and os.path.exists(rf_model_path):
        with open(rf_model_path, 'rb') as f:
            rf_model = pickle.load(f)
        logging.info(f"Random Forest model loaded: {rf_model_path}")
        model.classifier = rf_model
    
    return model, categories, rf_model


def plot_confusion_matrix(confusion_matrix_normalized: pd.DataFrame,
                          confusion_matrix_counts: pd.DataFrame,
                          results_dir: str = 'results',
                          timestamp: Optional[str] = None) -> None:
    """
    混同行列をプロットして保存
    
    Args:
        confusion_matrix_normalized: 正規化された混同行列（パーセンテージ）
        confusion_matrix_counts: 生の数値の混同行列（カウント）
        results_dir: 結果保存ディレクトリ
        timestamp: タイムスタンプ（Noneの場合は自動生成）
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs(results_dir, exist_ok=True)
    
    # 正規化された混同行列のプロット（パーセンテージ）
    plt.figure(figsize=(14, 7))
    sns.set(font_scale=1.4)
    # パーセンテージを0-1の範囲に変換（seaborn用）
    df_cm_normalized = confusion_matrix_normalized / 100.0
    sns.heatmap(df_cm_normalized, cmap="Blues", annot=True, 
                annot_kws={"size": 16}, fmt='.2%', cbar_kws={'label': 'Percentage'})
    plt.title('Confusion Matrix (Normalized %)', fontsize=16, pad=20)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    
    cm_plot_normalized_path = os.path.join(results_dir, f'confusion_matrix_normalized_plot_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(cm_plot_normalized_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Normalized confusion matrix plot saved: {cm_plot_normalized_path}")
    
    # 生の数値の混同行列のプロット（カウント、marginsを除外）
    # margins行と列を除外
    df_cm_counts = confusion_matrix_counts.iloc[:-1, :-1] if 'All' in confusion_matrix_counts.index else confusion_matrix_counts
    
    plt.figure(figsize=(14, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm_counts, cmap="Blues", annot=True, 
                annot_kws={"size": 16}, fmt='d', cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix (Counts)', fontsize=16, pad=20)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    
    cm_plot_counts_path = os.path.join(results_dir, f'confusion_matrix_counts_plot_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(cm_plot_counts_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Counts confusion matrix plot saved: {cm_plot_counts_path}")


def save_confusion_matrix_with_labels(confusion_matrix_normalized: pd.DataFrame, 
                                     confusion_matrix_counts: pd.DataFrame,
                                     results_dir: str = 'results', 
                                     timestamp: Optional[str] = None) -> None:
    """
    ラベル名付き混同行列を保存
    
    Args:
        confusion_matrix_normalized: 正規化された混同行列（パーセンテージ）
        confusion_matrix_counts: 生の数値の混同行列（カウント）
        results_dir: 結果保存ディレクトリ
        timestamp: タイムスタンプ（Noneの場合は自動生成）
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs(results_dir, exist_ok=True)
    
    # 正規化された混同行列を保存（パーセンテージ、ラベル名付き）
    cm_normalized_path = os.path.join(results_dir, f'confusion_matrix_normalized_labels_{timestamp}.csv')
    confusion_matrix_normalized.to_csv(cm_normalized_path, index=True)
    logging.info(f"Normalized confusion matrix (with labels) saved: {cm_normalized_path}")
    
    # 生の数値の混同行列を保存（カウント、ラベル名付き）
    cm_counts_path = os.path.join(results_dir, f'confusion_matrix_counts_labels_{timestamp}.csv')
    confusion_matrix_counts.to_csv(cm_counts_path, index=True)
    logging.info(f"Confusion matrix counts (with labels) saved: {cm_counts_path}")
    
    # 混同行列を表示
    logging.info("=" * 50)
    logging.info("Confusion Matrix (Counts) with Labels:")
    logging.info("\n" + str(confusion_matrix_counts))
    logging.info("=" * 50)
    logging.info("Confusion Matrix (Normalized %) with Labels:")
    logging.info("\n" + str(confusion_matrix_normalized))
    logging.info("=" * 50)
    
    # 混同行列をプロット
    logging.info("Plotting confusion matrices...")
    plot_confusion_matrix(confusion_matrix_normalized, confusion_matrix_counts, results_dir, timestamp)
    logging.info("Confusion matrices plotted")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='Evaluate trained IDS model on CICIDS2017_improved')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model (.pkl file)')
    parser.add_argument('--categories_path', type=str, required=True,
                        help='Path to categories file (.categories file)')
    parser.add_argument('--rf_model_path', type=str, default=None,
                        help='Path to Random Forest model (.pkl file, optional)')
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
    
    args = parser.parse_args()
    
    # ログ設定
    timestamp = setup_logging(args.logs_dir)
    logging.info("=" * 50)
    logging.info("Starting IDS model evaluation")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Categories path: {args.categories_path}")
    logging.info(f"RF model path: {args.rf_model_path}")
    logging.info(f"Dataset path: {args.dataset_path}")
    logging.info(f"Results directory: {args.results_dir}")
    logging.info("=" * 50)
    
    try:
        # 1. 訓練済みモデルの読み込み
        logging.info("Loading trained model...")
        model, categories, rf_model = load_trained_model(
            args.model_path, 
            args.categories_path, 
            args.rf_model_path
        )
        logging.info("Model loaded successfully")
        
        # 2. データ読み込み
        logging.info("Loading dataset...")
        df = load_cicids2017_dataset(args.dataset_path)
        logging.info(f"Dataset loaded: {len(df)} samples")
        
        # 3. 前処理
        logging.info("Preprocessing dataset...")
        df = preprocess_cicids2017(df)
        logging.info("Preprocessing completed")
        
        # 4. データ分割（訓練時と同じ分割を使用）
        logging.info("Splitting dataset...")
        train_df, test_df = split_dataset(df, train_test_split=args.train_test_split, random_seed=args.random_seed)
        logging.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
        
        # 5. ラベルのインデックス変換マッピングを作成
        label_to_idx = {label: idx for idx, label in enumerate(categories)}
        logging.info(f"Label to index mapping: {label_to_idx}")
        
        # 6. テストデータのみを拡張（評価用）
        logging.info("Creating extended dataset for test data only...")
        test_cats = test_df['Label'].copy().reset_index(drop=True)
        test_df_ = test_df.drop(['Label'], axis=1).reset_index(drop=True)
        rf_features, cols = extend_dataset(model, test_df_, test_cats, label_tag='Label')
        logging.info("Extended test dataset created")
        
        # 7. 拡張テストデータセットの作成
        test_df_extended = pd.DataFrame(data=rf_features, columns=cols)
        
        # 無限大やNaNの値をチェック・処理
        numeric_cols = test_df_extended.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = test_df_extended[col].values
            if np.isinf(col_data).any() or np.isnan(col_data).any():
                logging.warning(f"Column {col} contains inf or NaN values. Replacing with 0.")
                test_df_extended[col] = test_df_extended[col].replace([np.inf, -np.inf], 0)
                test_df_extended[col] = test_df_extended[col].fillna(0)
        
        # 再度チェック
        numeric_data = test_df_extended[numeric_cols].values
        if np.isinf(numeric_data).any() or np.isnan(numeric_data).any():
            logging.error("Data still contains inf or NaN values after cleaning")
            raise ValueError("Data contains invalid values (inf or NaN)")
        
        logging.info(f"Extended test samples: {len(test_df_extended)}")
        
        # 8. 評価の実行
        logging.info("Evaluating model...")
        test_df_extended_ = test_df_extended.drop(['Label'], axis=1)
        x_test = test_df_extended_.values
        
        if rf_model is not None:
            y_hat = rf_model.predict(x_test)
        else:
            # RFモデルがない場合は、Adaptive Clusteringモデルから予測を取得
            # これはモデルの実装に依存します
            raise ValueError("Random Forest model is required for evaluation")
        
        y_actual_ = test_df_extended['Label'].map(label_to_idx).values
        
        # 評価メトリクスの計算（ラベル名付き混同行列を含む）
        confusion_matrix_normalized, confusion_matrix_counts, metrics_data, feature_importance = evaluate_model(
            model, test_df_extended_, categories, y_actual_, y_hat
        )
        
        # 9. 結果の表示
        logging.info("=" * 50)
        logging.info("Evaluation Results:")
        logging.info(f"Accuracy: {metrics_data.get('accuracy', 0):.4f}")
        logging.info(f"Precision: {metrics_data.get('precision', 0):.4f}")
        logging.info(f"Recall: {metrics_data.get('DR/Recall', 0):.4f}")
        logging.info(f"F1 Score: {metrics_data.get('F1 SCORE', 0):.4f}")
        logging.info(f"False Alarm Rate: {metrics_data.get('FAR', 0):.4f}")
        logging.info("=" * 50)
        
        # 10. ラベル名付き混同行列の保存
        logging.info("Saving confusion matrices with labels...")
        save_confusion_matrix_with_labels(
            confusion_matrix_normalized, 
            confusion_matrix_counts,
            args.results_dir, 
            timestamp
        )
        logging.info("Confusion matrices saved")
        
        logging.info("=" * 50)
        logging.info("Evaluation completed successfully")
        logging.info("=" * 50)
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

