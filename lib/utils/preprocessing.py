"""
CICIDS2017データセットの前処理関数
"""
import pandas as pd
import numpy as np
from typing import Tuple
from glob import glob
import os
from sklearn.model_selection import train_test_split as sk_train_test_split


def preprocess_cicids2017(df: pd.DataFrame) -> pd.DataFrame:
    """
    CICIDS2017_improvedデータセットの前処理
    
    - Protocol列のワンホットエンコーディング
    - 不要な列の削除
    - ラベルの正規化
    
    Args:
        df: 入力データフレーム
        
    Returns:
        前処理済みデータフレーム
    """
    # Protocol列のワンホットエンコーディング
    df["Protocol_0"] = (df["Protocol"] == 0).astype(int)
    df["Protocol_6"] = (df["Protocol"] == 6).astype(int)
    df["Protocol_17"] = (df["Protocol"] == 17).astype(int)
    
    # 不要な列の削除
    columns_to_drop = [
        "Protocol", "Attempted Category", "ICMP Type", "ICMP Code",
        "Flow ID", "Timestamp", "Src IP", "Src Port", "Dst IP", "Dst Port", "id"
    ]
    # 存在する列のみ削除
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_columns_to_drop)
    
    # ラベルの正規化
    labels = df["Label"].unique()
    for label in labels:
        if "Attempted" in str(label):
            df.loc[df["Label"] == label, "Label"] = "BENIGN"
        if "Web Attack" in str(label):
            df.loc[df["Label"] == label, "Label"] = "Web Attack"
        if "Portscan" in str(label):
            df.loc[df["Label"] == label, "Label"] = "Portscan"
        if "Infiltration" in str(label):
            df.loc[df["Label"] == label, "Label"] = "Infiltration"
        if "DoS" in str(label) and label != "DDoS":
            df.loc[df["Label"] == label, "Label"] = "DoS"
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        df = df.reset_index(drop=True)
    
    return df


def load_cicids2017_dataset(dataset_path: str, chunk_size: int = 50000) -> pd.DataFrame:
    """
    CICIDS2017_improvedデータセットを読み込む
    
    Args:
        dataset_path: データセットパス
        chunk_size: チャンクサイズ
        
    Returns:
        読み込んだデータフレーム
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    if os.path.isdir(dataset_path):
        dataset_path_pattern = os.path.join(dataset_path, '*.csv')
        files = sorted(glob(dataset_path_pattern))
        
        dfs = []
        for file in files:
            # チャンク読み込みでメモリ効率を改善
            df_chunks = pd.read_csv(file, chunksize=chunk_size, low_memory=False)
            for chunk in df_chunks:
                dfs.append(chunk)
        
        # 一度に結合
        df = pd.concat(dfs, ignore_index=True)
        del dfs
    else:
        # 単一ファイルの場合もチャンク読み込みを使用
        dfs = []
        for chunk in pd.read_csv(dataset_path, chunksize=chunk_size, index_col=False, low_memory=False):
            dfs.append(chunk)
        df = pd.concat(dfs, ignore_index=True)
        del dfs
    
    # データ型を最適化（メモリ使用量削減）
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df


def split_dataset(df: pd.DataFrame, train_test_split: float = 0.7, 
                  random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    データセットを訓練用とテスト用に分割
    
    Args:
        df: データフレーム
        train_test_split: 訓練データの割合
        random_seed: ランダムシード
        
    Returns:
        (訓練データフレーム, テストデータフレーム)のタプル
    """
    train_df, test_df = sk_train_test_split(
        df, 
        test_size=1-train_test_split, 
        random_state=random_seed, 
        stratify=df['Label']
    )
    
    return train_df, test_df
