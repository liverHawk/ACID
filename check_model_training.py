#!/usr/bin/env python
"""
訓練済みモデルの状態を確認するスクリプト

モデルが適切に訓練されているかを確認します。
"""
import pickle
import json
import torch
import numpy as np
import sys

def check_model(model_path: str, categories_path: str):
    """
    モデルの状態を確認
    
    Args:
        model_path: モデルファイルのパス
        categories_path: カテゴリファイルのパス
    """
    # モデルを読み込み
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # カテゴリを読み込み
    with open(categories_path, 'r') as f:
        categories = json.load(f)
    
    print("=" * 50)
    print("Model Information")
    print("=" * 50)
    print(f"Number of categories: {len(categories)}")
    print(f"Categories: {categories}")
    print(f"Number of kernels: {model.n_kernels_}")
    print(f"Kernel size: {model.kernel_size}")
    print(f"Model training mode: {model.training}")
    print()
    
    # カーネル重みの統計を確認
    print("=" * 50)
    print("Kernel Weights Statistics")
    print("=" * 50)
    for i in range(model.n_kernels_):
        kernel_weights = model.sub_nets[i].kernel_weights.detach().cpu().numpy()
        print(f"Kernel {i}:")
        print(f"  Shape: {kernel_weights.shape}")
        print(f"  Min: {kernel_weights.min():.6f}")
        print(f"  Max: {kernel_weights.max():.6f}")
        print(f"  Mean: {kernel_weights.mean():.6f}")
        print(f"  Std: {kernel_weights.std():.6f}")
        print(f"  Norm: {np.linalg.norm(kernel_weights):.6f}")
        print()
    
    # カーネル間の距離を確認
    print("=" * 50)
    print("Inter-Kernel Distances")
    print("=" * 50)
    kernel_weights_list = [
        model.sub_nets[i].kernel_weights.detach().cpu().numpy().flatten()
        for i in range(model.n_kernels_)
    ]
    
    for i in range(model.n_kernels_):
        for j in range(i + 1, model.n_kernels_):
            dist = np.linalg.norm(kernel_weights_list[i] - kernel_weights_list[j])
            print(f"Distance between kernel {i} and {j}: {dist:.6f}")
    
    print()
    print("=" * 50)
    print("Analysis")
    print("=" * 50)
    
    # カーネル重みが初期化のままかどうかを確認
    # 通常、訓練後はカーネル間の距離が大きくなる
    min_dist = float('inf')
    for i in range(model.n_kernels_):
        for j in range(i + 1, model.n_kernels_):
            dist = np.linalg.norm(kernel_weights_list[i] - kernel_weights_list[j])
            min_dist = min(min_dist, dist)
    
    if min_dist < 0.01:
        print("WARNING: Kernel weights are very close to each other.")
        print("This suggests the model may not have been trained properly.")
        print("Possible causes:")
        print("  1. Training was not completed")
        print("  2. Learning rate was too small")
        print("  3. Training data was insufficient")
        print("  4. Early stopping triggered too early")
    else:
        print(f"Kernel weights are well separated (min distance: {min_dist:.6f})")
        print("Model appears to be trained.")
    
    # カーネル重みのノルムを確認
    norms = [np.linalg.norm(kw) for kw in kernel_weights_list]
    print(f"\nKernel weight norms: {[f'{n:.6f}' for n in norms]}")
    
    if all(n < 0.1 for n in norms):
        print("WARNING: Kernel weights have very small norms.")
        print("This suggests the model may not have been trained properly.")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python check_model_training.py <model_path> <categories_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    categories_path = sys.argv[2]
    
    check_model(model_path, categories_path)
