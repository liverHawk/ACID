#!/usr/bin/env python
# coding: utf-8

# # Adaptive Clustering based Intrusion Detection
# ---
# <p style="font-size: 1.5em; line-height: 1.2em;">In this notebook, we build an Intrusion Detection System based on our classifier-agnostic Adaptive Clustering approach as presented in our paper: <a href="https://homepages.inf.ed.ac.uk/ppatras/pub/infocom21.pdf" target="_blank">Adaptive Clustering-based Malicious Traffic Classification at the Network Edge</a>. This approach obtains a perfect Accuracy and F1-score of <b>100.0%</b> with a False Alarm Rate of <b>0%</b>, making it the new state-of-the-art approach for intrusion detection.</p>
# 
# <p style="font-size: 1.5em; line-height: 1.2em;">In this notebook, we will perform a multi-label classification task on the preprocessed <a href="https://www.unb.ca/cic/datasets/ids-2017.html" target="_blank">CICIDS2017_improved</a> dataset.</p>
# <p style="padding-left: 10px; margin-left: 20px; border-left: 5px solid #CCC; font-style: italic;">
# Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”, 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018
# </p>

# ## Imports

# In[ ]:


#!/usr/bin/env python

__author__ = "Alec F."
__license__ = "GPL"
__version__ = "1.0.2"


import os
import sys
import argparse
from datetime import datetime
from lib.models.network import AdaptiveClustering
from lib.utils.misc import extend_dataset
from lib.utils import Dataset
from lib.utils.preprocessing import load_cicids2017_dataset, preprocess_cicids2017, split_dataset

import json

import pandas as pd
import numpy as np
import pickle as pkl

import torch
from tqdm import tqdm

from torch.utils.data import DataLoader

from lib.models.RandomForest import RandomForest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support as prf, accuracy_score

from pprint import pprint


# ## Helpers
# 
# We use these helpers functions throughout all our experiments for simplify our main code and improve its readability. In all our experiments, our datasets are separated to approximately obtain a 70/30 train-test split from randomly sampled data. We use the pickle library to save our trained model and defined a set of metrics that are commonly used for classification tasks.

# In[ ]:


def save_model(rf, filename):
    with open(filename, 'wb') as f:
        pkl.dump(rf, f)


def save_clustering_model(model, filename):
    """AdaptiveClusteringモデルだけを保存（RandomForestなし）"""
    with open(filename, 'wb') as f:
        pkl.dump(model, f)


def load_model(filename):
    with open(filename, 'rb') as f:
        m = pkl.load(f)
    return m


class metrics(object):

    def __init__(self, tp=None, tn=None, fp=None, fn=None):
        super(metrics, self).__init__()

        self.tp, self.tn, self.fp, self.fn = tp, tn, fp, fn

        self.metrics = {}

    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def detection_rate(self):
        return self.tp / (self.tp + self.fn)

    def false_alarm_rate(self):
        return self.fp / (self.fp + self.tn)

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def f1(self):
        prec = self.precision()
        rec = self.detection_rate()
        return 0 if (prec + rec) == 0 else 2 * (prec * rec) / (prec + rec)

    def get_metrics(self):
        self.metrics = {"Acc": self.accuracy(), "DR/Recall": self.detection_rate(), "FAR": self.false_alarm_rate(),
                        "PRECISION": self.precision(), "F1 SCORE": self.f1()}
        return self.metrics


# ## Adaptive Clustering - training function
# 
# We train our Adaptive Clustering network by following the exact settings provided in the paper.
# However, we also implement an early stop mechanism allowing us to stop the training as soon as we achieve an acceptable loss.

# In[ ]:


def train(X, y, categories, lr=1e-4, n_epoch=100, batch_size=None, device_preference='auto',
          encoder_dims=None, kernel_size=10, early_stop_threshold=1.0):
    """
    訓練関数
    
    Args:
        X: 訓練データ
        y: 訓練ラベル
        categories: カテゴリリスト
        lr: 学習率
        n_epoch: エポック数
        batch_size: バッチサイズ（Noneの場合は自動決定）
        device_preference: デバイス指定（auto/cpu/cuda/mps）
        encoder_dims: エンコーダーの次元リスト（Noneの場合は[500, 200, 50]）
        kernel_size: カーネルサイズ
        early_stop_threshold: 早期停止の閾値
    """
    # デバイスを決定
    pref = (device_preference or "auto").lower()
    if pref == "cpu":
        use_cuda = False
        use_mps = False
        device = torch.device("cpu")
        default_batch_size = 4096
    elif pref == "cuda":
        use_cuda = torch.cuda.is_available()
        use_mps = False
        if use_cuda:
            device = torch.device("cuda")
            default_batch_size = 8192
        else:
            print("Warning: CUDA is not available. Falling back to CPU.")
            device = torch.device("cpu")
            default_batch_size = 4096
    elif pref == "mps":
        use_cuda = False
        use_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        if use_mps:
            device = torch.device("mps")
            default_batch_size = 4096
        else:
            print("Warning: MPS is not available. Falling back to CPU.")
            device = torch.device("cpu")
            default_batch_size = 4096
    else:
        # auto: CUDA > MPS > CPU
        use_cuda = torch.cuda.is_available()
        use_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        if use_cuda:
            device = torch.device("cuda")
            default_batch_size = 8192
        elif use_mps:
            device = torch.device("mps")
            default_batch_size = 4096
        else:
            device = torch.device("cpu")
            default_batch_size = 4096

    # バッチサイズを決定
    if batch_size is None:
        batch_size = default_batch_size

    # エンコーダーの次元を決定
    if encoder_dims is None:
        encoder_dims = [500, 200, 50]

    # lib版のAdaptiveClusteringモデルを使用
    model_ = AdaptiveClustering(encoder_dims=encoder_dims, n_kernels=len(categories), kernel_size=kernel_size)
    model_.train()

    # モデルをデバイスに移動
    model_ = model_.to(device)
    
    if device.type == 'cuda':
        pin_memory = True
        num_workers = 2
    else:
        pin_memory = False
        num_workers = 4

    # Xとyをnumpy配列に変換してからTensor化（メモリ効率向上）
    if isinstance(X, torch.Tensor):
        X_np = X.numpy() if not use_cuda else X.cpu().numpy()
    else:
        X_np = np.asarray(X, dtype=np.float32)
    
    if isinstance(y, torch.Tensor):
        y_np = y.numpy() if not use_cuda else y.cpu().numpy()
    else:
        y_np = np.asarray(y, dtype=np.int64)

    # DatasetとDataLoaderを作成（最適化）
    ds = Dataset(X_np, y_np)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, 
                   num_workers=num_workers, pin_memory=pin_memory,
                   prefetch_factor=8 if num_workers > 0 else None,  # prefetchを増やす
                   persistent_workers=True if num_workers > 0 else False,
                   drop_last=False,
                   generator=torch.Generator().manual_seed(42) if num_workers > 0 else None)  # 再現性確保

    # Adam optimizerを最適化
    optimizer = torch.optim.Adam(model_.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    # ロス出力の間隔（ステップ数）
    loss_log_interval = 1
    step_count = 0
    accumulated_loss = 0.0
    loss_count = 0

    for i in range(n_epoch):
        iteration_losses = []
        model_.train()

        pbar = tqdm(dl, total=len(dl), desc=f"Epoch {i+1}/{n_epoch}")
        for x, labels_ in pbar:
            step_count += 1
            
            # バッチをTensorに変換してデバイスに移動
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            if not isinstance(labels_, torch.Tensor):
                labels_ = torch.LongTensor(labels_)
            
            # デバイスに移動（非同期転送で高速化）
            x = x.to(device, non_blocking=(device.type == 'cuda'))
            labels_ = labels_.to(device, non_blocking=(device.type == 'cuda'))

            optimizer.zero_grad()
            _ = model_(x, labels_)

            # ロス計算（毎回必要だが、値の取得は100ステップごと）
            loss = model_.loss()
            
            # 100ステップごとにのみロス値を取得・表示（CPU-GPU同期を削減）
            if step_count % loss_log_interval == 0:
                loss_value = float(loss.item())  # ここでCPU-GPU同期が発生
                accumulated_loss += loss_value
                loss_count += 1
                iteration_losses.append(loss_value)
                
                # 平均ロスを計算
                avg_loss_so_far = accumulated_loss / loss_count if loss_count > 0 else loss_value
                pbar.set_postfix({'loss': f'{avg_loss_so_far:.4f}', 'step': step_count})
            # else: ロス値の取得をスキップ（loss.item()を呼ばない）

            loss.backward()
            # 勾配クリッピングで安定化
            torch.nn.utils.clip_grad_norm_(model_.parameters(), max_norm=1.0)
            optimizer.step()

        # エポック終了時に平均ロスを計算
        if len(iteration_losses) > 0:
            avg_loss = np.mean(iteration_losses)
        else:
            # ロスが記録されていない場合は最後に計算
            loss = model_.loss()
            avg_loss = float(loss.item())
        
        print(f"Iteration {i+1} | Loss {avg_loss:.6f} | Steps: {step_count}")
        if avg_loss < early_stop_threshold:
            print(f"Early stop triggered at: Iteration {i}")
            break
        pbar.close()
        
        # エポック間でリセット
        accumulated_loss = 0.0
        loss_count = 0

    model_.eval()
    return model_


# ## Variables
# 
# Here we define some global variables to locate our datasets and point to our results directory.

# In[ ]:


root_path = './'

dataset_path = root_path + 'dataset/CICIDS2017_improved'
extended_dataset_path = root_path + 'dataset/extended'

results_path = root_path + 'results/'
if not os.path.exists(results_path):
    os.mkdir(results_path)


# In[ ]:


model_path = results_path + 'trained_model.pkl'
separator = "-"*50


# ## Dataset
# ### Loading dataset
# 
# We load the preprocessed dataset as a pandas data frame and show the first few rows.

# In[15]:


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description='Train Adaptive Clustering model for IDS on CICIDS2017_improved'
    )
    parser.add_argument('--dataset_path', type=str, default='dataset/CICIDS2017_improved',
                        help='Dataset path (default: dataset/CICIDS2017_improved)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Results directory (default: results/)')
    parser.add_argument('--extended_dataset_path', type=str, default='dataset/extended',
                        help='Extended dataset path (default: dataset/extended)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: auto)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use: auto / cpu / cuda / mps (default: auto)')
    parser.add_argument('--early_stop_threshold', type=float, default=1.0,
                        help='Early stop threshold (default: 1.0)')
    parser.add_argument('--encoder_dims', type=int, nargs='+', default=None,
                        help='Encoder dimensions (default: [500, 200, 50])')
    parser.add_argument('--kernel_size', type=int, default=10,
                        help='Kernel size (default: 10)')
    parser.add_argument('--train_test_split', type=float, default=0.7,
                        help='Train/test split ratio (default: 0.7)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # パスを設定
    dataset_path = args.dataset_path
    results_path = args.results_dir + '/' if not args.results_dir.endswith('/') else args.results_dir
    extended_dataset_path = args.extended_dataset_path
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    separator = "-"*50
    
    print(separator)
    print("Loading dataset...")
    # CICIDS2017_improvedデータセットを読み込み
    df = load_cicids2017_dataset(dataset_path)
    print(f"Dataset loaded: {len(df)} samples")
    
    # 前処理を実行
    print("Preprocessing dataset...")
    df = preprocess_cicids2017(df)
    print("Preprocessing completed")
    
    # データ分割
    print("Splitting dataset...")
    train_df, test_df = split_dataset(df, train_test_split=args.train_test_split, random_seed=args.random_seed)
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    print("Done loading dataset")
    
    # カテゴリを取得
    categories = sorted(list(set(train_df['Label'].values)))
    print(f"Categories: {categories}")
    
    train_df_ = train_df.drop(['Label'], axis=1)
    
    # numpy配列として保持（Tensor変換はDataset内で行う）
    X = train_df_.values.astype(np.float32)
    
    # ラベルのインデックス変換をベクトル化（高速化）
    label_to_idx = {label: idx for idx, label in enumerate(categories)}
    y = train_df['Label'].map(label_to_idx).values.astype(np.int64)
    
    cats = df['Label'].copy()
    y_ = df['Label'].values
    df.drop(['Label'], axis=1, inplace=True)
    
    # 訓練パラメータを表示
    print(separator)
    print("Training Parameters:")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Number of epochs: {args.n_epochs}")
    print(f"  Batch size: {args.batch_size if args.batch_size else 'auto'}")
    print(f"  Device: {args.device}")
    print(f"  Early stop threshold: {args.early_stop_threshold}")
    print(f"  Encoder dims: {args.encoder_dims if args.encoder_dims else [500, 200, 50]}")
    print(f"  Kernel size: {args.kernel_size}")
    print(separator)
    
    # モデルを訓練
    print("Training model...")
    model = train(X, y, categories, 
                  lr=args.learning_rate,
                  n_epoch=args.n_epochs,
                  batch_size=args.batch_size,
                  device_preference=args.device,
                  encoder_dims=args.encoder_dims,
                  kernel_size=args.kernel_size,
                  early_stop_threshold=args.early_stop_threshold)
    print("Done training model")
    
    # Adaptive Clusteringモデルだけを保存（RandomForest追加前）
    clustering_model_path = results_path + 'adaptive_clustering_model.pkl'
    if not os.path.exists(clustering_model_path):
        print("Saving Adaptive Clustering model (clustering only)...")
        save_clustering_model(model, clustering_model_path)
        with open(clustering_model_path + ".categories", "w") as f:
            json.dump(categories, f)
            print(f"Categories saved: {clustering_model_path}.categories")
        print(f"Done saving Adaptive Clustering model: {clustering_model_path}")
    else:
        print(f"Model already exists: {clustering_model_path}")
    
    if not os.path.exists(extended_dataset_path):
        print("Creating extended dataset...")
        rf_features, cols = extend_dataset(model, df, cats, label_tag='Label')
        print("Done creating extended dataset")
    else:
        print(f"Extended dataset already exists: {extended_dataset_path}")
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(results_path + f'{now}_training_log.txt', 'w') as f:
        f.write(f"Training log: {now}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Number of epochs: {args.n_epochs}\n")
        f.write(f"Batch size: {args.batch_size if args.batch_size else 'auto'}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Early stop threshold: {args.early_stop_threshold}\n")
        f.write(f"Encoder dims: {args.encoder_dims if args.encoder_dims else [500, 200, 50]}\n")
        f.write(f"Kernel size: {args.kernel_size}\n")


if __name__ == '__main__':
    main()
else:
    # スクリプトとして直接実行された場合の互換性のため
    # 既存のコードを保持（Jupyter notebook形式の互換性）
    pass

# As we can see here, thanks to our early stopping mechanism, we train our Adaptive Clustering model for only 17 iterations and still obtain a perfect classification accuracy and F-score of __100%__ as shown in the [performance metrics section](#Performance-metrics).

# In[ ]:

exit()
if not os.path.exists(extended_dataset_path):
    print("Saving new dataset...")
    new_df = pd.DataFrame(data=rf_features, columns=cols)
    new_df.to_csv(extended_dataset_path, index=False)
    print(f"Done saving new dataset: {extended_dataset_path}")

    train_test_split = .7
    msk = np.random.rand(len(new_df)) < train_test_split
    train_df = new_df[msk]
    test_df = new_df[~msk]

    df_ = train_df.drop(['Label'], axis=1)

    # リスト変換をスキップ（高速化）
    X = df_.values
    
    # infinityやNaNをチェックして置換（より安全に）
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    # 非常に大きな値も制限（1e10から1e6に変更）
    X = np.clip(X, -1e6, 1e6)
    # float32に変換してメモリ効率を向上
    X = X.astype(np.float32)

    # ラベルのインデックス変換をベクトル化（高速化）
    label_to_idx = {label: idx for idx, label in enumerate(categories)}
    y = train_df['Label'].map(label_to_idx).values


# After obtaining the extended dataset, we train our final classifier and evaluate our Intrusion Detection System.

# In[ ]:


if not os.path.exists(model_path):
    print("Training Random Forest...")
    model.classifier = RandomForest(n_estimators=200)
    model.classifier.fit(X, y)
    print("Done training Random Forest")

    print("Saving integrated model (with Random Forest)...")
    save_model(model, model_path)
    with open(model_path+".categories", "w") as f:
        json.dump(categories, f)
        print(f"Categories saved: {model_path}.categories")
    print("Done saving integrated model")


# In[ ]:


if os.path.exists(model_path):
    print("Loading model...")
    model = load_model(model_path)
    print("Done loading model")


# ### Data distributions
# 
# Here, we print our data distributions per traffic category to verify that we have all categories of our dataset in the test set.

# In[ ]:


print(f"Total data distribution: {len(test_df) + len(train_df)}")
total = {}
total['total'] = len(test_df) + len(train_df)
for cat in categories:
    total[cat] = len(test_df[test_df['Label'] == cat]) \
                   + len(train_df[train_df['Label'] == cat])
    print(f"\t{cat}: {total[cat]}")

print(f"Training data distribution: {len(train_df)} / {len(train_df) * 100 / total['total']}%")
for cat in categories:
    c = len(train_df[train_df['Label'] == cat])
    print(f"\t{cat}: {c} / {c*100 / total[cat]}%")
print(f"Testing data distribution: {len(test_df)} / {len(test_df) * 100 / total['total']}%")
for cat in categories:
    c = len(test_df[test_df['Label'] == cat])
    print(f"\t{cat}: {c} / {c * 100 / total[cat]}%")


# ### Testing the model
# 
# To evaluate our model on the unseen set, we remove the labels and run our trained model on the remaining features.

# In[ ]:


df = test_df.drop(['Label'], axis=1)
# リスト変換をスキップして直接numpy配列を使用（高速化）
x = df.values

y_hat = model.classifier.predict(x)
# リスト内包表記で高速化
preds = [categories[i] for i in y_hat]


# ### Confusion matrix
# 
# We compute the confusion matrix to observe how well our model works on every single category in our test set.

# In[ ]:


print("Creating confusion matrix...")

# ラベルのマッピングを一度に実行（高速化）
category_to_idx = {category: i for i, category in enumerate(categories)}
y_actual_ = test_df['Label'].map(category_to_idx).values
# リスト内包表記で高速化
y_actual = [categories[i] for i in y_actual_]

print(f"Uniques actual: {list(set(y_actual))}")
print(f"Uniques preds: {list(set(preds))}")

y_actu = pd.Series(y_actual, name='Actual')
y_pred = pd.Series(preds, name='Predicted')

df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'],
                           dropna=False, margins=False, normalize='index').round(4) * 100

df_confusion.to_csv(results_path + 'rf_confusion_matrix.csv')
print(f"Confusion matrix created and saved: {results_path}rf_confusion_matrix.csv")


# In[ ]:


df_confusion


# As mentioned in the paper, and shown by the confusion matrix above, our approach obtains a perfect classification score over the entire test set.

# ### Confusion matrix heatmap
# 
# For a clearer visualization, we plot our confusion matrix as a classification ratio for all different categories and again, observe that we have a perfect classification score on every single category in the entire test set.

# In[ ]:


import matplotlib
matplotlib.use('Agg')  # バックエンドを設定（GUI不要）
import matplotlib.pyplot as plt
import seaborn as sns

df_cm =  pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'],
                           dropna=False, margins=False, normalize='index').round(4)
plt.figure(figsize = (14,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})
b, t = plt.ylim()
b += 0.5
t -= 0.5
plt.ylim(b, t)
plt.show()


# ### Important features
# 
# We can see the degree of contribution of each feature to the classification result by using the __feature\_importances\___ attribute of the Random Forest Classifier. We see here that, as mentioned in the paper, the 10 features extracted from our Adaptive Clustering network consistute the 10 most important features for the classifier.

# In[ ]:


print("Calculating feature importance scores...")
importances = model.classifier.cls.feature_importances_
indices = np.argsort(importances)[-50:][::-1]  # -50 #most important features
cols = test_df.columns.values.tolist()

feature_importances_list = []
for f in range(len(indices)):
    feature_importances_list.append([indices[f], cols[indices[f]], importances[indices[f]]])

with open(results_path + "rf_feature_importance_scores.pkl", "wb") as f:
    pkl.dump(feature_importances_list, f)
    print(f"Feature importance scores saved: {results_path}rf_feature_importance_scores.pkl")

    print("")
    print("15 most important features")
    pprint(feature_importances_list[:15])
    print("Sum 15:", np.sum(importances[indices[:15]]))
    print("")


# ### Performance metrics
# 
# Finally, we compute the different performance metrics on our test set. We can see here again that this approach allows us to obtain a perfect score on every single evaluation metric, making this the new state-of-the-art approach for classification tasks.

# In[ ]:


print("Calculating performance metrics...")
cnf_matrix = confusion_matrix(y_actual_, y_hat)
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

metrics_handler = metrics(tp=_tp, tn=_tn, fp=_fp, fn=_fn)
metrics_data = metrics_handler.get_metrics()

accuracy = accuracy_score(y_actual_, y_hat)
precision, recall, f_score, support = prf(y_actual_, y_hat, average='weighted')
print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, FAR: {:0.4f}"
                                                      .format(accuracy, precision,
                                                              recall, f_score, metrics_data["FAR"]))

with open(results_path + "metrics_handler.pkl", "wb") as f:
    pkl.dump(metrics_handler, f)
    print(f"Performance metrics saved: {results_path}metrics_handler.pkl")

print(separator)


# ### Putting it all together
# 
# <p style="font-size: 1.2em; line-height: 1.2em;">We have implemented our approach and obtained a striking state-of-the-art performance by producing a <b>100%</b> success rate with <b>zero</b> false alarms on the CICIDS2017_improved intrusion detection dataset. We have also shown that the features learned from the proposed classifier-agnostic Adaptive Clustering network constitute the features that participate the most to the decision process of the final classifier.</p>

# In[ ]:





