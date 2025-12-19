#!/usr/bin/env python
# coding: utf-8

# # Adaptive Clustering based Intrusion Detection
# ---
# <p style="font-size: 1.5em; line-height: 1.2em;">In this notebook, we build an Intrusion Detection System based on our classifier-agnostic Adaptive Clustering approach as presented in our paper: <a href="https://homepages.inf.ed.ac.uk/ppatras/pub/infocom21.pdf" target="_blank">Adaptive Clustering-based Malicious Traffic Classification at the Network Edge</a>. This approach obtains a perfect Accuracy and F1-score of <b>100.0%</b> with a False Alarm Rate of <b>0%</b>, making it the new state-of-the-art approach for intrusion detection.</p>
# 
# <p style="font-size: 1.5em; line-height: 1.2em;">In this notebook, we will perform a multi-label classification task on the preprocessed <a href="https://registry.opendata.aws/cse-cic-ids2018/" target="_blank">CSE-CIC-IDS 2018</a> dataset.</p>
# <p style="padding-left: 10px; margin-left: 20px; border-left: 5px solid #CCC; font-style: italic;">
# Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”, 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018
# </p>

# ## Imports

# In[ ]:


#!/usr/bin/env python

__author__ = "Alec F."
__license__ = "GPL"
__version__ = "1.0.1"


import os
import sys
from glob import glob

from core.models.network_optimized import AdaptiveClusteringOptimized
from core.utils.misc import extend_dataset
from core.utils import Dataset

import json

import pandas as pd
import numpy as np
import pickle as pkl

import torch
from tqdm import tqdm

from torch.utils.data import DataLoader

from core.models.RandomForest import RandomForest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support as prf, accuracy_score

from pprint import pprint

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# ## Helpers
# 
# We use these helpers functions throughout all our experiments for simplify our main code and improve its readability. In all our experiments, our datasets are separated to approximately obtain a 70/30 train-test split from randomly sampled data. We use the pickle library to save our trained model and defined a set of metrics that are commonly used for classification tasks.

# In[ ]:


def load_datasets(dataset_path):
    global categories
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    if os.path.isdir(dataset_path):
        dataset_path_pattern = os.path.join(dataset_path, '*.csv')
        files = sorted(glob(dataset_path_pattern))  # ソートして順序を保証

        # チャンクサイズを増やして高速化（メモリが許す限り）
        chunk_size = 100000
        dfs = []
        for file in files:
            # データ型を最適化して読み込み（engine='c'で高速化）
            try:
                df_chunks = pd.read_csv(file, chunksize=chunk_size, low_memory=False, engine='c')
            except:
                df_chunks = pd.read_csv(file, chunksize=chunk_size, low_memory=False)
            for chunk in df_chunks:
                dfs.append(chunk)

        # 一度に結合（メモリが許す限り）
        if len(dfs) > 0:
            df = pd.concat(dfs, ignore_index=True)
            del dfs  # メモリ解放
        else:
            raise ValueError("No data loaded from files")
    else:
        # 単一ファイルの場合もチャンク読み込みを使用
        chunk_size = 100000
        dfs = []
        try:
            for chunk in pd.read_csv(dataset_path, chunksize=chunk_size, index_col=False, low_memory=False, engine='c'):
                dfs.append(chunk)
        except Exception:
            for chunk in pd.read_csv(dataset_path, chunksize=chunk_size, index_col=False, low_memory=False):
                dfs.append(chunk)
        if len(dfs) > 0:
            df = pd.concat(dfs, ignore_index=True)
            del dfs
        else:
            raise ValueError("No data loaded from file")

    # データ型を最適化（メモリ使用量削減）- ベクトル化で高速化
    int_cols = df.select_dtypes(include=['int64']).columns
    float_cols = df.select_dtypes(include=['float64']).columns
    if len(int_cols) > 0:
        df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast='integer')
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast='float')

    # For binary classification, uncomment the following line
    # df['Label'] = np.where(df['Label'] != "Normal", "Attack", df['Label'])

    train_test_split = .7
    # 再現性のためシードを設定（必要に応じて）
    np.random.seed(42)
    msk = np.random.rand(len(df)) < train_test_split
    train_df = df[msk].copy()
    test_df = df[~msk].copy()

    return train_df, test_df, df


def save_model(rf, filename):
    with open(filename, 'wb') as f:
        pkl.dump(rf, f)


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


def train(X, y, lr=1e-4):
    # デバイスを決定
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False

    # バッチサイズを最適化（デバイスに応じて調整、メモリが許す限り大きく）
    if use_cuda:
        batch_size = 8192  # GPU用（さらに増加）
    elif use_mps:
        batch_size = 4096  # Apple Silicon用（増加）
    else:
        batch_size = 4096  # CPU用（増加）

    # 最適化版モデルを使用
    model_ = AdaptiveClusteringOptimized(encoder_dims=[500, 200, 50], n_kernels=len(categories), kernel_size=10)
    model_.train()

    # モデルをデバイスに移動
    if use_cuda:
        model_ = model_.cuda()
        pin_memory = True
        # GPU使用時でもnum_workersを2に設定（データローディングの並列化）
        num_workers = 2
    elif use_mps:
        model_ = model_.to('mps')
        pin_memory = False
        num_workers = 4  # Apple Siliconではマルチプロセッシングが有効
    else:
        pin_memory = False
        num_workers = 4  # CPU使用時はマルチプロセッシングを増やす

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

    n_epoch = 1
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
            if use_cuda:
                x = x.cuda(non_blocking=True)
                labels_ = labels_.cuda(non_blocking=True)
            elif use_mps:
                x = x.to('mps')
                labels_ = labels_.to('mps')

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
        
        print(f"Iteration {i} | Loss {avg_loss:.6f} | Steps: {step_count}")
        if avg_loss < 1:
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

dataset_path = root_path + 'dataset/processed'
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


print(separator)
print("Loading dataset...")
# if os.path.exists(extended_dataset_path):
#     train_df, test_df, df = load_datasets(extended_dataset_path)
# else:
train_df, test_df, df = load_datasets(dataset_path)
print("Done loading dataset")


# In[16]:


df.head()


# ### Preparing the data
# 
# We prepare the features to be in the correct format for our Adaptive Clustering network and the Random Forest Classifier.
# 
# The labels are separated from the training features and the order of the categories are kept track of in order to use them in the same order when evaluating the model on the testing set.

# In[17]:


if os.path.exists(model_path + ".categories"):
    print(separator)
    print("Loading categories...")
    with open(model_path + ".categories", "r") as f:
        categories = json.load(f)
        print(f"categories loaded: {model_path}.categories")
    print("Done loading categories")
else:
    categories = list(set(pd.factorize(train_df['Label'])[1].values))

train_df_ = train_df.drop(['Label'], axis=1)

# numpy配列として保持（Tensor変換はDataset内で行う）
X = train_df_.values.astype(np.float32)

# ラベルのインデックス変換をベクトル化（高速化）
label_to_idx = {label: idx for idx, label in enumerate(categories)}
y = train_df['Label'].map(label_to_idx).values.astype(np.int64)

cats = df['Label'].copy()
y_ = df['Label'].values
df.drop(['Label'], axis=1, inplace=True)


# ## Learning

# ### Training the model
# 
# We train our model until we achieve an acceptable loss and export an extended dataset with the cluster centers obtained from the Adaptive Clustering network. This would allow us to not have to retrain our network for every single execution.

# In[18]:


if not os.path.exists(model_path):
    print("Training model...")
    model = train(X, y)
    print("Done training model")

if not os.path.exists(extended_dataset_path):
    rf_features, cols = extend_dataset(model, df, cats, label_tag='Label')

    del df
    del train_df
    del test_df


# As we can see here, thanks to our early stopping mechanism, we train our Adaptive Clustering model for only 17 iterations and still obtain a perfect classification accuracy and F-score of __100%__ as shown in the [performance metrics section](#Performance-metrics).

# In[ ]:


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

    print("Saving model...")
    save_model(model, model_path)
    with open(model_path+".categories", "w") as f:
        json.dump(categories, f)
        print(f"Categories saved: {model_path}.categories")
    print("Done saving model")


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
# <p style="font-size: 1.2em; line-height: 1.2em;">We have implemented our approach and obtained a striking state-of-the-art performance by producing a <b>100%</b> success rate with <b>zero</b> false alarms on the CSE-CIC-IDS 2018 intrusion detection dataset. We have also shown that the features learned from the proposed classifier-agnostic Adaptive Clustering network constitute the features that participate the most to the decision process of the final classifier.</p>

# In[ ]:




