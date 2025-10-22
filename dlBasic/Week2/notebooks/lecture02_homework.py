# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] id="03FTy4lwzXiQ"
# # 第2回講義 宿題

# %% [markdown] id="JtnWqKVGzXiT"
# ## 課題
# 今回のLessonで学んだことを元に，MNISTのファッション版 (Fashion MNIST，クラス数10) をソフトマックス回帰によって分類してみましょう．
#
# Fashion MNISTの詳細については以下のリンクを参考にしてください．
#
# Fashion MNIST: https://github.com/zalandoresearch/fashion-mnist

# %% [markdown] id="zyD3F677zXiU"
# ### 目標値
# Accuracy: 80%

# %% [markdown] id="9pt9rMcxzXiU"
# ### ルール
# - 訓練データは`x_train`， `y_train`，テストデータは`x_test`で与えられます．
# - 予測ラベルは one_hot表現ではなく0~9のクラスラベル で表してください．
# - **下のセルで指定されている`x_train、y_train`以外の学習データは使わないでください．**
# - **ソフトマックス回帰のアルゴリズム部分の実装はnumpyのみで行ってください** (sklearnやtensorflowなどは使用しないでください)．
#     - データの前処理部分でsklearnの関数を使う (例えば `sklearn.model_selection.train_test_split`) のは問題ありません．

# %% [markdown] id="2bq41q5SzXiV"
# ### 提出方法
# - 2つのファイルを提出していただきます．
#     1. テストデータ (`x_test`) に対する予測ラベルをcsv形式で保存し，**Omnicampusの宿題タブから「第2回 機械学習基礎」を選択して**提出してください．
#     2. それに対応するpythonのコードを　ファイル＞ダウンロード＞.pyをダウンロード　から保存し，**Omnicampusの宿題タブから「第2回 機械学習基礎 (code)」を選択して**提出してください．pythonファイル自体の提出ではなく，「提出内容」の部分にコード全体をコピー&ペーストしてください．
#
# - なお，採点は1で行い，2はコードの確認用として利用します（成績優秀者はコード内容を公開させていただくかもしれません）．コードの内容を変更した場合は，**1と2の両方を提出し直してください**．

# %% [markdown] id="KJKR8DMtzXiW"
# ### 評価方法
# - 予測ラベルの`y_test`に対する精度 (Accuracy) で評価します．
# - 即時採点しLeader Boardを更新します（採点スケジュールは別アナウンス）．
# - 締切時の点数を最終的な評価とします．

# %% [markdown] id="RVM6MjD_4Okx"
# ### ドライブのマウント

# %% id="1oI-Fjs4btCn"
# from google.colab import drive
#
# drive.mount('/content/drive')
#
# # %% id="Oq0onk1fb_lf"
# # 作業ディレクトリを指定
# work_dir = 'drive/MyDrive/Colab Notebooks/DLBasics2025_colab'
#
# # %% [markdown] id="r20uRSCUzXiX"
# # ### データの読み込み（このセルは修正しないでください）
#
# # %% id="OZodouZWzXiX"
import os
import sys

import numpy as np
import pandas as pd

#
# sys.modules['tensorflow'] = None


def load_fashionmnist():
    # # 学習データ
    # x_train = np.load(work_dir + '/Lecture02/data/x_train.npy')
    # y_train = np.load(work_dir + '/Lecture02/data/y_train.npy')
    #
    # # テストデータ
    # x_test = np.load(work_dir + '/Lecture02/data/x_test.npy')
    # 学習データ
    x_train = np.load("../data/x_train.npy")
    y_train = np.load("../data/y_train.npy")

    # テストデータ
    x_test = np.load("../data/x_test.npy")

    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    y_train = np.eye(10)[y_train.astype("int32")]
    x_test = x_test.reshape(-1, 784).astype("float32") / 255

    return x_train, y_train, x_test


# %% [markdown] id="mHynK8xAzXid"
# ### ソフトマックス回帰の実装

# %% id="gU9E1ppuzXie"
x_train, y_train, x_test = load_fashionmnist()

import time

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- parameters ---
W_WIDTH = 0.1


def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)


class SoftmaxRegression:
    # set learning_rate as function parameter
    def __init__(
        self,
        x_train,
        y_train,
        x_valid,
        y_valid,
        learning_rate,
        W_WIDTH=0.5,
        L2_lambda=0.01,
        batch_size=32,
        epochs=500,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.num_samples, self.input_dim = x_train.shape
        self.num_classes = len(y_train[1])
        self.W_WIDTH = W_WIDTH
        self.learning_rate = learning_rate
        self.L2_lambda = L2_lambda
        self.batch_size = batch_size
        self.epochs = epochs

    def initialize_weights(self):
        self.W = np.random.uniform(
            -self.W_WIDTH, self.W_WIDTH, (self.input_dim, self.num_classes)
        )
        self.b = np.zeros(shape=(self.num_classes,)).astype("float32")

    def forward(self, X):
        return softmax(np.dot(X, self.W) + self.b)

    def predict_proba(self, X):
        # return probability of y
        return self.forward(X)

    def cross_entropy(self, y_pred_batch, y_train_batch):
        E = (-y_train_batch * np.log(y_pred_batch + 1e-10)).mean()
        L2 = self.L2_lambda * np.sum(self.W * self.W)
        # self.cost = E + L2
        return E + L2

    def update_weight(self, epoch, x_train_batch, y_train_batch, y_pred_batch):
        N = self.num_samples
        M = x_train_batch.shape[0]
        learning_rate = self.learning_rate(epoch)

        error = y_pred_batch - y_train_batch
        dW = (1 / M) * np.dot(x_train_batch.T, error) + 2 * self.L2_lambda * self.W
        db = (1 / M) * np.sum(error, axis=0)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

    def train(self):
        print("Training started...")
        self.initialize_weights()
        cost_history = []
        start_time = time.time()

        # prepare validation labels
        y_valid_label = self.y_valid.argmax(axis=1)
        num_batches = int(np.ceil(self.num_samples / self.batch_size))

        for epoch in range(self.epochs):
            indices = np.arange(self.num_samples)
            np.random.shuffle(indices)

            epoch_cost_sum = 0

            for i in range(num_batches):
                # pick up batch data
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, self.num_samples)
                batch_indices = indices[start_idx:end_idx]

                x_batch = self.x_train[batch_indices]
                y_batch = self.y_train[batch_indices]

                y_pred_batch = self.predict_proba(x_batch)

                batch_cost = self.cross_entropy(y_pred_batch, y_batch)
                epoch_cost_sum += (
                    batch_cost * x_batch.shape[0]
                )  # cross_entropy returns mean cost

                self.update_weight(epoch, x_batch, y_batch, y_pred_batch)

            avg_epoch_cost = epoch_cost_sum / self.num_samples
            cost_history.append(avg_epoch_cost)

            if (epoch + 1) % 50 == 0 or epoch == 0:

                y_pred_proba_train = self.predict_proba(self.x_train)
                y_pred_train_label = y_pred_proba_train.argmax(axis=1)
                y_train_label = self.y_train.argmax(axis=1)
                train_acc = accuracy_score(y_train_label, y_pred_train_label)

                y_pred_proba_valid = self.predict_proba(self.x_valid)
                y_pred_valid_label = y_pred_proba_valid.argmax(axis=1)
                valid_acc = accuracy_score(y_valid_label, y_pred_valid_label)
                print(
                    f"Epoch {epoch+1}/{epochs}, Avg Cost: {avg_epoch_cost}, TrainAcc: {train_acc:.6f}, ValidAcc: {valid_acc:.6f}, Elapsed time: {time.time() - start_time:.2f} sec"
                )

        print("Learning finished.")
        return cost_history


# 学習データと検証データに分割
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)
print(f"    Train samples: {x_train.shape[0]}, Valid samples: {x_valid.shape[0]}")

print("\n2. Initialize model and training the model...")


def lr_scheduler(epoch):
    return 0.015 / (1 + 0.001 * epoch)


learning_rate = 0.005
epochs = 700
print(f"Before compression: {x_train.shape[1]} features.")
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)

# centering
x_train -= np.mean(x_train, axis=0)
x_valid -= np.mean(x_valid, axis=0)
x_test -= np.mean(x_test, axis=0)

var_thresh = 0.02
variance = np.var(x_train, axis=0)
keep_idx = variance >= var_thresh
x_train = x_train[:, keep_idx]
x_valid = x_valid[:, keep_idx]
x_test = x_test[:, keep_idx]
print(f"After variance thresholding: {x_train.shape[1]} features remain.")

pca = PCA(n_components=500)
x_train = pca.fit_transform(x_train)
x_valid = pca.transform(x_valid)
x_test = pca.transform(x_test)
print(f"After PCA: {x_train.shape[1]} features remain.")


model = SoftmaxRegression(
    x_train=x_train,
    y_train=y_train,
    learning_rate=lr_scheduler,
    L2_lambda=0.00001,
    W_WIDTH=W_WIDTH,
    x_valid=x_valid,
    y_valid=y_valid,
    batch_size=32,
    epochs=epochs,
)

cost_history = model.train()

print("Training completed.")
print("\n3. Evaluating the model...")

# === Validation Accuracy ===
y_pred_proba_valid = model.predict_proba(x_valid)
y_pred_valid_label = y_pred_proba_valid.argmax(axis=1)
y_valid_label = y_valid.argmax(axis=1)

## show cost graph
import matplotlib.pyplot as plt

plt.plot(cost_history[100:])
plt.show()

y_pred = model.predict_proba(x_test)

# count each label distribution
unique, counts = np.unique(y_pred.argmax(axis=1), return_counts=True)
print("Predicted label distribution on test data:")
for label, count in zip(unique, counts):
    print(f"  Class {label}: {count} samples")


submission = pd.Series(y_pred.argmax(axis=1), name="label")
submission.to_csv("./submission_pred_02.csv", header=True, index_label="id")
