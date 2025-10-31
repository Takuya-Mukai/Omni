# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="93cl9JWr_5nm"
# ## 問題文
# 下記の「#common」で始まるセルの中で指定されたリンク先にあるデータ（ワインの品質）が分析対象になります。
#
# このデータを読み込み、カラムの`volatile acidity`について$n$等分（$n$はデータ数を越えず、分位数に同一の値が存在しない自然数。データ数が$n$で割り切れるとは限らず、この場合の処理は`Pandas.qcut`の処理に準じます。）にグループ分けします。次にそれぞれのグループのデータのうち、カラムの`quality`の値が`5`であるものについて、それらの`alcohol`の平均値を算出してください。さらに、ここで算出した各グループの`alcohol`の平均値の中で、1番小さい値を返り値とするような関数を作成してください。
#
# 提出するときは、以下の点に注意してください。
# >- 以下の関数`homework`の`!!WRITE ME!!`に処理を書いてください。(**「`!!WRITE ME!!`」は消して、記入してください。**)
# >- 実際の提出は記述された`homework`関数全てになり、**提出はOmnicampus内の宿題の欄から今週の課題を選択後、提出内容に関数を貼り付けてから[Pythonコード提出]を押してください。**
# >- 返り値が数値型になるようにしてください。
# >- 関数は1つにまとめてください。
# >- Omnicampus提出時にimport文は記載しない。

# %% [markdown] id="XinSt-ZnZJSM"
# 以下は共通の前処理になります。第4回のドライブにあるデータ（winequality-red.csv）を各自でダウンロードし、各自のMy Driveのディレクトリに格納し、データを読み込む形式にしています。
#
# 以下の説明などを参考にして各自の環境で適宜変更してください。

# %% id="thHL7HyXBXvS"
# common
import numpy as np
import pandas as pd
from pandas import DataFrame

# googleドライブから読み込む(自分の環境に合わせて要修正)
url_winequality_data = "./winequality-red.csv"


# %% id="2LO2_UoXl3Ed"
# working place. everything
n = 5


def homework(url_winequality_data, n):
    data: DataFrame = pd.read_csv(url_winequality_data, sep=";")
    # separate `volatile acidity` into n groups
    data["group"] = pd.qcut(data["volatile acidity"], q=n)

    group_means = data[data["quality"] == 5].groupby("group")["alcohol"].mean()

    return group_means.min()


# if __name__ == "__main__":
# homework(url_winequality_data, 4)
# %% [markdown] id="j80pDrFlIJkF"
# **謝辞**：以下のデータセットの利用に関して
# http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
#
# 引用元：Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [[http://archive.ics.uci.edu/ml](http://archive.ics.uci.edu/ml)]. Irvine, CA: University of California, School of Information and Computer Science.
#
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
