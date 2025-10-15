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
#     name: python3
# ---

# %% [markdown] id="93cl9JWr_5nm"
# ## 問題文
# ある箱のwidth(幅)、depth(奥行き)、height(高さ)を引数として、その箱の体積が1000以上であればTrue, 1000未満であればFalseを出力する関数を実装してください。単位は全て省略とします。
#
# 以下が想定される出力です
#
# ```python
# width = 10
# depth = 20
# height = 35
# print(homework(width, depth, height))
#
# ---------
# True
# ```
#
# 提出するときは、以下の点に注意してください。適宜例題(次のセクションにあります)を参考にしてください。Omnicampusには例題の回答を提出する欄はありません。
# >- 以下の関数`homework`の`!!WRITE ME!!`に処理を書いてください。(**「`!!WRITE ME!!`」は消して、記入してください。**)
# >- 実際の提出は記述された`homework`関数全てになり、**提出はOmnicampus内の宿題の欄から今週の課題を選択後、提出内容に関数を貼り付けてから[Pythonコード提出]を押してください。**
# >- 関数は1つにまとめてください。

# %% id="thHL7HyXBXvS"
# !!WRITE ME!!に処理を記入する（このセルのhomework関数のみを提出することに注意）
def homework(width, depth, height):

    !!WRITE ME!!

    return my_result


# %% colab={"base_uri": "https://localhost:8080/"} id="Nk2WvS3aE8QA" outputId="d675042b-3c3a-44a7-e58f-c3db777d32e7" executionInfo={"status": "ok", "timestamp": 1745332492334, "user_tz": -540, "elapsed": 15, "user": {"displayName": "\u9ad8\u5c71\u4e00\u6a39", "userId": "18442637108375737583"}}
# homework関数の動作確認をします（このコードは提出しません）
width = 10
depth = 20
height = 35
print(homework(width, depth, height))

# %% colab={"base_uri": "https://localhost:8080/"} id="cO5VE-peExXR" outputId="37170807-51eb-4c8c-a6d6-b6bb4c348b98" executionInfo={"status": "ok", "timestamp": 1745332494077, "user_tz": -540, "elapsed": 9, "user": {"displayName": "\u9ad8\u5c71\u4e00\u6a39", "userId": "18442637108375737583"}}
# homework関数の動作確認をします（このコードは提出しません）
width = 10
depth = 10
height = 9
print(homework(width, depth, height))
