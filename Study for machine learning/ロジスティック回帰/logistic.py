import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split　

"""
https://qiita.com/0NE_shoT_/items/b702ab482466df6e5569
変数	         説明
sepal_length	がく片の長さ
sepal_width	    がく片の幅
petal_length	花弁の長さ
petal_width	    花弁の幅
species	        品種

https://best-biostatistics.com/correlation_regression/logistic.html
"""

iris_df = sns.load_dataset("iris")  # データセットの読み込み

# 簡単のため、2品種に絞る
iris_df = iris_df[
    (iris_df["species"] == "versicolor") | (iris_df["species"] == "virginica")
]

# # データを確認する
# sns.pairplot(iris_df, hue="species")
# plt.show()

X = iris_df[["petal_length"]]
# versicolorをクラス0, virginicaをクラス1とする
Y = iris_df["species"].map({"versicolor": 0, "virginica": 1})

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)  # 80%のデータを学習データに、20%を検証データにする

lr = LogisticRegression()  # ロジスティック回帰モデルのインスタンスを作成
lr.fit(X_train, Y_train)  # ロジスティック回帰モデルの重みを学習

print("切片")