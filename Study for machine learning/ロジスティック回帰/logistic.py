import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix

"""
https://qiita.com/0NE_shoT_/items/b702ab482466df6e5569
変数	         説明
sepal_length	がく片の長さ
sepal_width	    がく片の幅
petal_length	花弁の長さ
petal_width	    花弁の幅
species	        品種

https://best-biostatistics.com/correlation_regression/logistic.html
グラフに関して
https://python.atelierkobato.com/logistic/
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
# 結果を出力
print("傾き = ", lr.coef_[0][0])
print("切片 = ", lr.intercept_[0])

Y_score = lr.predict_proba(X_test)[:, 1]  # 検証データがクラス1に属する確率
Y_pred = lr.predict(X_test)  # 検証データのクラスを予測
fpr, tpr, thresholds = roc_curve(y_true=Y_test, y_score=Y_score)


print("confusion matrix = \n", confusion_matrix(y_true=Y_test, y_pred=Y_pred))
# plt.plot(fpr, tpr, label="roc curve (area = %0.3f)" % auc(fpr, tpr))
# plt.plot([0, 1], [0, 1], linestyle="--", label="random")
# plt.plot([0, 0, 1], [0, 1, 1], linestyle="--", label="ideal")
# plt.legend()
# plt.xlabel("false positive rate")
# plt.ylabel("true positive rate")
# plt.show()
