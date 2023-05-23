from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris["data"], iris["target"], stratify=iris["target"], random_state=0
)
knc = KNeighborsClassifier(n_neighbors=3)
knc.fit(X_train, y_train)
knc.predict(X_test)
print(knc.predict(X_test) == y_test)
print("予測精度=", knc.score(X_test, y_test))
