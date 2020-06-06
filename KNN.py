import numpy as np
from collections import Counter

def euclidian_distances(x1, x2):
    return  np.sqrt(np.sum((x1 - x2)**2))

class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit_method(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array((predicted_labels))

    def _predict(self, x):
        # compute distances
        distances = [euclidian_distances(x, x_train) for x_train in self.X_train]
        # get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority vote , most common class labels
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]




if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    iris =datasets.load_iris()
    X, y = iris.data, iris.target

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = KNN(k=5)
    clf.fit_method(x_train,y_train)
    prediction = clf.predict(x_test)

    acc = np.sum(prediction == y_test) / len(y_test)
    print(acc)


