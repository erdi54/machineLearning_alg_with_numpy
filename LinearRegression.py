import numpy as np

def mse(y_true, y_pred):
   return np.mean((y_true-y_pred)**2)

class LinearRegression:

    def __init__(self, lr= 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        # gradient descent
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            # compute gradients
            dw = (1/ n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X):
        y_approximated  = np.dot(X, self.weights) + self.bias
        return y_approximated


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    X,y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    linear_regression = LinearRegression()

    linear_regression.fit(x_train, y_train)
    predicted = linear_regression.predict(x_test)

    mse_value = mse(y_test, predicted)
    print(mse_value)





