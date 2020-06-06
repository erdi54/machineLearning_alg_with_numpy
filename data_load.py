# csv
# numpy, 2x
# pandas

# missing data, header, dtype
import csv
import numpy as np
import pandas as pd

FILE_NAME = "spambase.data"
""",
with open(FILE_NAME, 'r') as f:
    data = list(csv.reader(f, delimiter=","))
    
    data = np.array(data)
"""
# data = np.loadtxt(FILE_NAME, delimiter=",")
data = np.genfromtxt(FILE_NAME, delimiter=',', dtype=np.float32, skip_header=1, missing_values="Hello", filling_values=0.64)

print(data.shape, type(data[0][0]))

n_sapmles, n_features = data.shape
n_features -= 1
X = data[:, 0:n_features]
y = data[:, n_features]

print(X.shape, y.shape)

print(X[0, 0:5])

print("////////////////////////////////////////////")

df = pd.read_csv(FILE_NAME, header=None, delimiter=',', dtype=np.float32, skiprows=1, na_values=["Hello"])
df = df.fillna(0.64)
data = df.to_numpy()
data = np.asarray(data, dtype=np.float32)
n_sapmles1, n_features1 = data.shape
n_features1 -= 1
X = data[:, 0:n_features1]
y = data[:, n_features1]

print(X.shape, y.shape)
print(X[0, 0:5])
