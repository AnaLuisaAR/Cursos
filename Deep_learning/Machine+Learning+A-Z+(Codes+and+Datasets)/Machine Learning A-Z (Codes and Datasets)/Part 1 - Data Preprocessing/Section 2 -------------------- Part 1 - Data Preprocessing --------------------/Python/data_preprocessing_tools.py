# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)

# Plot different scalings
import matplotlib.pyplot as plt
n_bins = 10
fig, axs = plt.subplots(5, 2, tight_layout=True)

## No scaling
X_train_base, X_test_base, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

axs[0][0].hist(X_train_base[:, 3], bins=n_bins)
axs[0][1].hist(X_train_base[:, 4], bins=n_bins)
axs[0][0].set_title('No Scaling')
axs[0][1].set_title('No Scaling')

## Standard Scaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train_base[:, 3:])
X_test[:, 3:] = sc.transform(X_test_base[:, 3:])

axs[1][0].hist(X_train[:, 3], bins=n_bins)
axs[1][1].hist(X_train[:, 4], bins=n_bins)
axs[1][0].set_title('StandardScaler')
axs[1][1].set_title('StandardScaler')

## Min Max Scaler
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train[:, 3:] = sc.fit_transform(X_train_base[:, 3:])
X_test[:, 3:] = sc.transform(X_test_base[:, 3:])

axs[2][0].hist(X_train[:, 3], bins=n_bins)
axs[2][1].hist(X_train[:, 4], bins=n_bins)
axs[2][0].set_title('MinMaxScaler')
axs[2][1].set_title('MinMaxScaler')

## Normalizer
from sklearn.preprocessing import Normalizer
sc = Normalizer()
X_train[:, 3:] = sc.fit_transform(X_train_base[:, 3:])
X_test[:, 3:] = sc.transform(X_test_base[:, 3:])

axs[3][0].hist(X_train[:, 3], bins=n_bins)
axs[3][1].hist(X_train[:, 4], bins=n_bins)
axs[3][0].set_title('Normalizer')
axs[3][1].set_title('Normalizer')

## Power Transformer
from sklearn.preprocessing import PowerTransformer
sc = PowerTransformer()
X_train[:, 3:] = sc.fit_transform(X_train_base[:, 3:])
X_test[:, 3:] = sc.transform(X_test_base[:, 3:])

axs[4][0].hist(X_train[:, 3], bins=n_bins)
axs[4][1].hist(X_train[:, 4], bins=n_bins)
axs[4][0].set_title('PowerTransformer')
axs[4][1].set_title('PowerTransformer')

plt.show()