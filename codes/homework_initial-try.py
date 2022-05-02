# Importing dataset
from pandas import read_csv
import numpy as np
column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
data = read_csv("../datasets/boston-housing.csv", header=None, delimiter=r"\s+", names=column_names)

# Keeping only our required data
boston_dataframe = data[['RM','AGE']]
print (boston_dataframe.head(5))
print (np.shape(boston_dataframe))
print (boston_dataframe.describe())

# Discovering outliers by box plot
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in boston_dataframe.items():
    sns.boxplot(y=k, data=boston_dataframe, ax=axs[index])
    index+=1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()

# Discovering outliers by Z-Score
ZScore = np.abs(stats.zscore(boston_dataframe))
print(ZScore)

# Deciding outliers
print(np.where(ZScore > 1))
print(np.where(ZScore > 2))
print(np.where(ZScore > 3))
print(np.where(ZScore > 4))
print(np.where(ZScore > 5))
print(np.where(ZScore > 6))

# Removing outliers
boston_dataframe_o = boston_dataframe[(ZScore<2).all(axis=1)]
print (np.shape(boston_dataframe),np.shape(boston_dataframe_o))

# Visualising data after removing outliers
fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in boston_dataframe_o.items():
    sns.boxplot(y=k, data=boston_dataframe_o, ax=axs[index])
    index+=1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()
