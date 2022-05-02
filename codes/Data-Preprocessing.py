# Importing dataset
print("IMPORTING DATASET")
import pandas as pd
dataset = pd.read_csv("../datasets/data.csv")
print("___ Dataset ___\n",dataset)
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values
print("___ X ___\n",X)

# Handling missing values
print("\nHANDLING MISSING VALUES")
import numpy as np
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print("___ X ___\n",X)

# Encoding categorical data - Preprocessing
print("\nENCODING CATEGORICAL DATA - PREPROCESSING")
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
print("___ X ___\n",X)

# Splitting dataset in Training sets and Test sets
print("\nSPLITTING DATASET IN TRAINING SETS AND TEST SETS")
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print("___ X_train ___\n",X_train)
print("___ X_test ___\n",X_test)
print("___ Y_train ___\n",Y_train)
print("___ Y_test ___\n",Y_test)

# Feature Scaling
print("\nFEATURE SCALING")
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print("___ X_train ___\n",X_train)
print("___ X_test ___\n",X_test)
