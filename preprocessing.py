import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Data.csv') #delimeter is comma by default
#If the delimeter is not comma, we can specify it using the sep parameter, e.g. pd.read_csv('Data.csv', sep=';')

X = dataset.iloc[: , :-1].values #independent variables (features)
y = dataset.iloc[: , -1].values #dependent variable

""" Another way to do the same thing as above
X = dataset.drop('target',axis = 1)
y = dataset['target'] #target is the name of the dependent variable column
"""




"""
total_missing_entries = 0
for i in X:
    for y in i:
        if pd.isnull(y): #checking if the value is missing
            total_missing_entries += 1
print(total_missing_entries) #prints 2, as there are 2 missing entries in the dataset
"""


#Taking care of missing data

from sklearn.impute import SimpleImputer


#Taking care of missing data
# Creating an object of SimpleImputer class, we can replace missing values with mean, median or most frequent value
#In this case, we are replacing missing values with mean of the column
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

imputer.fit(X[:, 1:3]) #This function expects only numerical values, not strings
#It covers the 1st and 2nd column, but not the 3rd column
#column 0 is country, column 1 is age, column 2 is salary 

X[:, 1:3] = imputer.transform(X[:, 1:3]) #.transform function returns the updated array, which we assign back to X


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#Encoding categorical data
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder='passthrough') #remainder='passthrough' means that the other columns should remain unchanged
X = np.array(ct.fit_transform(X)) #fit_transform returns a 2D array, so we convert it to a numpy array


print(X)

from sklearn.preprocessing import LabelEncoder 
#LabelEncoder is used to encode the dependent variable, as it has only 2 categories (yes and no)

le = LabelEncoder()
y = le.fit_transform(y)



#Splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#test_size=0.2 means 20% of the data will be used for testing, and 80% for training
#random_state=1 means that the data will be shuffled in a specific way, so that the results are reproducible

from sklearn.preprocessing import StandardScaler
#Feature Scaling
sc = StandardScaler() #Only apply standardization to the numerical columns, not the dummy variable columns

X_train[:, 3:] = sc.fit_transform(X_train[:, 3:]) #fit_transform is used on training set
X_test[:, 3:] = sc.transform(X_test[:, 3:]) #transform is used on test set, as we want to use the same scaling as the training set

print(X_train)
print(X_test)