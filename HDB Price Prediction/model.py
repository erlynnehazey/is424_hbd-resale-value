import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold as sk
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.preprocessing import StandardScaler, scale
from sklearn.metrics import accuracy_score as acs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import pickle

#For AdaBoost implementation
from sklearn.ensemble import AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB

#For hyperparamter tuning
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


hdb_model_data = pd.read_csv("hdb_model_data_regression.csv")
X = hdb_model_data.iloc[:,0:8]
y= hdb_model_data.iloc[:,-1]  


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
regressor = RandomForestRegressor(n_estimators = 8,  #no. of trees to be used in the model
                        max_depth = 15,  # maximum number of splits each tree can take. Too low the model will be trained less and have high bias
                        min_samples_split = 2, #every node have 2 subnodes
                        max_features = 6)#maximum features in each tree

model = regressor.fit(X_train,y_train)                   

pickle.dump(model, open('randomForest.pkl', 'wb'))