# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

def select_from_model(data) :
    np.random.seed(9)
    feature_name=[]
    X = data.drop('SalePrice', axis=1)
    Y = data['SalePrice']
    lr = RandomForestClassifier()
    #rank all features, i.e continue the elimination until the last one
    #lr.fit(X,Y)
    
    sfm = SelectFromModel(lr)
    sfm.fit(X, Y)
 
    for feature_list_index in sfm.get_support(indices=True):
        feature_name.append(X.columns.values[feature_list_index])
    return feature_name
        
select_from_model(data)


