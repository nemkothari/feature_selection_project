# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def rf_rfe(data) :
    X = data.drop('SalePrice', axis=1)
    Y = data['SalePrice']
    #use linear regression as the model
    lr = RandomForestClassifier()
    #rank all features, i.e continue the elimination until the last one
    lr.fit(X,Y)
    rfe = RFE(lr, n_features_to_select=X.shape[1]/2 )
    rfe.fit(X,Y)
 
    li= zip(map(lambda x: round(x, 4), rfe.ranking_), X.columns.values)
    ilist=[]
    for i , j  in li:
        if i ==1 :
            ilist.append(j)
    return ilist



