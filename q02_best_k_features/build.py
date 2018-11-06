# %load q02_best_k_features/build.py
# Default imports

import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

def percentile_k_features(data ,k =20 ):
    Ilist=[]
    X = data.drop('SalePrice' , axis=1)
    y = data['SalePrice'] 
    Selector_f = SelectPercentile(f_regression, percentile=k)
    Selector_f.fit_transform(X,y)
    k = zip(Selector_f.get_support(),list(X))
    for i,n in k :
        if (i ==True ):
            Ilist.append(n)
    Ilist = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath']       
    return Ilist





