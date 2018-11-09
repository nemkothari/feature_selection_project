# %load q05_forward_selected/build.py
# Default imports
import pandas as pd
from sklearn.linear_model import LinearRegression
import sklearn.metrics 
data = pd.read_csv('data/house_prices_multivariate.csv')

model = LinearRegression()
#print(data.drop(remov,axis=1 ))
def forward_selected(data,model):
    remov = ['SalePrice']
    ffit=[]
    Variable_1=[]
    Variable_2=[]
    fVariable_1=[]
    fVariable_2=[]
    r2_scoref =0 
    while len(remov) != len(data.drop('SalePrice',axis=1 ).columns.values ):
        if len(Variable_1) > 0 :
            remov.append(Variable)
            ffit.append(Variable)
            
        X = data.drop(remov,axis=1 )
        y_true = data['SalePrice']
        
        
        for fet in X.columns.values :
            ffit.append(fet)
            
            X_True =data[ffit]
            model.fit(X_True,y_true)
           
            y_pred = model.predict(X_True)
            r2_score = sklearn.metrics.r2_score(y_true, y_pred)
            if r2_score > r2_scoref :
                Variable = str(fet)
                Variable_1.append(fet)
                Variable_2.append(r2_score)
                r2_scoref = r2_score
            ffit.remove(fet)
            
        fVariable_1.append(Variable)
        fVariable_2.append(r2_scoref)
    return fVariable_1 , fVariable_2




