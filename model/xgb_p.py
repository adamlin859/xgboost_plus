import xgboost as xgb
import numpy as np
from sklearn.linear_model import LogisticRegression


class XGBoostPlus:
    def __init__(self, params={}, alpha=0.9):
        global alpha_
        alpha_ = alpha
        self.params = params
        
    def fit(self, X, y, Xs):
        global soft_label
        clf = LogisticRegression()
        clf.fit(Xs, y)

        soft_label = clf.predict_proba(Xs)[:,1]
        self.model = xgb.train(self.params, X, obj=logregobj) 

    def predict(self, X):
        return np.array([1 if x > 0 else 0 for x in self.model.predict(X)])

    def predict_proba(self, X):
        return self.model.predict(X)

def logregobj(preds, dtrain):
    global soft_label, alpha_
    labels = dtrain.get_label()

    preds = 1.0 / (1.0 + np.exp(-preds))

    grad = preds - labels
    hess = preds * (1.0 - preds)
    
    grad2 = preds - soft_label

        
    grad_final = (1- alpha_) * grad + (alpha_) * grad2
    
    return grad_final, hess