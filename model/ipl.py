import xgboost as xgb
import numpy as np
from typing import Any, Tuple
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd


class IPL:
    def __init__(self, lr=0.1, num_rounds=10, c1=0.5, c2 = 0.5, tree_params={'max_depth': 3}):

        self.lr = lr
        self.num_rounds = num_rounds
        self.c1 = c1
        self.c2 = c2
        self.tree_params = tree_params  
        
        self.loss = [0] * (num_rounds+1)

    
    def fit(self, X, Xs, y):
        self.trees = []
        pf_trees = []
        
        self.lo_0 = np.log((y == 1).sum() / (y == 0).sum())
        self.p_0 = np.exp(self.lo_0)/(np.exp(self.lo_0)+1)
        self.leaf_values = {}

        log_odds = np.zeros([self.num_rounds+1, len(y)])
        log_odds[0] = [self.lo_0] * len(y)

        loss0=-(y*np.log(self.p_0)+(1-y)*np.log(1-self.p_0))
        self.loss[0] = loss0.sum()

        probs = np.array([self.p_0] * len(y))

        
        self.weights_priv = np.random.rand(1, Xs.shape[1])

        try:
            for i in range(self.num_rounds):
                residuals = y - probs 
                residuals_star = residuals + ((self.c1/(self.c1+1)) * self.weights_priv @ Xs.transpose())[0]

                dt = DecisionTreeRegressor(**self.tree_params)
                dt.fit(X, residuals_star)
                self.trees.append(dt)

                leaf_gamma = get_gamma(dt, X, probs)
                self.leaf_values[i] = leaf_gamma

                gamma = np.array([leaf_gamma[i] for i in dt.apply(X)])


                log_odds[i + 1] = log_odds[i] + self.lr * gamma

                self.loss[i+1]=np.sum(-y * log_odds[i+1] + np.log(1+np.exp(log_odds[i+1])))

                probs = np.array([np.exp(odds)/(np.exp(odds)+1) for odds in log_odds[i+1]])
                probs = np.clip(probs, 1e-15, 1 - 1e-15)

                A = probs - residuals
                self.weights_priv = (self.c1/(self.c1+self.c2)) * ((A.transpose()@Xs) @ (np.linalg.inv(Xs.transpose()@Xs)))
                
        except Exception as e:
            return False
        
        return True

    def grid_search(self, X, Xs, y):
        max_auc = 0
        best_params = {}
        for lr in [0.001, 0.01, 0.1]:
            for c1 in [0.1, 0.5, 0.9]:
                for num_rounds in [10, 50, 100]:
                    for tree_params in [{'max_depth': 1}, {'max_depth': 3}, {'max_depth': 5}]:
                        for c2 in [0.1, 0.5, 0.9]:
                            self.tree_params = tree_params
                            self.lr = lr
                            self.c1 = c1
                            self.c2 = c2
                            self.num_rounds = num_rounds
                            
                            X_train, X_test, Xs_train, Xs_test, y_train, y_test = train_test_split(X, Xs, y)
                            outcome = self.fit(X_train, Xs_train, y_train)
                            
                            if outcome == False:
                                continue
                            
                            self.predict_proba(X_test)
                            auc = roc_auc_score(y_test, self.predict_proba(X_test))
                            if auc > max_auc:
                            

                                max_auc = auc
                                best_params = {'lr': lr, 'c1': c1, 'num_rounds': num_rounds, 'tree_params': tree_params, 'c2': c2}
        
        self.lr = best_params['lr']
        self.c1 = best_params['c1']
        self.c2 = best_params['c2']
        self.num_rounds = best_params['num_rounds']
        self.tree_params = best_params['tree_params']
        self.best_params = best_params


    def predict_proba(self, X):

        log_odds = np.zeros([self.num_rounds+1, len(X)])
        log_odds[0] = [self.lo_0] * len(X)
        

        for i in range(self.num_rounds):
            gamma = np.array([self.leaf_values[i][j] for j in self.trees[i].apply(X)])
            log_odds[i + 1] = log_odds[i] + self.lr * gamma

        probs = np.array([np.exp(odds)/(np.exp(odds)+1) for odds in log_odds[i+1]])
        return probs

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int) 

def get_gamma(dt, X, probs):
    leaf_idx = dt.apply(X)
    unique_leaves=np.unique(leaf_idx)
    leaf_gamma = {}

    for leaf in unique_leaves:
        leaf_mask = np.where(leaf_idx == leaf)[0]
        leaf_probs = probs[leaf_mask]
        leaf_val = dt.tree_.value[leaf][0][0] 
        
        leaf_gamma[leaf] = leaf_val * len(leaf_mask) / np.sum(leaf_probs * (1-leaf_probs))

    return leaf_gamma
