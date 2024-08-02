import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA

from model.xgb_p import XGBoostPlus
from sklearn.metrics import roc_auc_score
from model.svm_p import SVMPlus
from model.gb_p import GB_P
from model.ipl import IPL

def get_pcb_data():

    data = pd.read_csv("data/pcbseq.csv", index_col=0)

    # this is the x data
    first_visit = data.loc[data.groupby('id').day.idxmin()] 
    first_visit.sort_values(by="id", inplace=True)

    # this is the privleged information (x*)
    last_visit = data.loc[data.groupby('id').day.idxmax()]
    last_visit.sort_values(by="id", inplace=True)

    # one hot encode the categorical data
    first_visit = pd.get_dummies(first_visit, prefix=["sex"])
    last_visit = pd.get_dummies(last_visit, prefix=["sex"])

    # creating the label 1: liver transplant, 2: death
    last_visit["OUTCOME"] = 0
    last_visit.loc[last_visit.status.isin([1, 2]), "OUTCOME"] = 1
    y = last_visit.OUTCOME.values

    # drop the columns that are not needed
    X = first_visit.drop(["id", "futime", "status", "day"], axis=1).values

    # impute missing values
    imp = SimpleImputer(strategy="mean")
    X = imp.fit_transform(X)

    Xs = last_visit.drop(["id", "futime", "status","day", "OUTCOME"], axis=1).values
    # impute missing values
    Xs = imp.fit_transform(Xs)

    return X, y, Xs

def experiment3():
    X, y, Xs = get_pcb_data()

    n_epochs = 100
    XGB = np.zeros((n_epochs,2))
    XGB_p = np.zeros((n_epochs,2))
    SVM = np.zeros((n_epochs,2))
    SVM_p = np.zeros((n_epochs,2))
    GB_p = np.zeros((n_epochs,2))
    IPL_p = np.zeros((n_epochs,2))

    for rep in range(n_epochs):
        X_train, X_test, Xs_train, _, y_train, y_test =  train_test_split(X, Xs, y, test_size=0.3, random_state=rep)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        sc = StandardScaler()
        Xs_train = sc.fit_transform(Xs_train)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # XGBoost model
        xgb_model = xgb.train({'objective':'binary:logistic',"eval_metric":'logloss'}, dtrain)
        preds_prob = xgb_model.predict(dtest)
        y_pred = np.array([1 if x > 0.5 else 0 for x in preds_prob])
        xgb_acc = np.mean(y_pred == y_test)
        xgb_roc = roc_auc_score(y_test, preds_prob)


        # XGBoost+ model
        xgbp_model = XGBoostPlus()
        xgbp_model.fit(dtrain, y_train, Xs_train)
        preds_prob = xgbp_model.predict_proba(dtest)
        y_pred = xgbp_model.predict(dtest)
        xgbp_acc = np.mean(y_pred == y_test)
        xgbp_roc = roc_auc_score(y_test, preds_prob)

        # GB+ model 
        gb_model = GB_P() 
        # gb_model.grid_search(X_train, Xs_train, y_train)
        gb_model.fit(X_train, Xs_train, y_train)
        preds_prob = gb_model.predict_proba(X_test)
        y_pred = gb_model.predict(X_test)
        gb_acc = np.mean(y_pred == y_test)
        gb_roc = roc_auc_score(y_test, preds_prob)   

        # IPL model 
        ipl_model = IPL() 
        pca = PCA(n_components=2)
        pca.fit(Xs_train)
        xs_tr = pca.transform(Xs_train)
        # ipl_model.grid_search(X_train, xs_tr, y_train)
        ipl_model.fit(X_train, xs_tr, y_train)
        preds_prob = ipl_model.predict_proba(X_test)
        y_pred = ipl_model.predict(X_test)
        ipl_acc = np.mean(y_pred == y_test)
        ipl_roc = roc_auc_score(y_test, preds_prob)   


        # SVM model implementation using label 1 and -1
        y_train_svm = np.array([1 if x== 1 else -1 for x in y_train])
        y_test_svm = np.array([1 if x== 1 else -1 for x in y_test])
        

        # SVM
        svm = SVMPlus('pcb')
        success = svm.train(X_train, y_train_svm, params = {'gamma_w': 1.0, 'gamma_w_star': 0.1})
        preds, preds_probas = svm.predict(X_test)
        svm_acc = np.mean(preds == y_test_svm)
        svm_roc = roc_auc_score(y_test_svm, preds_probas)

        # SVM+
        svm_p = SVMPlus('pcb')
        success = svm_p.train(X_train, y_train_svm, x_star = Xs_train, params = {'gamma_w': 1.0, 'gamma_w_star': 0.1})
        preds, preds_probas = svm_p.predict(X_test)
        svm_p_acc = np.mean(preds == y_test_svm)
        svm_p_roc = roc_auc_score(y_test_svm, preds_probas)



        XGB[rep,:] += [xgb_acc, xgb_roc]
        XGB_p[rep,:] += [xgbp_acc, xgbp_roc]
        SVM[rep,:] += [svm_acc, svm_roc]
        SVM_p[rep,:] += [svm_p_acc, svm_p_roc]
        GB_p[rep,:] += [gb_acc, gb_roc]
        IPL_p[rep,:] += [ipl_acc, ipl_roc]




    XGB_means = XGB.mean(axis=0).round(3)
    XGB_stds  = XGB.std(axis=0).round(3)

    XGB_p_means = XGB_p.mean(axis=0).round(3)
    XGB_p_stds  = XGB_p.std(axis=0).round(3)

    SVM_means = SVM.mean(axis=0).round(3)
    SVM_stds  = SVM.std(axis=0).round(3)

    SVM_p_means = SVM_p.mean(axis=0).round(3)
    SVM_p_stds  = SVM_p.std(axis=0).round(3)

    GB_p_means = GB_p.mean(axis=0).round(3)
    GB_p_stds  = GB_p.std(axis=0).round(3)

    IPL_means = IPL_p.mean(axis=0).round(3)
    IPL_stds  = IPL_p.std(axis=0).round(3)


    result = pd.DataFrame([XGB_means, XGB_stds, XGB_p_means, XGB_p_stds, GB_p_means, GB_p_stds, IPL_means, IPL_stds, SVM_means, SVM_stds, SVM_p_means, SVM_p_stds], columns=['Accuracy', 'AUC'], 
                          index=['XGBoost', 'XGBoost std', 'XGBoost+', 'XGBoost+ std', "GB+", "GB+ std", "IPL", "IPL std", "SVM", "SVM std", "SVM+", "SVM+ std"])
    result.to_csv('results/experiment3.csv')
    print(result)
    



