
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import pandas as pd

from model.xgb_p import XGBoostPlus
from model.svm_p import SVMPlus
from model.ipl import IPL
from model.gb_p import GB_P


def synthetic_train(a,n, J, R):
    x  = np.random.randn(n,a.size)
    xs = np.copy(x)
    xs = xs[:,0:J]
    a  = a[0:J]
    y  = (np.dot(xs,a) > 0).ravel()
    xs = xs[:,R:J]
    return (xs,x,y)

def synthetic_test(a,n, J, R):
    x  = np.random.randn(n,a.size)
    xs = np.copy(x)
    xs = xs[:,R:J]
    a  = a[R:J]
    y  = (np.dot(xs,a) > 0).ravel()
    return (x,y)


def synthetic_exp(x_tr,xs_tr,y_tr,x_te,y_te):
    # normalizing the data
    s_x   = StandardScaler().fit(x_tr)
    s_s   = StandardScaler().fit(xs_tr)
    x_tr  = s_x.transform(x_tr)
    x_te  = s_x.transform(x_te)
    xs_tr = s_s.transform(xs_tr)

    # creating the input to xgboost model
    dtrain = xgb.DMatrix(x_tr, label=y_tr)
    dtest = xgb.DMatrix(x_te, label=y_te)
    dstrain = xgb.DMatrix(x_tr, label=y_tr)

    # XGBoost model
    xgb_model = xgb.train({'objective':'binary:logistic',"eval_metric":'logloss'}, dtrain, )
    preds_prob = xgb_model.predict(dtest)
    y_pred = np.array([1 if x > 0.5 else 0 for x in preds_prob])
    xgb_acc = np.mean(y_pred == y_te)
    xgb_roc = roc_auc_score(y_te, preds_prob)


    # XGBoost+ model
    xgbp_model = XGBoostPlus()
    xgbp_model.fit(dtrain, y_tr, xs_tr)
    preds_prob = xgbp_model.predict_proba(dtest)
    y_pred = xgbp_model.predict(dtest)
    xgbp_acc = np.mean(y_pred == y_te)
    xgbp_roc = roc_auc_score(y_te, preds_prob)

    # GB+ model 
    gb_model = GB_P() 
    y_tr = y_tr * 1.0
    # gb_model.grid_search(x_tr, xs_tr, y_tr)
    gb_model.fit(x_tr, xs_tr, y_tr)
    preds_prob = gb_model.predict_proba(x_te)
    y_pred = gb_model.predict(x_te)
    gb_acc = np.mean(y_pred == y_te)
    gb_roc = roc_auc_score(y_te, preds_prob)   

    # IPL model 
    ipl_model = IPL() 
    # gb_model.grid_search(x_tr, xs_tr, y_tr)
    ipl_model.fit(x_tr, xs_tr, y_tr)
    preds_prob = ipl_model.predict_proba(x_te)
    y_pred = ipl_model.predict(x_te)
    ipl_acc = np.mean(y_pred == y_te)
    ipl_roc = roc_auc_score(y_te, preds_prob)   


    # convert labels to -1 and 1
    y_te = np.array([1 if x == 1 else -1 for x in y_te])
    y_tr = np.array([1 if x == 1 else -1 for x in y_tr])

    svmplus = SVMPlus('synthetic')
    p = svmplus.hyper_parameters()[1]
    success = svmplus.train(x_tr, y_tr, params=p, x_star=xs_tr)
    preds, preds_probas = svmplus.predict(x_te)

    svmplus_acc =  np.mean(preds == y_te)
    svmplus_roc =  roc_auc_score(y_te, preds_probas)

    svm = SVMPlus('synthetic')
    p = svmplus.hyper_parameters()[-1]
    success = svm.train(x_tr, y_tr, params=p)
    preds, preds_probas = svm.predict(x_te)
    svm_acc = np.mean(preds == y_te)
    svm_roc = roc_auc_score(y_te, preds_probas)


    return [ xgb_acc, xgb_roc, xgbp_acc, xgbp_roc,gb_acc, gb_roc, ipl_acc, ipl_roc, svm_acc, svm_roc, svmplus_acc, svmplus_roc]


def expriment1():
    # This experiment corresponds the the result of Table 1 in the paper
    # XGBoost+ vs XGBoost, |H| = 2 and |J| = 3
    d      = 50
    n_tr   = 200
    n_te   = 1000
    n_epochs = 100
    eid    = 0

    np.random.seed(1)

    eid += 1
    XGB = np.zeros((n_epochs,2))
    XGB_p = np.zeros((n_epochs,2))
    GB_p = np.zeros((n_epochs,2))
    IPL = np.zeros((n_epochs,2))
    SVM = np.zeros((n_epochs,2))
    SVM_p = np.zeros((n_epochs,2))

    for rep in range(n_epochs):
        a   = np.random.randn(d)
        (xs_tr,x_tr,y_tr) = synthetic_train(a,n_tr, 3, 1)
        (x_te,y_te) = synthetic_test(a,n_te, 3, 1)
        exp_result = synthetic_exp(x_tr,xs_tr,y_tr,x_te,y_te)
        XGB[rep,:] += exp_result[:2]
        XGB_p[rep,:] += exp_result[2:4]
        GB_p[rep,:] += exp_result[4:6]
        IPL[rep,:] += exp_result[6:8]
        SVM[rep,:] += exp_result[8:10]
        SVM_p[rep,:] += exp_result[10:12]

    XGB_means = XGB.mean(axis=0).round(3)
    XGB_stds  = XGB.std(axis=0).round(3)

    XGB_p_means = XGB_p.mean(axis=0).round(3)
    XGB_p_stds  = XGB_p.std(axis=0).round(3)

    GB_p_means = GB_p.mean(axis=0).round(3)
    GB_p_stds  = GB_p.std(axis=0).round(3)

    IPL_means = IPL.mean(axis=0).round(3)
    IPL_stds  = IPL.std(axis=0).round(3)

    SVM_means = SVM.mean(axis=0).round(3)
    SVM_stds  = SVM.std(axis=0).round(3)

    SVM_p_means = SVM_p.mean(axis=0).round(3)
    SVM_p_stds  = SVM_p.std(axis=0).round(3)


    
    
    result = pd.DataFrame([XGB_means, XGB_stds, XGB_p_means, XGB_p_stds, GB_p_means, GB_p_stds, IPL_means, IPL_stds, SVM_means, SVM_stds, SVM_p_means, SVM_p_stds], columns=['Accuracy', 'AUC'], 
                          index=['XGBoost', 'XGBoost std', 'XGBoost+', 'XGBoost+ std', "GB+", "GB+ std", "IPL", "IPL std", "SVM", "SVM std", "SVM+", "SVM+ std"])
    result.to_csv('results/experiment1.csv')
    print(result)

def expriment2():
    d      = 50
    n_tr   = 200
    n_te   = 1000
    n_epochs = 100
    eid    = 0

    np.random.seed(1)
    eid += 1

    XGB = np.zeros((10, 10))
    XGB_p = np.zeros((10, 10))
    for J in range(1, 10):
        for r in range(0, J):
            R = np.zeros((n_epochs,2))
            for rep in range(n_epochs):
                a   = np.random.randn(d)
                (xs_tr,x_tr,y_tr) = synthetic_train(a, n_tr, J, r)
                (x_te,y_te) = synthetic_test(a, n_te, J, r)
                exp_result = synthetic_exp(x_tr,xs_tr,y_tr,x_te,y_te)
                R[rep,:] += [exp_result[0], exp_result[2]]

            means = R.mean(axis=0).round(3)
            
            XGB[r, J] = means[0]
            XGB_p[r, J] = means[1]

    table = []
    for i in range(10):
        table.append(XGB[i])
        table.append(XGB_p[i])

    df = pd.DataFrame(table)
    
    col0 = []
    col1 = []
    for i in range(10):
        col0.append("XGB")
        col0.append("XGB+")
        col1.append(i)
        col1.append(i)

    df[0] = col1
    df.index = col0 
    df.to_csv('results/experiment2.csv')
