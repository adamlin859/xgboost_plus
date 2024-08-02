from re import L
import cvxpy as cvx
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, auc


class SVMPlus:

    def __init__(self, datasetName):

        # model name
        self.name = 'svm+'
        self.linestyle = '-'
        self.color = 'r'

        self.datasetName = datasetName

        # prediction decision threshold
        self.pred_thresh = 0

        # model parameters
        self.w = None
        self.b = None

    def hyper_parameters(self):

        # generate hyper-parameter test space
        gamma = np.logspace(-8, 2, 11)

        # return list of parameter dictionaries
        p = []
        for i in range(len(gamma)):
            for j in range(len(gamma)):
                p.append({'gamma_w': gamma[i], 'gamma_w_star': gamma[j]})

        return p

    def train(self, x, y, params=None, x_star=None):

        # default parameters
        if params is None:
            params = {'gamma_w': 1e-3,
                      'gamma_w_star': 1e-3}

        # ensure labels are [-1, 1]
        y[y == 0] = -1
        assert np.unique(y).tolist() == [-1, 1]

        # ensure y is a m x 1 vector
        if len(y.shape) < 2:
            y = np.expand_dims(y, axis=-1)
        assert y.shape[1] == 1

        # if x* not supplied just make it ones
        if x_star is None:
            x_star = np.ones([x.shape[0], 1])

        # regularization parameter
        gamma_w = params['gamma_w'] / x.shape[1]
        gamma_w_star = params['gamma_w_star'] / x_star.shape[1]

        # define model variables
        w = cvx.Variable((x.shape[1], 1))
        b = cvx.Variable()
        w_star = cvx.Variable((x_star.shape[1], 1))
        d = cvx.Variable()

        # compute slack
        u = x_star @ w_star - d

        # balanced slack term
        slack = cvx.sum(cvx.multiply(u, y + 1)) / cvx.sum(y + 1) + \
                cvx.sum(cvx.multiply(u, y - 1)) / cvx.sum(y - 1)

        # define objective
        obj = cvx.Minimize(gamma_w * cvx.sum_squares(w) + gamma_w_star * cvx.sum_squares(w_star) + slack)


        # define constraints
        constraints = [cvx.multiply(y, (x @ w - b)) >= 1 - u,
                       u >= 0]



        # form problem and solve
        prob = cvx.Problem(obj, constraints)

        # ‘ECOS’, ‘SCS’, or ‘OSQP’.
        if self.datasetName == 'cancer':
            prob.solve(verbose=False, solver = 'SCS' )
        else:
            prob.solve(verbose=False, solver = 'ECOS' )

        try:

            # throw error if not optimal
            assert prob.status == 'optimal'

            # save model parameters
            self.w = np.array(np.squeeze(w.value, axis=-1))
            self.b = np.array(b.value)

            # return success
            return True

        except:

            # return failure
            return False

    def predict(self, x):

        # make prediction
        y_prob  = x @ self.w - self.b - self.pred_thresh
        y_hat = np.squeeze(np.sign(y_prob))

        return y_hat, y_prob