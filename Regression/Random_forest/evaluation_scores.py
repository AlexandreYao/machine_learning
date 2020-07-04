import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             max_error, r2_score)
import matplotlib.pyplot as plt


def min_error(y_true, y_pred):
    return np.min(np.abs(y_true - y_pred))

def percentage_error(y_true, y_pred, q):
    y_temp = np.abs(y_true - y_pred)
    return round(np.quantile(y_temp, q / 100), 3)

def my_score_neg(y_true, y_pred):
    d = y_true - y_pred
    pred_neg = d[(d < 0)]
    s = (pred_neg).mean()
    return s

def my_score_pos(y_true, y_pred):
    d = y_true - y_pred
    pred_pos = d[(d >= 0)]
    s = pred_pos.mean()
    return s

def get_regression_metrics(y_true, y_pred):
    regr_metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred)**0.5,
        'min_error': min_error(y_true, y_pred),
        'max_error': max_error(y_true, y_pred),
        'R^2': r2_score(y_true, y_pred),
        '25% error': percentage_error(y_true, y_pred, 25),
        '50% error': percentage_error(y_true, y_pred, 50),
        '95% error': percentage_error(y_true, y_pred, 95),
        'MNE': my_score_neg(y_true, y_pred),
        'MPE': my_score_pos(y_true, y_pred)
    }
    # return reg_metrics
    df_regr_metrics = pd.DataFrame.from_dict(regr_metrics, orient='index')
    df_regr_metrics.columns = ['Value']
    return df_regr_metrics

def plot_true_pred(y_true, y_pred):
    y_pred = y_pred.sort_values()
    y_true = y_true[y_pred.index.tolist()]
    plt.figure()
    plt.scatter(y_true, np.arange(y_true.shape[0]), marker='+', color='g')
    plt.scatter(y_pred, np.arange(y_pred.shape[0]), marker='x', color='r')
    plt.grid()
    plt.xlabel('Value')
    plt.ylabel('Obs.')
    plt.title('y_true VS y_pred')
    plt.legend(['y_true', 'y_pred'])
    plt.show()

def plot_residual(y_true, y_pred):
    plt.scatter(
        y_pred, y_true - y_pred, c='lightgreen', marker='.', label='Test data')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    #plt.legend(loc='upper left')
    plt.hlines(
        y=0, xmin=y_pred.min() - 5, xmax=y_pred.max() + 10, color='red', lw=2)
    plt.title('Test data  -- Model Residuals')
    plt.grid(True)
    plt.show()