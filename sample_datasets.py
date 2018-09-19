# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 11:56:54 2018
@author: Aveedibya Dey
---
Create sample datasets for Regression Simulator App
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor)


def sample_data():
    # Generate sample data
    X_sample_rbf = np.sort(5 * np.random.rand(60), axis=0)
    y_sample_rbf = np.sin(X_sample_rbf).ravel()
    #y_sample_rbf = X_sample_rbf.ravel()
    
    # Add noise to targets
    y_sample_rbf_eps = y_sample_rbf.copy()
    y_sample_rbf_c = y_sample_rbf.copy()
    y_sample_rbf_eps[5:25] += 0.25* (-np.random.rand(20))
    y_sample_rbf_eps[50:60] += 1 * (-np.random.rand(10))
    
    # Generate sample data
    X = np.random.normal(size=100)
    y = X
    
    # Add noise to targets
    y_errors = y.copy()
    y_errors[::10] += 3*(np.full(10,0.5))
    
    X_errors = X.copy()
    X_errors[::10] += 2*(1- np.random.rand(10))
    
    y_errors_large = y.copy()
    y_errors_large[::10] += 10*(0.5 - np.random.rand(10))
    
    X_errors_large = X.copy()
    X_errors_large[::10] += 10*(0.5 - np.random.rand(10))
    
    Regressors = {
      'LinearRegression':LinearRegression(),
      'TheilSenRegressor':TheilSenRegressor(),
      'RANSACRegressor':RANSACRegressor(),
    }
    
    Datasets = {
        'df_normal': pd.DataFrame({'X': X, 'y': y}),
        'df_x_errors': pd.DataFrame({'X': X_errors, 'y': y}),
        'df_y_errors': pd.DataFrame({'X': X, 'y': y_errors}),
        'df_x_large_errors': pd.DataFrame({'X': X_errors_large, 'y': y}),
        'df_y_large_errors': pd.DataFrame({'X': X, 'y': y_errors_large}),
        'df_rbf_eps': pd.DataFrame({'X': X_sample_rbf, 'y': y_sample_rbf_eps}),
        'df_rbf_c': pd.DataFrame({'X': X_sample_rbf, 'y': y_sample_rbf_c})
    }
    
    return Regressors, Datasets

