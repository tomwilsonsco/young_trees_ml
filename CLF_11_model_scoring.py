#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 08:59:24 2017

@author: ubuntu
"""

#==============================================================================
# Model Scoring
#==============================================================================



#==============================================================================
# Import Modules
#==============================================================================
import numpy as np
import os
#import cPickle as pickle
import dill as pickle
import itertools
import matplotlib.pyplot as plt
import random
import pandas as pd
import rasterio
import copy
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score as roc
from sklearn.metrics import f1_score
import xgboost as xgb

## Just for runniong on spyder
#import sys
#sys.path.append('/home/ubuntu/Documents/Model_Build')
#from CLF_9helper_build_feature_functions import *

WORK_DIR = "/home/ubuntu/Documents/Model_Build_RAW/"
experiment_name = 'clf_lbp_entr_dop10_temporal'
dataframe_local_path = os.path.join(WORK_DIR, 'Polygon_Reference_DataFrame', 'Filtered_Polygon_DataFrame.csv')

# Load model predictions
save_filename = os.path.join(WORK_DIR, 'Models', experiment_name+'.pkl')
with open(save_filename, 'r') as fp:
        features, f1s, best_model, best_estimators, p_valid_ds, cfmt, classes, report, meta_data, (X_train_ds, y_train_ds, X_valid_ds, y_valid_ds, X_valid, y_valid) = pickle.load(fp)
#        features, f1s, best_model, best_estimators, p_valid_ds, cfmt, classes, report, meta_data, (X_train, y_train, X_valid, y_valid) = pickle.load(fp)
#        f1s, best_model, best_estimators, p_valid_ds, cfmt, classes, report, meta_data, (X_train_ds, y_train_ds, X_valid_ds, y_valid_ds, X_valid) = pickle.load(fp)
        
        
#==============================================================================
# Build Model
#==============================================================================
model = xgb.XGBClassifier(**best_estimators[best_model].get_params())
model.fit(X_train_ds, y_train_ds)
p_valid = model.predict(X_valid)
p_valid = pd.Series(p_valid)
p_valid.index = X_valid.index
p_valid_copy = copy.deepcopy(p_valid)
#model.predict_proba(X_valid)
#
#model = xgb.XGBClassifier(**best_estimators[best_model].get_params())
#model.fit(X_train, y_train)
#p_valid = model.predict(X_valid)
#p_valid = pd.Series(p_valid)
#p_valid.index = X_valid.index
#p_valid_copy = copy.deepcopy(p_valid)
#model.predict_proba(X_valid)

# Predictions
p_valid = pd.DataFrame(p_valid).astype(int)
p_valid['Local_Filepath'] = list(p_valid.index.values)
p_valid.columns = ['Prediction', 'Local_Filepath']
p_valid.loc[:, 'key'] = map(lambda x: ('/').join(x.split('/')[-2:]), p_valid.index)
predictions = p_valid

# Pixel counts
meta_data.loc[:,'key'] = map(lambda x: ('/').join(x.split('/')[-2:]), meta_data.index)

# Load polygon reference dataframe
polygon_df = pd.read_csv(dataframe_local_path)

# Merge
polygon_df.loc[:, 'key'] = map(lambda x: ('/').join(x.split('/')[-2:]), polygon_df['Filepath'])
results_breakdown_df = polygon_df.join(predictions.set_index('key'), on='key')
results_breakdown_df = results_breakdown_df.join(meta_data.set_index('key'), on='key')
results_breakdown_df.loc[:, 'Polygon_Area_in_Ha'] = results_breakdown_df['Polygon_Pixel_Count'] / 200.0
#results_breakdown_df['Polygon_Area_in_Ha'].describe()
polygon_area_bins = [0, 0.5, 1.43, 3.64, 9.39, 30, 10000]
polygon_area_class = ['VLow: less than 0.5 Ha', 'Med: Q1', 'Med: Q2', 'Med: Q3', 'Med: Q4', 'VHigh: more than 30 Ha']
results_breakdown_df.loc[:, 'Polygon_Area_Class'] = pd.cut(results_breakdown_df['Polygon_Area_in_Ha'], polygon_area_bins, labels = polygon_area_class)
results_breakdown_df = results_breakdown_df[['Polygon_id', 'Filepath', 'Local_Filepath', 'Datetime', 'Shapefile_Name', 'Polygon_Area_in_Ha', 'Polygon_Area_Class', 'Polygon_Pixel_Count', 'Empty_Polygon_Flag', 'CULTIVATN', 'PRIHABITAT', 'PRILANDUSE', 'PRIPCTAREA', 'PRISPECIES', 'PRI_PLYEAR', 'SECHABITAT', 'SECLANDUSE', 'SECPCTAREA', 'SECSPECIES', 'SEC_PLYEAR', 'Tree_Age', 'Young_Tree_Flag', 'Woodland_Flag', 'Train_Flag', 'woodland_category', 'Woodland_Class', 'Prediction']]
results_breakdown_df = results_breakdown_df[pd.notnull(results_breakdown_df['Prediction'])]
results_breakdown_df['Y_vs_P'] = [row[1].values*1 for row in results_breakdown_df[['Young_Tree_Flag', 'Prediction']].iterrows()]
results_breakdown_df[['Woodland_Class']] = results_breakdown_df[['Woodland_Class']].fillna('Grassland')


#==============================================================================
# Get Statistics
#==============================================================================
def collect_results(df, col_name = 'CULTIVATN'):
    set_title('Model Statistics')
    support = df[col_name].value_counts()
    cm = df.groupby(col_name).Y_vs_P.agg(lambda x: list(confusion_matrix(np.array(list(x))[:,0].astype(float) , np.array(list(x))[:,1].astype(float))))
    clf_score = df.groupby(col_name).Y_vs_P.agg(lambda x: sum(np.array(list(x))[:,0].astype(int) == np.array(list(x))[:,1].astype(int))/float(len(x)))
    # Temporary average f1 score!
    f1 = df.groupby(col_name).Y_vs_P.agg(lambda x: float(confusion_matrix(np.array(list(x))[:,0].astype(float) , np.array(list(x))[:,1].astype(float))[1,1]) / (2 * confusion_matrix(np.array(list(x))[:,0].astype(float) , np.array(list(x))[:,1].astype(float))[1,1] + confusion_matrix(np.array(list(x))[:,0].astype(float) , np.array(list(x))[:,1].astype(float))[0,1] + confusion_matrix(np.array(list(x))[:,0].astype(float) , np.array(list(x))[:,1].astype(float))[1,0]) + float(confusion_matrix(np.array(list(x))[:,0].astype(float) , np.array(list(x))[:,1].astype(float))[0,0]) / (2 * confusion_matrix(np.array(list(x))[:,0].astype(float) , np.array(list(x))[:,1].astype(float))[0,0] + confusion_matrix(np.array(list(x))[:,0].astype(float) , np.array(list(x))[:,1].astype(float))[0,1] + confusion_matrix(np.array(list(x))[:,0].astype(float) , np.array(list(x))[:,1].astype(float))[1,0]) )
    
    #clf_score = df.groupby(col_name).Y_vs_P.agg(lambda x: list(sum(x)))
    #clf_score = df.groupby(col_name).Y_vs_P.agg(lambda x: list(sum(x)))
    #clf_score = df.groupby(col_name).Y_vs_P.agg(lambda x: list(sum(x)))
    
    results = pd.concat([support, clf_score, f1, cm], axis =1)
    results.columns = ['Support', 'CLF Score', 'Avg F1 Score', 'Conf Matrix']
    results.index.name = col_name
    print results
    return results

def set_title(string):
    # Check if string is too long
    string_size = len(string)
    max_length = 57
    if string_size > max_length:
        print 'TITLE TOO LONG'
    else:
        lr_buffer_len = int((max_length - string_size) / 2)
        full_buffer_len = lr_buffer_len * 2 + string_size
        print '\n'
        print full_buffer_len * '='
        print full_buffer_len * ' '
        print lr_buffer_len * ' ' + string + lr_buffer_len * ' '
        print full_buffer_len * ' '
        print full_buffer_len * '=' + '\n\n'


def plot_samples(results_breakdown_df, pred_classes = ['TP', 'TN', 'FP', 'FN'], number_of_samples = 1, decibel = True, random_seed = 1):
    random.seed(random_seed)
    for pred_class in pred_classes:
        print '\n==============================================='
        print '\tSample Validation Images:\t', pred_class
        print '===============================================\n'
        if pred_class == 'TP':
            index = [1,1]
        elif pred_class == 'TN':
            index = [0,0]
        elif pred_class == 'FP':
            index = [0,1]
        elif pred_class == 'FN':
            index = [1,0]
        else:
            print 'PRED_CLASSES value must be inside: TP, TN, FP, FN'
        full_paths = results_breakdown_df[(results_breakdown_df['Young_Tree_Flag'] == index[0]) & (results_breakdown_df['Prediction'] == index[1])]['Local_Filepath']
        paths = random.sample(full_paths, number_of_samples)
        
        images = []
        for path in paths:
            if decibel:
                image = 10 * np.log10(rasterio.open(path).read())
            else:
                image = rasterio.open(path).read()
            fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (10,5))
            ax1.set_title('Band 0')
            im1 = ax1.imshow(image[0])
            fig.colorbar(im1, ax = ax1)
            ax2.set_title('Band 1')
            im2 = ax2.imshow(image[1])
            fig.colorbar(im2, ax = ax2)
            plt.show()
            
            images.append(image)
    return images

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
# Display results
set_title('Balanced Performance')
plt.figure(figsize=(7,7))
plot_confusion_matrix(cfmt, ['Non-Young Trees', 'Young Trees'])
print report
print best_model
print best_estimators

set_title('Overall Performance')
plt.figure(figsize=(7,7))
plot_confusion_matrix(confusion_matrix(y_valid, p_valid_copy), ['Non-Young Trees', 'Young Trees'])
print classification_report(y_valid, p_valid_copy)
print best_model
print best_estimators

pd.options.display.max_rows = 10
pd.options.display.max_columns = 999
_ = collect_results(results_breakdown_df, col_name = 'CULTIVATN')
_ = collect_results(results_breakdown_df, col_name = 'Polygon_Area_Class')
_ = collect_results(results_breakdown_df, col_name = 'woodland_category')
_ = collect_results(results_breakdown_df, col_name = 'Woodland_Class')
_ = collect_results(results_breakdown_df, col_name = 'Woodland_Flag')
#_ = collect_results(results_breakdown_df, col_name = 'Tree_Age')
_ = collect_results(results_breakdown_df, col_name = 'PRILANDUSE')
_ = collect_results(results_breakdown_df, col_name = 'PRISPECIES')
# To be added: bucket of sizes, scale, vegetation?

# Display example TP, TN, FP, FN results
_ = plot_samples(results_breakdown_df, pred_classes = ['TP', 'TN', 'FP', 'FN'], number_of_samples = 3, decibel = True, random_seed = 1)
images = plot_samples(results_breakdown_df, pred_classes = ['FP'], number_of_samples = 10, decibel = True, random_seed = 3)
