#==============================================================================
# Build Models
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

WORK_DIR = "/home/ubuntu/Documents/Model_Build_RAW/"

# Just for running on spyder
import sys
sys.path.append(WORK_DIR)
from CLF_9helper_build_feature_functions import *


#==============================================================================
# Utility Functions
#==============================================================================
def drop_meta_data(df, list_of_cols_to_drop):
    df = df.drop(list_of_cols_to_drop, axis=1)
    df = np.apply_along_axis(lambda x: np.hstack(x), 1, df.values)
    return df

def downsample_df(df, labels_df, random_seed):
    num_of_yt   = sum(labels_df)
    random.seed(random_seed+1)
    downsample_bad_ix   = random.sample(np.where(labels_df == 0)[0], num_of_yt)
    good_ix             = np.where(labels_df == 1)[0]
    downsampled_full_ix = np.append(downsample_bad_ix, good_ix)
    df_ds          = pd.concat([df.iloc[[index]] for index in downsampled_full_ix])
    return df_ds

def build_train_test_data(df, target_col, extracted_target_columns, balanced=False, train_valid_split=0.3, random_state = 1):
    train_ids = df[df['Train_Flag']]['Polygon_id'].unique()
    train_ids, valid_ids = train_test_split(train_ids, test_size = train_valid_split, random_state = random_state)
    
    # Train / valid / test
    train_df = df[df['Polygon_id'].isin(train_ids)]
    valid_df = df[df['Polygon_id'].isin(valid_ids)]
    test_df = df[~df['Train_Flag']]
    
    X_train = drop_meta_data(train_df, ['Polygon_id','Local_Filepath', 'Train_Flag'] + extracted_target_columns)
    X_train = pd.DataFrame(X_train).set_index([train_df['Local_Filepath'].values], drop=True)
    y_train = train_df[[target_col]].astype('int').values.reshape(-1)
    y_train=pd.Series(y_train)
    y_train.index = X_train.index
    
    X_valid = drop_meta_data(valid_df, ['Polygon_id','Local_Filepath', 'Train_Flag'] + extracted_target_columns)
    X_valid = pd.DataFrame(X_valid).set_index([valid_df['Local_Filepath'].values], drop=True)
    y_valid = valid_df[[target_col]].astype('int').values.reshape(-1)
    y_valid = pd.Series(y_valid)
    y_valid.index = X_valid.index
    
    X_test = drop_meta_data(test_df, ['Polygon_id','Local_Filepath', 'Train_Flag'] + extracted_target_columns)
    X_test = pd.DataFrame(X_test).set_index([test_df['Local_Filepath'].values], drop=True)
    y_test = test_df[[target_col]].astype('int').values.reshape(-1)
    y_test = pd.Series(y_test)
    y_test.index = X_test.index
    
    # Downsample
    if balanced:
        train_labels = train_df.loc[df['Train_Flag'], target_col].astype('int').values
        train_df_ds = downsample_df(train_df, train_labels, random_seed)
        valid_labels = valid_df.loc[df['Train_Flag'], target_col].astype('int').values
        valid_df_ds = downsample_df(valid_df, valid_labels, random_seed)
        
        X_train_ds = drop_meta_data(train_df_ds, ['Polygon_id','Local_Filepath', 'Train_Flag'] + extracted_target_columns)
        X_train_ds = pd.DataFrame(X_train_ds).set_index([train_df_ds['Local_Filepath'].values], drop=True)
        y_train_ds = train_df_ds[[target_col]].astype('int').values.reshape(-1)
        y_train_ds = pd.Series(y_train_ds)
        y_train_ds.index = X_train_ds.index
    
        X_valid_ds = drop_meta_data(valid_df_ds, ['Polygon_id','Local_Filepath', 'Train_Flag'] + extracted_target_columns)
        X_valid_ds = pd.DataFrame(X_valid_ds).set_index([valid_df_ds['Local_Filepath'].values], drop=True)
        y_valid_ds = valid_df_ds[[target_col]].astype('int').values.reshape(-1)
        y_valid_ds = pd.Series(y_valid_ds)
        y_valid_ds.index = X_valid_ds.index
        return X_train, y_train, X_valid, y_valid, X_test, y_test, X_train_ds, y_train_ds, X_valid_ds, y_valid_ds
    
    else:
        return X_train, y_train, X_valid, y_valid, X_test, y_test

def get_polygon_pixel_count(df, column_name = 'Polygon_Pixel_Count'):
    pixel_counts = df[[column_name]].set_index([df['Local_Filepath'].values], drop=True)
    return pixel_counts
#def remove_rows_all_nan(X_data, y_data):
#    X_data_no_nan_rows = np.array([xrow for xrow, yrow in zip(X_data, y_data) if np.sum(np.isnan(xrow)) < len(xrow)])
#    y_data_no_nan_rows = np.array([yrow for xrow, yrow in zip(X_data, y_data) if np.sum(np.isnan(xrow)) < len(xrow)])
#    return X_data_no_nan_rows, y_data_no_nan_rows

#def split_and_impute_train_test(feature_df, target_column, random_state_):
#    X_train, y_train, X_test, y_test, _, _ = build_train_test_data(feature_df, target_column, train_valid_split=0.3, balanced=True, random_state = random_state_)
#    X_train, y_train = remove_rows_all_nan(X_train, y_train)
#    imputer = Imputer(strategy='mean',axis=0)
#    X_train = imputer.fit_transform(X_train)
#    X_test = imputer.transform(X_test)
#    
#    return X_train, y_train, X_test, y_test

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

def gridsearch(X_train, y_train, X_test, y_test, models, params, nfoldCV = 4, score = 'f1'):
        
    classes, y_train_binary     = np.unique(y_train, return_inverse = True)
    _, y_test_binary            = np.unique(y_test, return_inverse = True)

    # Allocate variables
    y_best_preds                = {}
    best_estimators             = {}
    
    print '\n============================================='
    print 'Tuning hyper-parameters ( ranking: %s' % score, ')'
    print '============================================='    
    for model_name in params.keys():
        print '\n\n--------------------------------'
        print 'Models: %s' % model_name
        print '--------------------------------'
        # Define model    
        clf = GridSearchCV(models[model_name], 
                           params[model_name], 
                           cv=nfoldCV,
                           scoring=score,
                           n_jobs = -1,
                           verbose=1)
        print '\n',
        
        # Model train
        clf.fit(X_train, y_train_binary)
        #clf.fit(X_train, y_train)
        print '.',
        
        # Predict on test
        #y_best_pred                 = clf.predict_proba(X_test)
        y_best_pred                 = clf.predict(X_test)
        y_best_preds[model_name]    = y_best_pred
        print '. .  Done'
        
        # Append best estimators
        best_estimators[model_name] = clf.best_estimator_
    
    # Display results
    print '\n============================================='
    print 'Best Hyper-parameters ( ranking: %s' % score, ')'
    print '============================================='  
    
    # Score
    print '\n\n------------------------------'
    print 'Score: %s' % score
    print '------------------------------'
    #    for model_name in params.keys():
    #        print '%20s: \t' % model_name, np.round(roc(y_test_binary, y_best_preds[model_name][:,1]), 4)
        
    # ROC curves
    plt.figure(figsize  = (6,5))
    #y_best_preds_list   = [ v[:,1] for v in y_best_preds.values()]
    #y_best_preds_list   = [ v for v in y_best_preds.values()]
    
    # Best classifier
    #roc_aucs            = []
    f1s                 = []
    for model_name in params.keys():
        #roc_auc_ = roc(y_test_binary, y_best_preds[model_name][:,1])
    #    #_, y_best_pred_binary = np.unique(y_best_preds[model_name][:,1], return_inverse = True)
    #    #roc_auc_ = roc(y_test_binary, y_best_pred_binary)
        f1                  = f1_score(y_test_binary, y_best_preds[model_name], average = 'weighted')
        #roc_aucs.append(roc_auc_)
        f1s.append(f1)
    
    #best_clf_index = np.where(roc_aucs == max(roc_aucs))[0][0]
    best_clf_index      = np.where(f1s == max(f1s))[0][0]
    best_model          = params.keys()[best_clf_index]
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
    print 'Best Model: %20s' % best_model
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
    
    # Confusion matrix
    #cfmt   = confusion_matrix(y_test, [classes[i] for i in y_best_preds[best_model][:,1]])
    cfmt                = confusion_matrix(y_test_binary, y_best_preds[best_model])
    plt.figure(figsize=(6,6))
    plot_confusion_matrix(cfmt, classes)
    plt.show()
    #print(classification_report(y_test, [classes[i] for i in y_best_preds[best_model][:,1]]))
    report              = classification_report(y_test_binary, y_best_preds[best_model])
    print '\n', classification_report(y_test_binary, y_best_preds[best_model])
    
    # Best estimators
    print '\n\n------------------------------'
    print 'Best Estimator'
    print '------------------------------'
    print '\n%20s: \t' % best_model, '\n\n', best_estimators[best_model]
    
    # Plot sample results
    #p_test                  = [classes[i] for i in y_best_preds[best_model][:,1]]
    p_test                  = [classes[i] for i in y_best_preds[best_model]]
    return f1s, best_model, best_estimators, p_test, cfmt, classes, report


#==============================================================================
# Main
#==============================================================================
if __name__ == '__main__':
    # Settings
    experiment_name = 'lbp_entr_dop10_temporal'
    save_results_name = 'clf_lbp_entr_dop10_weather_temporal'
    #features_columns = ['Entropy_0','Entropy_1','LocalBinaryPattern_0','LocalBinaryPattern_1','PixelDistribution_0','PixelDistribution_1', 'TemporalVariation_0', 'TemporalVariation_1', 'TemporalMedian_0', 'TemporalMedian_1']
    features_columns = ['Entropy_0','Entropy_1','LocalBinaryPattern_0','LocalBinaryPattern_1','PixelDistribution_0','PixelDistribution_1', 'TemporalVariation_0', 'TemporalVariation_1', 'TemporalMedian_0', 'TemporalMedian_1', 'Humidity', 'Pressure', 'Temperature', 'WindSpeed']
    keys = ['Local_Filepath', 'Polygon_id','Train_Flag']
    #features = ['HaralickFeatures_0']
    target_column = 'Young_Tree_Flag' #'Woodland_Class'
    random_seed = 1
    
    # Load data
    with open(os.path.join(WORK_DIR, 'Experiments', 'lbp_entr_dop10_temporal.pkl'), 'r') as fp:
        (feature_df, features, extracted_target_columns, meta_data_columns) = pickle.load(fp)
    # feature_df.columns
    # len(feature_df)
    
    # Get meta data
    meta_data = get_polygon_pixel_count(feature_df)
    meta_data.loc[:, 'Empty_Polygon_Flag'] = list(feature_df['Empty_Polygon_Flag'])
    feature_df = feature_df[feature_df['Empty_Polygon_Flag']==0].drop('Empty_Polygon_Flag',axis=1)
    feature_df.drop('Polygon_Pixel_Count', axis = 1)
    
    # Woodland class
    feature_df[['Woodland_Class']] = feature_df[['Woodland_Class']].fillna('Grassland')
    feature_df[['Woodland_Class']]
    classes, feature_df[['Woodland_Class']] = np.unique(feature_df['Woodland_Class'], return_inverse = True)
    
    # Compile dataset
    feature_df = feature_df[features_columns + keys + extracted_target_columns]

    # Model Params
    models = {'Random Forest': RandomForestClassifier(),
            'Logistic Regression': LogisticRegression(),
            'Support Vector': svm.SVC(),
            'XGBoost': xgb.XGBClassifier(objective='binary:logistic')}
            #'Ensemble': VotingClassifier(estimators=[('lr', )], voting='hard')}
    params = {
            'XGBoost':
                {'learning_rate': [ 0.3],
                 'max_depth': [8],
                 'min_child_weight': [1],
                 'n_estimators' : [100],
                 'seed': [random_seed]
                    }
            }
#            'Random Forest': 
#                {'n_estimators': [100],
#                'criterion': ['gini'],
#                'max_depth': [None],
#                'min_samples_split': [2,3,4],
#                'min_samples_leaf': [1,2,3],
#                'min_weight_fraction_leaf': [0.0],
#                'max_features': ['auto'],
#                'max_leaf_nodes': [None],
#                'bootstrap': [True],
#                'oob_score': [False],
#                'verbose': [0],
#                'warm_start': [False],
#                'class_weight': [None, 'balanced'],
#                'random_state': [random_seed]
#                    }
#            }
#            'Logistic Regression': 
#                {'penalty': ['l2'],
#                'dual': [False, True],
#                'tol': [0.0001, 0.001],
#                'C': [1.0],
#                'fit_intercept': [True],
#                'intercept_scaling': [1],
#                'class_weight': [None, 'balanced'],
#                'solver': ['liblinear'],
#                'max_iter': [100, 200],
#                'multi_class': ['ovr'],
#                'verbose': [0],
#                'warm_start': [False, True],
#                'random_state': [random_seed]
#                    }
#            }
#            'Support Vector': 
#                [
#                    {'kernel': ['rbf'],
#                     'gamma': [1e-3, 1e-4],
#                     'C': [1e0, 1e1, 1e2],
#                     'probability': [True],
#                     'random_state': [random_seed] 
#                        },
#                    {'kernel': ['linear'], 
#                     'C': [1e0, 1e1, 1e2],
#                     'probability': [True],
#                     'random_state': [random_seed]
#                        },
#                    {'kernel': ['poly'],
#                     'gamma': [1e0, 1e1, 2e1, 3e1, 1e2],
#                     'probability': [True],
#                     'random_state': [random_seed]
#                        }
#                     ]
#             }

    X_train, y_train, X_valid, y_valid, X_test, y_test, X_train_ds, y_train_ds, X_valid_ds, y_valid_ds = build_train_test_data(feature_df, target_column, extracted_target_columns, balanced=True, train_valid_split=0.3, random_state = random_seed)

    print '\n\n=============================================================='
    print 'Training:Test Size:\t {} : {}'.format(len(X_train_ds), len(X_valid_ds))
    print 'Number of features:\t {}'.format(np.shape(X_train_ds)[1])
    print '==============================================================\n\n'

    f1s, best_model, best_estimators, p_valid_ds, cfmt, classes, report = gridsearch(X_train_ds, y_train_ds, X_valid_ds, y_valid_ds, models, params, nfoldCV = 4, score = 'f1_weighted')
#==============================================================================
#     X_train, y_train, X_valid, y_valid, X_test, y_test = build_train_test_data(feature_df, target_column, extracted_target_columns, balanced=False, train_valid_split=0.3, random_state = random_seed)
# 
#     print '\n\n=========================================================='
#     print 'Training/Test Size: {}, {}'.format(len(X_train), len(X_valid))
#     print '==========================================================\n\n'
# 
#     f1s, best_model, best_estimators, p_valid_ds, cfmt, _, report = gridsearch(X_train, y_train, X_valid, y_valid, models, params, nfoldCV = 4, score = 'f1_weighted')
#==============================================================================

#
#
#
#    x = pd.DataFrame(X_train_ds).set_index([train_df_ds['Local_Filepath'].values], drop=True)
#    y=pd.Series(y_train_ds)
#    y.index = x.index
#    x2 = pd.DataFrame(X_valid_ds).set_index([valid_df_ds['Local_Filepath'].values], drop=True)
#    y2=pd.Series(y_valid_ds)
#    y2.index = x2.index
#    x = x.iloc[:1000, :10].append( x.iloc[-100:, :10])
#    x2 = x2.iloc[:1000, :10].append( x2.iloc[-100:, :10])
#    y = y.iloc[:1000].append( y.iloc[-100:])
#    y2 = y2.iloc[:1000].append( y2.iloc[-100:])
#    
#    a, b, c, d, e, f, g = gridsearch(x, y, x2, y2, models, params, nfoldCV = 4, score = 'f1_weighted')

    # Display results
    plt.figure(figsize=(7,7))
    plot_confusion_matrix(cfmt, classes)
    print report
    print best_model
    print best_estimators
    
    # Feature importance
    model = xgb.XGBClassifier(**best_estimators[best_model].get_params())
    model.fit(X_train_ds, y_train_ds)
    plt.figure(figsize=(10,10))
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.xlim(0, len(model.feature_importances_))
    plt.show()
    
    # Save results
    save_data = features, f1s, best_model, best_estimators, p_valid_ds, cfmt, classes, report, meta_data, (X_train_ds, y_train_ds, X_valid_ds, y_valid_ds, X_valid, y_valid)
    save_filename = os.path.join(WORK_DIR, 'Models', save_results_name + '.pkl')
    with open(save_filename, 'w') as fp:
            pickle.dump(save_data, fp)
            
            

#==============================================================================
#     # Save results
#     save_data = features, f1s, best_model, best_estimators, p_valid_ds, cfmt, classes, report, meta_data, (X_train, y_train, X_valid, y_valid)
#     save_filename = os.path.join(WORK_DIR, 'Models', save_results_name + '.pkl')
#     with open(save_filename, 'w') as fp:
#             pickle.dump(save_data, fp)
#==============================================================================
