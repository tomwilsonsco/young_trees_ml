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

WORK_DIR = "/home/ubuntu/Documents/East_Anglia_Model_Build/"

# Just for running on spyder
import sys
sys.path.append(WORK_DIR)


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
#        return X_train, y_train, X_valid, y_valid, X_test, y_test, X_train_ds, y_train_ds, X_valid_ds, y_valid_ds
        return X_train, y_train, X_valid, y_valid, X_train_ds, y_train_ds, X_valid_ds, y_valid_ds
    
    else:
        return X_train, y_train, X_valid, y_valid, X_test, y_test

def get_polygon_pixel_count(df, column_name = 'Polygon_Pixel_Count'):
    pixel_counts = df[[column_name]].set_index([df['Local_Filepath'].values], drop=True)
    return pixel_counts


#==============================================================================
# Main
#==============================================================================
if __name__ == '__main__':
    # Settings
    experiment_name = 'YT_VS_NWL_redefined_grassland_TEST_temporal_5_10_season_rainfall'
    #features_columns = ['Entropy_0','Entropy_1','LocalBinaryPattern_0','LocalBinaryPattern_1','PixelDistribution_0','PixelDistribution_1', 'TemporalVariation_0', 'TemporalVariation_1', 'TemporalMedian_0', 'TemporalMedian_1']
    features_columns =['Entropy_0','Entropy_1','LocalBinaryPattern_0','LocalBinaryPattern_1','PixelDistribution_0','PixelDistribution_1', 'TemporalVariation_0', 'TemporalVariation_1', 'TemporalMedian_0', 'TemporalMedian_1', 'Humidity', 'Pressure', 'Temperature', 'WindSpeed', 'TemporalVariation10_0', 'TemporalVariation10_1',
       'TemporalMedian10_0', 'TemporalMedian10_1', 'Temp_Max', 'Temp_Min',
       'Air_Frost_Days', 'Rainfall']
    keys = ['Local_Filepath', 'Polygon_id','Train_Flag']
    #features = ['HaralickFeatures_0']
    target_column = 'Young_Tree_Flag' #'Woodland_Class'
    random_seed = 1
    
    # Load data
    with open(os.path.join(WORK_DIR, 'Experiments', experiment_name + '.pkl'), 'r') as fp:
        (feature_df, features, extracted_target_columns, meta_data_columns) = pickle.load(fp)
    feature_df.columns
    len(feature_df)
    quit()
    # Get meta data
    meta_data = get_polygon_pixel_count(feature_df)
    meta_data.loc[:, 'Empty_Polygon_Flag'] = list(feature_df['Empty_Polygon_Flag'])
    feature_df = feature_df[feature_df['Empty_Polygon_Flag']==0].drop('Empty_Polygon_Flag',axis=1)
    feature_df = feature_df.drop('Polygon_Pixel_Count', axis = 1)
    
    # Woodland class
    feature_df[['Woodland_Class']] = feature_df[['Woodland_Class']].fillna('Grassland')
    feature_df[['Woodland_Class']]
    classes, feature_df[['Woodland_Class']] = np.unique(feature_df['Woodland_Class'], return_inverse = True)
    
    # Compile dataset
    feature_df = feature_df[features_columns + keys + extracted_target_columns]

    # Model Params
    params = {
            'XGBoost':
                {'learning_rate': [ 0.3],
                 'max_depth': [8],
                 'min_child_weight': [1],
                 'n_estimators' : [100],
                 'seed': [random_seed]
                    }
            }

#    X_train, y_train, X_valid, y_valid, X_test, y_test, X_train_ds, y_train_ds, X_valid_ds, y_valid_ds = build_train_test_data(feature_df, target_column, extracted_target_columns, balanced=True, train_valid_split=0.3, random_state = random_seed)
    X_train, y_train, X_valid, y_valid, X_train_ds, y_train_ds, X_valid_ds, y_valid_ds = build_train_test_data(feature_df, target_column, extracted_target_columns, balanced=True, train_valid_split=0.3, random_state = random_seed)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_ds, y_train_ds)
    p_valid = model.predict(X_valid_ds)
    print confusion_matrix(y_valid_ds, p_valid)
    print classification_report(y_valid_ds, p_valid)