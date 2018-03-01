import itertools 
import numpy as np
import datetime
import dill
import cPickle as pickle
import skimage as sk
import glob
import sys
import os
import pandas as pd
import rasterio
from skimage import morphology
from CLF_9helper_build_feature_functions import *

from pyspark import SparkConf, SparkContext

dill.settings['recurse'] = True

WORK_DIR = "/home/ubuntu/Documents/Model_Build/"

def get_date(datetime_str):
    return datetime.datetime.strptime(datetime_str, '%Y%m%dT%H%M%S')

def apply_fun_to_image_path(image_path, function):
    datetime_str = image_path.split('/')[-1].split('_')[4]
    date = get_date(datetime_str)    
    src = rasterio.open(image_path).read()
    with np.errstate(divide='ignore'):
        src = np.where(src != 0, 10 * np.log10(src), 0)
    return function(src)

def get_date(datetime_str):
    return datetime.datetime.strptime(datetime_str, '%Y%m%dT%H%M%S')

def path2datetime(path):
    return get_date(path.split('/')[-1].split('_')[4])

def add_weather(feature_df, weather_dataframe_name):
    weather_df = pd.read_csv(weather_dataframe_name)
    feature_df['Datetime'] = feature_df['Local_Filepath'].apply(path2datetime)
    feature_df.loc[:,'Datetime'] = feature_df['Datetime'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d %H:00:00'))
    merged_df = feature_df.merge(weather_df, on='Datetime', how='left')
    return merged_df.drop(['Datetime','PrecipType'], axis=1)

def add_local_path(df, local_data_dir):
    S3_DIR = 's3://jncc-poc5/East_Anglia_PP5_DataFrame/'
    df.loc[:, 'Local_Filepath'] = df['Filepath'].apply(lambda x: os.path.join(local_data_dir, x.replace(S3_DIR, '')))

def import_feature_functions(feature_dir):
    features = []
    for path in glob.glob(os.path.join(feature_dir, '*')):
        with open(path, 'r') as fp:
            features.append(dill.load(fp))
    return features

def apply_features_old(sc, df, features):
    paths = df['Local_Filepath'].unique()
    path_rdd = sc.parallelize([[path] for path in paths])
    #feature_rdd = path_rdd
    feature_rdds = []
    for idx, feature in enumerate(features):
        if idx == 0:
            # feature_rdd = feature_rdd.map(lambda x: [x[0]] + list(apply_fun_to_image_path(x[0], features[idx].transform)))
            feature_rdds.append(path_rdd.map(lambda x: (x[0], list(apply_fun_to_image_path(x[0], feature.transform)))))

            path_rdd.map(lambda x: [x[0]] + list(apply_fun_to_image_path(x[0], features[0].transform))).\
                     map(lambda x: x + list(apply_fun_to_image_path(x[0], features[1].transform)))
        else:
            #feature_rdd = feature_rdd.map(lambda x: x + list(apply_fun_to_image_path(x[0], features[idx].transform)))
            feature_rdd = feature_rdd.join(feature_rdd.map(lambda x: [x[0]] + list(apply_fun_to_image_path(x[0], features[idx].transform))))

def apply_features(sc, df, features):
    paths = df['Local_Filepath'].unique()
    path_rdd = sc.parallelize([[path] for path in paths])
    features = sorted(features, key = lambda x: x.name)
    #feature_rdd = path_rdd
    feature_rdds = []
    for idx, feature in enumerate(features):
        print("{}: Applying feature {}".format(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), feature.name))
        key_arr = path_rdd.map(lambda x: (x[0], list(apply_fun_to_image_path(x[0], feature.transform)))).collect()        
        feature_rdds.append(key_arr)

    feature_dict = {}
    for idx, arr in enumerate(feature_rdds):
        if idx == 0:
            feature_dict = {pair[0]: pair[1]  for pair in arr}
        else:
            for pair in arr:
                feature_dict[pair[0]] += pair[1]
    feature_columns = list(itertools.chain(*[[feat.name.replace(' ','')+'_0', feat.name.replace(' ','')+'_1'] for feat in features]))
    feature_df = pd.DataFrame.from_dict(feature_dict, orient='index').reset_index()
    feature_df.columns = ['Local_Filepath'] + feature_columns
    return feature_df

def build_input_data(df, target_col):
    X = np.hstack(df.drop(['Local_Filepath', 'Train_Flag', target_col], axis=1).values)
    y = df[target_col].values
    return X, y

def import_features_and_build_dataframe(sc, dataframe_name, feature_function_dir, local_data_dir, meta_data_columns,
                                        target_columns, experiment_name, weather_dataframe_name, save_df=True):
    if isinstance(feature_function_dir, str):
        features = import_feature_functions(feature_function_dir)
    else:
        features = feature_function_dir
    df = pd.read_csv(dataframe_name)
    add_local_path(df, local_data_dir)
    get_polygon_meta_data(sc, df)
    feature_df = apply_features(sc, df, features)
    feature_df = add_weather(feature_df, weather_dataframe_name)
    feature_df = feature_df.merge(df[meta_data_columns + target_columns], on='Local_Filepath', how='inner')
    #this does not seem to workTW save_data = (feature_df, features, target_columns, meta_data_columns)
    save_data = (feature_df, features)
    print('Finished processing; saving the feature dataframe now')
    if save_df:
        with open(os.path.join(WORK_DIR, 'Experiments', experiment_name+'.pkl'), 'w') as fp:
            dill.dump(save_data, fp)
    return feature_df

def get_polygon_meta_data(sc, df):
    paths = df['Local_Filepath'].unique()
    path_rdd = sc.parallelize([[path] for path in paths])
    src_rdd = path_rdd.map(lambda x: read_image(x[0]))
    Empty_Polygon_Flag = src_rdd.map(check_empty_polygon).collect()
    Polygon_Pixel_Count = src_rdd.map(get_polygon_pixel_counts).collect()
    df.loc[:, 'Empty_Polygon_Flag'] = list(Empty_Polygon_Flag)
    df.loc[:, 'Polygon_Pixel_Count'] = list(Polygon_Pixel_Count)

def read_image(path):
    with rasterio.open(path) as data:
        return data.read()

def check_empty_polygon(sar_data):
    # Check if there is at least 1 zero value
    if np.shape(np.where(sar_data == 0))[1] > 0:
        return 1
    # Check if the whole polygon is empty
    elif np.shape(np.where(~np.isnan(sar_data)))[1] == 0:
        return 2
    # Check if the polygon is less than half a hectare
    elif np.shape(np.where(~np.isnan(sar_data)))[1] < 2 * 50:
        return 3
    else:
        return 0

def get_polygon_pixel_counts(sar_data):
    return np.shape(np.where(~np.isnan(sar_data)))[1]

if __name__ == "__main__":
    experiment_setup = {'sc': SparkContext(conf = (SparkConf().set('spark.driver.maxResultSize','2g'))),
                        'dataframe_name': os.path.join(WORK_DIR, 'Polygon_Reference_DataFrame/Filtered_Polygon_DataFrame.csv'),
                        'feature_function_dir': os.path.join(WORK_DIR, 'Features'),
                        'meta_data_columns': ['Polygon_id', 'Local_Filepath', 'Train_Flag', 'Empty_Polygon_Flag', 'Polygon_Pixel_Count'],
                        'target_columns': ['Young_Tree_Flag', 'Tree_Age', 'Woodland_Class'],
                        'local_data_dir': os.path.join(WORK_DIR, 'Polygon_Data'),
                        'experiment_name': 'lbp_entr_dop10',
                        'weather_dataframe_name': '/home/ubuntu/Documents/Weather/east_anglia_weather.csv'
    }
    experiment_setup['sc'].setLogLevel("ERROR")
    import_features_and_build_dataframe(**experiment_setup)


#    aws s3 cp /home/ubuntu/Documents/East_Anglia_Model_Build/Experiments/YT_VS_NWL_redefined_grassland_TEST_temporal_5_10_season_rainfall.pkl s3://jncc-poc5/Feature_Dataframes/YT_VS_NWL_redefined_grassland_TEST_temporal_5_10_season_rainfall.pkl
