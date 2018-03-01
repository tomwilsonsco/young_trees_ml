
# coding: utf-8

# # CLF_8a_Filter_Dataset

# ### Import Modules

# In[1]:

import sys
import rasterio
import pandas as pd
import fiona
import numpy as np
import pyproj
import time
import matplotlib.pyplot as plt
import os
import shutil
import subprocess
import datetime
import multiprocessing
import warnings
from joblib import Parallel, delayed

from os import listdir
from PIL import Image
from scipy.misc import imresize
from rasterio.tools.mask import mask
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from shapely.geometry import Polygon, MultiPolygon
from pyspark import SparkContext, SQLContext, SparkConf


# # Define Utility Functions

# In[2]:

# Map functions to download s3 data in // using boto
def map_download_s3_data_boto(bucket_, polygon_s3_full_path_, polygon_local_dir_ = '/home/ubuntu/Documents/Model_Build/Polygon_Data/'):    
    # Define paths
    polygon_local_full_path = polygon_local_dir_ + '/' + '/'.join(polygon_s3_full_path_.split('/')[-2:])

    # Create folder
    if not os.path.exists(polygon_local_dir_ + '/' + '/'.join(polygon_s3_full_path_.split('/')[-2])):
        try: 
            os.makedirs(polygon_local_dir_ + polygon_s3_full_path_.split('/')[-2])
        except OSError:
            pass

    # Download data
    key = bucket_.get_key(('/').join(polygon_s3_full_path_.split('/')[3:]))
    key.get_contents_to_filename(polygon_local_full_path)


# # Main

# In[7]:
test = 'test'
dataframe_local_path = '/home/ubuntu/Documents/Model_Build/Polygon_Reference_DataFrame/Polygon_DataFrame_Artificial_Var.csv' #sys.argv[1]
filtered_df_save_path = '/home/ubuntu/Documents/Model_Build/Polygon_Reference_DataFrame/Filtered_Polygon_DataFrame.csv' #sys.argv[2]
polygon_local_dir = '/home/ubuntu/Documents/Model_Build/Polygon_Data/'#sys.argv[3]
if __name__ == '__main__':
    # Load dataframe
    print 'Loading polygon dataframe ...',
    polygon_df = pd.read_csv(dataframe_local_path)
    print 'Done'
    
    # Reorder dataframe
#    polygon_df = polygon_df[['Polygon_id','Filepath','Datetime','woodland_category', 
#                            'tree_age', 'tree_age_class', 'AreaHa','BLOCK','COMPTMENT',
#                            'CULTIVATN','CULT_CODE','CaseRef','CaseType','Category',
#                            'CurrStat','DateApp','DateApprv','Descriptr','FLA','FOREST',
#                            'FSArea','FWP','Fell_ref','IFT_IOA','OBJECTID','PRIHABCODE',
#                            'PRIHABITAT','PRILANDUSE','PRIPCTAREA','PRISPECIES',
#                            'PRI_LUCODE','PRI_PLYEAR','PRI_SPCODE','PRI_YIELD','PropName',
#                            'SECHABCODE','SECHABITAT','SECLANDUSE','SECPCTAREA',
#                            'SECSPECIES','SEC_LUCODE','SEC_PLYEAR','SEC_SPCODE',
#                            'SEC_YIELD','SUBCOMPT','SUBCOMPTID',
#                            'Shape_Area','Shape_Leng','SubCpt','TERHABCODE','TERHABITAT',
#                            'TERLANDUSE','TERPCTAREA','TERSPECIES','TER_LUCODE',
#                            'TER_PLYEAR','TER_SPCODE','TER_YIELD','WAG','WCG','WIG',
#                            'WMG','WPG','WRG','WorkAreaID']]
    
    # Filter polygon data [CHANGE THIS]
    #########################################################################################################
    date_min = '2016-05-01'
    date_max = '2016-05-10'

    filtered_df = polygon_df[(polygon_df['Filepath'] != 'wrong_format') & (polygon_df['PRIPCTAREA'] >= 90) &                            ((polygon_df['woodland_category'] == 'coniferous_woodlands') | (polygon_df['woodland_category'] == 'grassland')) &                            (date_min <= polygon_df['Datetime']) &                            (polygon_df['Datetime'] < date_max)]

    print 'Filtered date size ratio:\t', str(len(filtered_df)), '/', str(len(polygon_df)), '\t', str(np.round(100*float(len(filtered_df))/ float(len(polygon_df)), 1)), '%'
    #########################################################################################################
    
    # Save dataframe
    print 'Saving filtered dataframe ...',
    filtered_df.to_csv(filtered_df_save_path)
    print 'done'
    
    # Download filtered polygon data
    print 'Downloading polygon data ...',
    if os.path.exists(polygon_local_dir):
        warnings.warn('Path for polygon data already exists polygon data will be merged.')
    aws_key        = 'AKIAIYS5OGSR5K2R5FXQ'
    aws_secret_key = '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'
    aws_connection = S3Connection(aws_key, aws_secret_key)
    bucket         = aws_connection.get_bucket('jncc-poc5')
    file_path_list = list(filtered_df['Filepath'])
    conf = SparkConf().setMaster("local").setAppName("My App")
    sc = SparkContext(conf = conf)
    s3_file_path_RDD = sc.parallelize(file_path_list)
    st = time.time()
    s3_file_path_RDD.map(lambda x: map_download_s3_data_boto(bucket, x, polygon_local_dir)).collect()
    print 'done in\t', np.round(time.time() - st), 'sec'


# In[ ]:

# Example
# unset PYSPARK_DRIVER_PYTHON
# spark-submit CLF_8a_Filter_Dataset.py /home/ubuntu/Documents/Model_Build/Polygon_Reference_DataFrame/Polygon_DataFrame_Artificial_Var.csv /home/ubuntu/Documents/Model_Build/Polygon_Reference_DataFrame/Filtered_Polygon_DataFrame.csv /home/ubuntu/Documents/Model_Build/Polygon_Data/

