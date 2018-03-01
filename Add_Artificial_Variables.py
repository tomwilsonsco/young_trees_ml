import os
import numpy as np
import datetime
import rasterio
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def get_tree_age(df):
    df.loc[:, 'Datetime'] = [datetime.datetime.strptime(d, '%Y%m%dT%H%M%S') for d in df['Date']]
    def _get_age(row):
        plt_year = row['PRI_PLYEAR']
        if plt_year > 0:
            return row['Datetime'].year - plt_year + 2
        else:
            return np.nan    
    df.loc[:, 'Tree_Age'] = [_get_age(row[1]) for row in df.iterrows()]    

def add_young_tree_flag(df):
    df.loc[:, 'Young_Tree_Flag'] = df['Tree_Age'].apply(lambda x: x <= 8) 

def add_shapefile_origin(df):
    df.loc[:, 'Shapefile_Name'] = df['Polygon_id'].apply(lambda x: x.split('_')[0]) 

def add_woodland_flag(df):
    df.loc[:, 'Woodland_Flag'] = (df['PRIHABITAT'] == 'CONIFEROUS WOODLANDS') | (
                                  df['PRIHABITAT'] == 'BROADLEAVED; MIXED/YEW WOODLANDS') | (
                                  df['PRIHABITAT'] == 'Upland birchwoods') | (
                                  df['PRIHABITAT'] == 'Wet woodland') | (
                                  df['PRIHABITAT'] == 'Upland oakwood') | (
                                  df['PRIHABITAT'] == 'Upland mixed ashwoods') | (
                                  df['PRIHABITAT'] == 'Lowland beech/yew woodland') | (
                                  df['PRIHABITAT'] == 'Non HAP native pinewood') | (
                                  df['PRIHABITAT'] == 'Native pine woodlands') 
                                  #| (
                                  #df['IFT_IOA'] == 'Conifer') | (
                                  #df['IFT_IOA'] == 'Young trees') | (
                                  #df['IFT_IOA'] == 'Broadleaved') | (
                                  #df['IFT_IOA'] == 'Mixed mainly broadleaved') | (
                                  #df['IFT_IOA'] == 'Mixed mainly conifer')
    #assumed_woodland_indices = df['IFT_IOA'] == 'Assumed woodland'
    #df[assumed_woodland_indices, 'Woodland_Flag'] = [np.nan] * len(assumed_woodland_indices)    

def add_train_flag(df):
    train_ids, test_ids = train_test_split(df['Polygon_id'].unique(), test_size=0.33)
    df.loc[:, 'Train_Flag'] = df['Polygon_id'].isin(train_ids)

def add_woodland_category(df):
    def get_woodland_category(primary_habitat):
       coniferous_woodlands = ['CONIFEROUS WOODLANDS']
       broadleaved_woodlands = ['BROADLEAVED; MIXED/YEW WOODLANDS']
       grassland = ['ACID GRASSLAND', 'IMPROVED GRASSLAND', 'NEUTRAL GRASSLAND']
       if primary_habitat in coniferous_woodlands:
           return 'coniferous_woodlands'
       elif primary_habitat in broadleaved_woodlands:
           return 'broadleaved_woodlands'
       elif primary_habitat in grassland:
           return 'grassland'
       elif pd.isnull(primary_habitat):
           return np.NaN
       else:
           return 'other'
    df.loc[:, 'woodland_category'] = [get_woodland_category(primary_habitat_)
                                                      for primary_habitat_ in df['PRIHABITAT']]

def add_tree_class(df):
    def _tree_class(age):
        if age <= 8:
            return 'Young_Tree'
        elif (age > 8) and (age <= 17):
            return 'Medium_Tree'
        elif age > 17:
            return 'Old_Tree'
        else:
            return np.NaN
    df.loc[df['Woodland_Flag'], 'Woodland_Class'] = df[df['Woodland_Flag']]['Tree_Age'].apply(_tree_class)

if __name__ == "__main__":
    dataframe_path = sys.argv[1]
    print dataframe_path
    #sys.exit(0)
    df = pd.read_csv(dataframe_path)
    get_tree_age(df)    
    add_young_tree_flag(df)
    add_shapefile_origin(df)
    add_woodland_flag(df)
    add_train_flag(df)
    add_woodland_category(df)
    add_tree_class(df)
    #new_dataframe_name = os.path.join(dataframe_path.split('/')[:-1], dataframe_path.split('/')[-1].replace('.csv','_artifical_variables.csv'))
    df.to_csv('add_artificial.csv', index=False)