#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:17:03 2017

@author: ubuntu
"""


#==============================================================================
# Import modules
#==============================================================================
import rasterio
import itertools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys


#==============================================================================
# Define Utility Functions
#==============================================================================
# Get polygon ids: yt vs nyt
def get_polygon_ids_yt_vs_nyt(df):
    yt_df = df[df['Woodland_Class'] == 'Young_Tree']
    nyt_df = df[df['Woodland_Class'] <> 'Young_Tree']
    return list(np.unique(yt_df['Polygon_id'])), list(np.unique(nyt_df['Polygon_id']))

# Get polygon ids: yt vs nwl
def get_polygon_ids_yt_vs_nyt(df):
    yt_df = df[df['Woodland_Class'] == 'Young_Tree']
    nyt_df = df[df['Woodland_Class'] <> 'Young_Tree']
    return list(np.unique(yt_df['Polygon_id'])), list(np.unique(nyt_df['Polygon_id']))

# Subsample polygon ids by taking all polygons ids for smallest set and 3x that size in larger set
def subsample_poly_ids(poly_ids_list_1, poly_ids_list_2, factor):
    size_list_1 = len(poly_ids_list_1)
    size_list_2 = len(poly_ids_list_2)
    min_size = min(size_list_1, size_list_2)
    max_size = max(size_list_1, size_list_2)
    valid_factor = min(factor, float(max_size)/min_size)
    
    if size_list_1 <= size_list_2:
        subsample_list_1 = poly_ids_list_1
        subsample_list_2 = random.sample(poly_ids_list_2, int(valid_factor * min_size))
    elif size_list_1 > size_list_2:
        subsample_list_1 = random.sample(poly_ids_list_1, int(valid_factor * min_size))
        subsample_list_2 = poly_ids_list_2
    else:
        raise Exception('Error in data')
    
    return subsample_list_1, subsample_list_2

def add_woodland_category(df):
    def get_woodland_category(primary_habitat):
       coniferous_woodlands = ['CONIFEROUS WOODLANDS']
       broadleaved_woodlands = ['BROADLEAVED; MIXED/YEW WOODLANDS']
       grassland = ['ACID GRASSLAND', 'IMPROVED GRASSLAND', 'NEUTRAL GRASSLAND', 'Lowland calcareous grassland', 'CALCAREOUS GRASSLAND', 'BRACKEN', 'Lowland heathland', 'Lowland dry acid grassland', 'Lowland meadows']
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

def add_woodland_flag(df):
    df.loc[:, 'Woodland_Flag'] = (df['PRIHABITAT'] == 'CONIFEROUS WOODLANDS') | (
                                  df['PRIHABITAT'] == 'BROADLEAVED; MIXED/YEW WOODLANDS') | (
                                  df['PRIHABITAT'] == 'Lowland Mixed Deciduous Woodland') | (
                                  df['PRIHABITAT'] == 'Upland birchwoods') | (
                                  #df['PRIHABITAT'] == 'Wet woodland') | (
                                  df['PRIHABITAT'] == 'Upland oakwood') | (
                                  df['PRIHABITAT'] == 'Upland mixed ashwoods') | (
                                  df['PRIHABITAT'] == 'Lowland beech/yew woodland') | (
                                  df['PRIHABITAT'] == 'Non HAP native pinewood') | (
                                  df['PRIHABITAT'] == 'Native pine woodlands')
    
#==============================================================================
# Main
#==============================================================================
if __name__ == '__main__':
    # Load dataframe
    df = pd.read_csv('/home/ubuntu/Documents/East_Anglia_Model_Build/Polygon_Reference_DataFrame/EastAnglia_dataframe_artificial_variables.csv')
    
    df = df[df['Filepath'] <>'wrong_format']
    print 'Length of DF:\t', len(df)
    print 'Length of DF drop duplicates:\t', len(df.drop_duplicates('Filepath'))
    df = df.drop_duplicates('Filepath').reset_index(drop = True)
    print 'Number of unique polygons:\t', len(np.unique(df['Polygon_id']))
    print 'Number of young tree polygons:\t', len(np.unique(df[df['Woodland_Class'] == 'Young_Tree']['Polygon_id']))
    
    
    # Set pandas display options
    pd.options.display.max_rows = 30
    pd.options.display.max_columns = 999
    
    #df.head
    #np.unique(df['Woodland_Class'])
    print '\n', df['PRIHABITAT'].value_counts(), '\n'
    print df['woodland_category'].value_counts(), '\n'
    #df[df['Woodland_Class'].isin(['Young_Tree', np.nan])]
    
    
    # Number of young trees for 1 timestamp
    print 'Number of young tree polygons in one timestamp:\t', np.sum(df[df['Date'] == '20160105T060622']['Woodland_Class']=='Young_Tree')
    
    # Train test split
    train_df = df[df['Train_Flag'] == True]
    test_df = df[df['Train_Flag'] == False]
    print 'Number of polygons in train:\t', len(np.unique(train_df['Polygon_id']))
    print 'Number of young tree polygons in train:\t', len(np.unique(train_df[train_df['Woodland_Class'] == 'Young_Tree']['Polygon_id']))
    
    
    #==============================================================================
    # Subsample Polygon IDs In Train: Young Trees VS Non-Young Trees (conifers)
    #==============================================================================
    train_filtered_df = train_df[(train_df['Date'] == train_df.iloc[0]['Date']) & (train_df['PRIPCTAREA'] >= 90) & ((train_df['woodland_category'] == 'coniferous_woodlands') | (train_df['woodland_category'] == 'grassland'))]
    
    # Young tree / non-young tree polygon ids
    yt_poly_ids, nyt_poly_ids = get_polygon_ids_yt_vs_nyt(train_filtered_df)
    
    # Subsample
    yt_polygon_ids, nyt_polygon_ids = subsample_poly_ids(yt_poly_ids, nyt_poly_ids, 3)
    subsampled_df = train_df[train_df['Polygon_id'].isin(yt_polygon_ids + nyt_polygon_ids)]
    subsampled_filtered_df = subsampled_df[(subsampled_df['PRIPCTAREA'] >= 90) & ((subsampled_df['woodland_category'] == 'coniferous_woodlands') | (subsampled_df['woodland_category'] == 'grassland'))]
    
    # Save
    subsampled_filtered_df.to_csv('/home/ubuntu/Documents/East_Anglia_Model_Build/Polygon_Reference_DataFrame/East_Anglia_subsampled_DF_conif_YT_VS_NYT.csv')
    

    #==============================================================================
    # Subsample Polygon IDs In Train: Young Trees VS Non-Young Trees (Conifers, Redefined grassland)
    #==============================================================================
    # Redefine
    pd.options.display.max_rows = 30
    print '\n', train_df['PRIHABITAT'].value_counts(), '\n'
    print train_df['woodland_category'].value_counts(), '\n'
    add_woodland_category(train_df)
    print '\n', train_df['PRIHABITAT'].value_counts(), '\n'
    print train_df['woodland_category'].value_counts(), '\n'
    train_filtered_df = train_df[(train_df['Date'] == train_df.iloc[0]['Date']) & (train_df['PRIPCTAREA'] >= 90) & ((train_df['woodland_category'] == 'coniferous_woodlands') | (train_df['woodland_category'] == 'grassland'))]
    
    # Young tree / non-young tree polygon ids
    yt_poly_ids, nyt_poly_ids = get_polygon_ids_yt_vs_nyt(train_filtered_df)
    
    # Subsample
    yt_polygon_ids, nyt_polygon_ids = subsample_poly_ids(yt_poly_ids, nyt_poly_ids, 3)
    subsampled_df = train_df[train_df['Polygon_id'].isin(yt_polygon_ids + nyt_polygon_ids)]
    subsampled_filtered_df = subsampled_df[(subsampled_df['PRIPCTAREA'] >= 90) & ((subsampled_df['woodland_category'] == 'coniferous_woodlands') | (subsampled_df['woodland_category'] == 'grassland'))]
    
    # Save
    subsampled_filtered_df.to_csv('/home/ubuntu/Documents/East_Anglia_Model_Build/Polygon_Reference_DataFrame/East_Anglia_subsampled_DF_conif_YT_VS_NYT_redefined_grassland.csv')
    

    #==============================================================================
    # Subsample Polygon IDs In Train: Young Trees VS Non-Young Trees (Broadleaves, Redefined grassland)
    #==============================================================================
    train_filtered_df = train_df[(train_df['Date'] == train_df.iloc[0]['Date']) & (train_df['PRIPCTAREA'] >= 90) & ((train_df['woodland_category'] == 'broadleaved_woodlands') | (train_df['woodland_category'] == 'grassland'))]
    
    # Young tree / non-young tree polygon ids
    yt_poly_ids, nyt_poly_ids = get_polygon_ids_yt_vs_nyt(train_filtered_df)
    
    # Subsample
    yt_polygon_ids, nyt_polygon_ids = subsample_poly_ids(yt_poly_ids, nyt_poly_ids, 3)
    subsampled_df = train_df[train_df['Polygon_id'].isin(yt_polygon_ids + nyt_polygon_ids)]
    subsampled_filtered_df = subsampled_df[(subsampled_df['PRIPCTAREA'] >= 90) & ((subsampled_df['woodland_category'] == 'broadleaved_woodlands') | (subsampled_df['woodland_category'] == 'grassland'))]
    
    # Save
    subsampled_filtered_df.to_csv('/home/ubuntu/Documents/East_Anglia_Model_Build/Polygon_Reference_DataFrame/East_Anglia_subsampled_DF_broadleaves_YT_VS_NYT_redefined_grassland.csv')
    
    
    #==============================================================================
    # Subsample Polygon IDs In Train: Young Trees VS Non-Woodland (Redefined grassland)
    #==============================================================================
    # Redefine
    print '\n', train_df['Woodland_Flag'].value_counts(), '\n'
    add_woodland_flag(train_df)
    print '\n', train_df['Woodland_Flag'].value_counts(), '\n'  
    train_filtered_df = train_df[(train_df['Date'] == train_df.iloc[0]['Date']) & (train_df['PRIPCTAREA'] >= 90) & (((train_df['Woodland_Flag'] == True) & (train_df['Young_Tree_Flag'] == True)) | ((train_df['Woodland_Flag'] == False) & (train_df['Young_Tree_Flag'] == False)))]
    
    # Young tree / non-young tree polygon ids
    yt_poly_ids, nyt_poly_ids = get_polygon_ids_yt_vs_nyt(train_filtered_df)
    
    # Subsample
    yt_polygon_ids, nyt_polygon_ids = subsample_poly_ids(yt_poly_ids, nyt_poly_ids, 3)
    subsampled_df = train_df[train_df['Polygon_id'].isin(yt_polygon_ids + nyt_polygon_ids)]
    subsampled_filtered_df = subsampled_df[(subsampled_df['PRIPCTAREA'] >= 90) & (((subsampled_df['Woodland_Flag'] == True) & (subsampled_df['Young_Tree_Flag'] == True)) | ((subsampled_df['Woodland_Flag'] == False) & (subsampled_df['Young_Tree_Flag'] == False)))]
    
    # Save
    subsampled_filtered_df.to_csv('/home/ubuntu/Documents/East_Anglia_Model_Build/Polygon_Reference_DataFrame/East_Anglia_subsampled_DF_YT_VS_NWL_redefined_grassland.csv')
    
    
    #==============================================================================
    # Subsample Polygon IDs In Train: Young Trees VS Grassland (Redefined grassland)
    #==============================================================================
    train_filtered_df = train_df[(train_df['Date'] == train_df.iloc[0]['Date']) & (train_df['PRIPCTAREA'] >= 90) & (((train_df['Woodland_Flag'] == True) & (train_df['Young_Tree_Flag'] == True)) | ((train_df['woodland_category'] == 'grassland')))]
    
    # Young tree / non-young tree polygon ids
    yt_poly_ids, nyt_poly_ids = get_polygon_ids_yt_vs_nyt(train_filtered_df)
    
    # Subsample
    yt_polygon_ids, nyt_polygon_ids = subsample_poly_ids(yt_poly_ids, nyt_poly_ids, 3)
    subsampled_df = train_df[train_df['Polygon_id'].isin(yt_polygon_ids + nyt_polygon_ids)]
    subsampled_filtered_df = subsampled_df[(subsampled_df['PRIPCTAREA'] >= 90) & (((subsampled_df['Woodland_Flag'] == True) & (subsampled_df['Young_Tree_Flag'] == True)) | ((train_df['woodland_category'] == 'grassland')))]
    
    # Save
    subsampled_filtered_df.to_csv('/home/ubuntu/Documents/East_Anglia_Model_Build/Polygon_Reference_DataFrame/East_Anglia_subsampled_DF_YT_VS_redefined_grassland.csv')
                        
    # Validation
    #len(np.unique(train_filtered_df['Polygon_id']))
    #train_filtered_df.sort_values('Polygon_id')['Filepath'].iloc[0] == train_filtered_df.sort_values('Polygon_id')['Filepath'].iloc[1]
    #
    #pd.options.display.max_rows = 50
    #train_df.loc[3756,:]['SHAPE_Leng'] == train_df.loc[437940,:]['SHAPE_Leng']
    #
    #train_df.loc[3756,:]
    #train_filtered_df.loc[3756,:]