import sys
import datetime
import dill
import subprocess
import pandas as pd

#WORK_DIR = "/home/ubuntu/Documents/Earth_Observation/Exploration/"
#WORK_DIR = "/home/ubuntu/Documents/Earth_Observation/Kielder/"
WORK_DIR = "/home/ubuntu/Documents/East_Anglia_Model_Build/Weather_Data/"

def get_date(datetime_str):
    return datetime.datetime.strptime(datetime_str, '%Y%m%dT%H%M%S')

def path2datetime(path):
    return get_date(path.split('/')[-1].split('_')[4])

def download_weather(region):
    weather_dict = {'Cambria':'cambrian_weather.csv',
                     'East_Anglia': 'east_anglia_weather.csv',
                     'Kielder': 'kielder_weather.csv'}    
    download_command = 'aws s3 cp s3://jncc-poc5/Weather/{} {}'.format(weather_dict[region],WORK_DIR+weather_dict[region])
    subprocess.check_output(download_command.split(' '))
    return weather_dict[region]    

def download_rainfall(region):
    rainfall_dict = {'Cambria':'Cambrian_rainfall.csv',
                     'East_Anglia': 'East_Anglia_rainfall.csv',
                     'Kielder': 'Kielder_rainfall.csv'}    
    download_command = 'aws s3 cp s3://jncc-poc5/Rainfall/{} {}'.format(rainfall_dict[region],WORK_DIR+rainfall_dict[region])
    subprocess.check_output(download_command.split(' '))
    return rainfall_dict[region]

def download_seasonality():
    download_command = 'aws s3 cp s3://jncc-poc5/seasonality_feature.csv {}'.format(WORK_DIR)
    subprocess.check_output(download_command.split(' '))
    return 'seasonality_feature.csv'

def add_rainfall(feature_df, rainfall_dataframe_name):
    rain_df = pd.read_csv(rainfall_dataframe_name)
    rain_df.index = rain_df[['Year','Month']].apply(lambda x: datetime.datetime(x['Year'],x['Month'],1),axis=1)
    rain_df = rain_df.resample('1H').pad()
    rain_df['Datetime'] = rain_df.index
    rain_df = rain_df.reset_index().drop(['Year','Month','index'],axis=1)    
    rain_df['Datetime'] = rain_df['Datetime'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d %H:00:00'))
    feature_df['Datetime'] = feature_df['Local_Filepath'].apply(path2datetime)
    feature_df.loc[:,'Datetime'] = feature_df['Datetime'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d %H:00:00'))
    merged_df = feature_df.merge(rain_df, on='Datetime', how='left')
    return merged_df.drop(['Datetime'], axis=1)

def add_weather(feature_df, weather_dataframe_name):
    weather_df = pd.read_csv(weather_dataframe_name).drop_duplicates()
    feature_df['Datetime'] = feature_df['Local_Filepath'].apply(path2datetime)
    feature_df.loc[:,'Datetime'] = feature_df['Datetime'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d %H:00:00'))
    merged_df = feature_df.merge(weather_df, on='Datetime', how='left')
    return merged_df.drop(['Datetime','PrecipType'], axis=1)

def add_seasonality(feature_df, seasonality_dataframe_name):
    seasonality_df = pd.read_csv(seasonality_dataframe_name).drop_duplicates()
    feature_df['Datetime'] = feature_df['Local_Filepath'].apply(path2datetime)
    feature_df.loc[:,'Datetime'] = feature_df['Datetime'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d %H:00:00'))
    merged_df = feature_df.merge(seasonality_df, on='Datetime', how='left')
    return merged_df.drop(['Datetime'], axis=1)

if __name__ == "__main__":
    
    experiment_name = '/home/ubuntu/Documents/East_Anglia_Model_Build/Experiments/YT_VS_NWL_redefined_grassland_TEST_temporal_5_10'
    region = 'East_Anglia'
    
    rainfall_csv = download_rainfall(region)
    season_csv = download_seasonality()

    
    with open(experiment_name + '.pkl', 'r') as df_file2:
        input_df, a,b,c  =  dill.load(df_file2)
    
    input_df = add_seasonality(input_df, WORK_DIR+season_csv)
    input_df = add_rainfall(input_df, WORK_DIR+rainfall_csv)
    
    with open(experiment_name + '_season_rainfall.pkl', 'w') as fp:
            dill.dump((input_df, a, b, c), fp)

    print input_df.columns
    
    