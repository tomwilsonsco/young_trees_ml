import time
import json
import os
import subprocess
import copy
import fiona
import glob
import numpy as np
import pandas as pd
import datetime

try:
    from sortedcontainers import SortedDict
except :
    os.system('pip install sortedcontainers')
    from sortedcontainers import SortedDict
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import MultiPolygon, Polygon

from pyspark import SparkConf, SparkContext

WORK_DIR = "/home/ubuntu/Documents/East_Anglia/"
OUTPUT_DIR = "/home/ubuntu/Documents/East_Anglia/Polygon_Data/"
S3_DIR = 's3://jncc-poc5/East_Anglia_PP5_DataFrame/'
#WORK_DIR = "/home/ubuntu/Documents/Cambria_Multi_Temporal_Filtering/"
#OUTPUT_DIR = "/home/ubuntu/Documents/Cambria_Multi_Temporal_Filtering/Final_Shapes/"
#S3_DIR = 's3://jncc-poc5/Cambria_Raw_Processed_Final/'
#WORK_DIR = "/mnt/Step_5/"
#OUTPUT_DIR = "/mnt/hadoop/Step_5/Final_Shapes/"
#S3_DIR = 's3://jncc-poc5/Cambria_Raw_Processed_Final/'

region = 'EastAnglia' # or 'EastAnglia' or 'Kielder'

def geometry2shapely(geom):
    if geom['type'] == 'Polygon':
        poly = Polygon(geom['coordinates'][0])
    elif geom['type'] == 'MultiPolygon':
        polygons = [Polygon(coords[0]) for coords in geom['coordinates']]
        poly = MultiPolygon(polygons)
    return poly

def clip_with_shape(raster, shape):
    if isinstance(raster,str):
        raster = rasterio.open(raster)
    window = raster.window(*shape.bounds)
    out_transform = raster.window_transform(window)
    out_image = raster.read(window=window, masked=True)
    out_shape = out_image.shape[1:]

    if out_shape[0] * out_shape[1] > 0:
        shape_mask = geometry_mask([shape], transform=out_transform, invert=False,
                                   out_shape=out_shape, all_touched=False)

        out_image.mask = out_image.mask | shape_mask        
        out_image.fill_value = None

        out_image = out_image.filled(np.nan)
        #out_image, out_transform = mask.mask(src, [Polygon(geoms_inters[41]['geometry']['coordinates'][0])], crop=True)
        out_meta = raster.meta.copy()

        # GTiff
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform,
                         "count": 2,
                         'compression': 'lzw',
                         'tiled': True
                         })
        return (out_image, out_transform, out_meta)
    else:
        return [False]

def create_path(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise        

"""
def make_path(polygon, output_path):
    path = False
    ID = str(polygon['properties'][u'SUBCOMPTID']) + '/'
    if polygon['properties']['PRIPCTAREA']==100:
        if (polygon['properties']['PRI_PLYEAR']>=2008) and (polygon['properties']['PRIHABITAT']=='CONIFEROUS WOODLANDS'):
            path = os.path.join(output_path, 'Young_Conifer', str(polygon['properties']['PRI_PLYEAR']), ID)
        elif (polygon['properties']['PRI_PLYEAR']>=2008) and (polygon['properties']['PRIHABITAT']=='BROADLEAVED; MIXED/YEW WOODLANDS'):
            path = os.path.join(output_path, 'Young_Broadleaved', str(polygon['properties']['PRI_PLYEAR']), ID)
        elif (polygon['properties']['PRI_PLYEAR']<=1980) and (polygon['properties']['PRIHABITAT']=='CONIFEROUS WOODLANDS'):
            path = os.path.join(output_path, 'Old_Conifer', ID)
        elif (polygon['properties']['PRI_PLYEAR']<=1980) and (polygon['properties']['PRIHABITAT']=='BROADLEAVED; MIXED/YEW WOODLANDS'):
            path = os.path.join(output_path, 'Old_Broadleaved', ID)
        elif (polygon['properties']['PRIHABITAT'] == 'ACID GRASSLAND') or (polygon['properties']['PRIHABITAT'] == 'IMPROVED GRASSLAND'):
            path = os.path.join(output_path, 'Grassland', ID)
        if path:        
            create_path(path)
        return path
    else:
        return False
"""

def make_path_old(polygon, output_path):
    path = False
    # identify if the polygon is from NFE, NFI or Felling Licenses
    if 'SUBCOMPTID' in polygon['properties'].keys():
        poly_type = 'NFE'
    elif 'OBJECTID' in polygon['properties'].keys():
        poly_type = 'NFI'
    else:
        poly_type = 'Felling_License'

    if poly_type == 'NFE':
        ID = str(polygon['properties'][u'SUBCOMPTID']) + '/'
        if polygon['properties']['PRIHABITAT']=='CONIFEROUS WOODLANDS':
            path = os.path.join(output_path, 'Conifer', str(polygon['properties']['PRI_PLYEAR']), ID)
        elif polygon['properties']['PRIHABITAT']=='BROADLEAVED; MIXED/YEW WOODLANDS':
            path = os.path.join(output_path, 'Broadleaved', str(polygon['properties']['PRI_PLYEAR']), ID)
        elif (polygon['properties']['PRIHABITAT'] == 'ACID GRASSLAND') or (
              polygon['properties']['PRIHABITAT'] == 'IMPROVED GRASSLAND') or (
              polygon['properties']['PRIHABITAT'] == 'NEUTRAL GRASSLAND'):
            path = os.path.join(output_path, 'Grassland', ID)
        elif polygon['properties']['PRIHABITAT'] == 'INLAND ROCK':
            path = os.path.join(output_path, 'Rock', ID)
        elif (polygon['properties']['PRIHABITAT'] == 'STANDING OPEN WATER/CANALS') or (
              polygon['properties']['PRIHABITAT'] == 'RIVERS & STREAMS') or (
              polygon['properties']['PRIHABITAT'] == 'Ponds'):
             path = os.path.join(output_path, 'Water', ID)
        else:
            path = os.path.join(output_path, 'Other', ID)
    elif poly_type == 'NFI':
        ID = str(polygon['properties'][u'OBJECTID']) + '/'    
        if polygon['properties'][u'IFT_IOA'] == 'Felled':
            path = os.path.join(output_path, 'Felled', ID)
        elif polygon['properties'][u'IFT_IOA'] == 'Bare area':
            path = os.path.join(output_path, 'Bare_Area', ID)
        elif polygon['properties'][u'IFT_IOA'] == 'Road':
            path = os.path.join(output_path, 'Road', ID)
        else:
            path = os.path.join(output_path, 'Other', ID)
    elif poly_type == 'Felling_License':
        ID = 'FL_'+str(polygon['id']) + '/'
        category = '_'.join(polygon['properties'][u'Descriptr'].replace('/','_').replace('(','').replace(')','').split(' '))
        path = os.path.join(output_path, category, ID)
    if path:        
        create_path(path)
    return path

def get_polygon_id(polygon):
    if 'SUBCOMPTID' in polygon['properties'].keys():
        ID = 'NFE_' + str(polygon['properties'][u'SUBCOMPTID'])
    elif 'OBJECTID' in polygon['properties'].keys():
        ID = 'NFI_' + str(polygon['properties'][u'OBJECTID'])
    elif u'Fell_ref' in polygon['properties'].keys():
        ID = 'FL_' + str(polygon['id'])    
    elif u'DateApprv' in polygon['properties']:
        ID = 'GR_' + str(polygon['id'])        
    return ID


def make_path(polygon, output_path):
    path = False
    # identify if the polygon is from NFE, NFI or Felling Licenses
    ID = get_polygon_id(polygon) + '/'
    path = os.path.join(output_path, ID)
    if path:        
        create_path(path)
    return path

def save_polygon(polygon, all_metadata):
    d = SortedDict([(m,'') for m in all_metadata])
    d.update(polygon['properties'])
    return d.values()


def aws_sync_fast(s3_dir_, dest_dir_):
    st = time.time()
    #print '\n\Downloading file:\t',s3_dir_.split('/')[-1], '\n' 
    cmd_code = ['aws', 's3', 'sync', s3_dir_, dest_dir_, '--profile', 'poc5']
    try:
        output = subprocess.check_output(cmd_code)
    except subprocess.CalledProcessError:
        pass
    #assert str(output)=='0', 'upload not successful: {}'.format()


def save_image(image_filename, image_tuple, polygon, output_path):    
    poly_id = get_polygon_id(polygon)
    path = make_path(polygon, output_path) 
    if path and (len(image_tuple)==3):        
        #image_save_name = '_'.join(image_filename.split('/')[-1].split('_')[:6] +['ID', poly_id[:-1], '_NFE']) + '.tif'
        image_save_name = image_filename.split('/')[-1].replace('.tif','') + '_ID_' + poly_id[:-1] + '.tif'
        with rasterio.open(os.path.join(path, image_save_name), "w", **image_tuple[2]) as dest:
            try:                
                dest.write(image_tuple[0])
                #upload_command = 'aws s3 cp {} {} --profile poc5'.format(os.path.join(path, image_save_name), S3_DIR).split(' ')
                #subprocess.check_output(upload_command)
                return os.path.join(S3_DIR, poly_id, image_save_name)
            except Exception as e:                          
                return False
    else:        
        return 'wrong_format'

def toCSVLine(data_row):
    polygon_properties = ','.join(str(d).replace(',','') for d in data_row[3])+'\n'
    try:
        return str(data_row[0]) +',' + data_row[1] + ',' + data_row[2] + ',' + polygon_properties
    except TypeError:
        return False

def write_to_file(data_row, filename):
    if data_row:
        f = open(filename, 'a')
        f.write(data_row)
        f.close()

def get_datetime(tif_filename):
    datetime_str = tif_filename.split('/')[-1].split('_')[4]
    return datetime_str

def upload_completed():
    # this checks if all files have been uploaded to S3 by checking if there is a difference between local and s3 files
    ps1 = subprocess.Popen('aws s3 ls {} --recursive --profile poc5'.format(S3_DIR).split(' '), stdout=subprocess.PIPE)
    no_files_s3 = int(subprocess.check_output(['wc','-l'], stdin=ps1.stdout).replace('\n',''))
    ps1.stdout.close()
    ps2 = subprocess.Popen('find {} -type f'.format(OUTPUT_DIR).split(' '), stdout=subprocess.PIPE)        
    no_files_local = int(subprocess.check_output(['wc','-l'], stdin=ps2.stdout).replace('\n',''))
    ps2.stdout.close()
    if no_files_s3 > no_files_local:
        return False
    else:
        return True

def get_shapefiles(region):
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)        
    if not os.path.exists(os.path.join(WORK_DIR, 'Auxiliary_Data')):
        os.makedirs(os.path.join(WORK_DIR,'Auxiliary_Data'))
    if not os.path.exists(os.path.join(WORK_DIR, 'Auxiliary_Data',region)):
        os.makedirs(os.path.join(WORK_DIR,'Auxiliary_Data',region))
    if region=='Cambria':
        download_command = 'aws s3 sync s3://jncc-poc5/Cambrian_Area/ {}'.format(os.path.join(WORK_DIR, 
                                                                                  'Auxiliary_Data', region+'/')).split(' ')
    elif region=='EastAnglia':        
        download_command = 'aws s3 sync s3://jncc-poc5/East_Anglia_Area/Auxiliary_Data/ {}'.format(os.path.join(WORK_DIR, 
                                                                                  'Auxiliary_Data', region+'/')).split(' ')        
    elif region=='Kielder':
        download_command = 'aws s3 sync s3://jncc-poc5/Kielder_Area/Auxiliary_Data/ {}'.format(os.path.join(WORK_DIR, 
                                                                                  'Auxiliary_Data', region+'/')).split(' ')    
    subprocess.check_call(download_command)    
    shapefiles = glob.glob(os.path.join(WORK_DIR, 'Auxiliary_Data/{}/*/*.shp'.format(region)))
    shapefiles += glob.glob(os.path.join(WORK_DIR, 'Auxiliary_Data/{}/*/2015/*.shp'.format(region)))
    print "shapefiles found "+str(len(shapefiles))
    return shapefiles

def download_path(path, region, DOWNLOAD_FLAG, aws_creds):
    tif_name = path.split('/')[-1]
    #download_command = 'aws s3 cp s3://jncc-poc5/{}_Raw_Processed_4/{} {}'.format(region, tif_name, path)
    download_command = 'aws s3 cp s3://jncc-poc5/{}_Raw_Processed_4/{} {}'.format(region, tif_name, '.')
    if DOWNLOAD_FLAG:
        subprocess.check_call(download_command.split(' '), env=aws_creds)
        return '~/'+tif_name
    else:
        return path

if __name__ == "__main__":
    conf = SparkConf().setMaster("local").setAppName("My App")
    sc = SparkContext(conf = conf)
    # dont display the INFO logging
    sc.setLogLevel("ERROR")

    output_filename = '{}_dataframe.csv'.format(region)

    shapefiles = get_shapefiles(region)

    all_metadata = []
    polygons = []
    for shp in shapefiles:
        all_metadata += fiona.open(shp)[0]['properties'].keys()
                       #fiona.open(nfi_shapefile)[0]['properties'].keys() + \
                       #fiona.open(felling_licenses_shapefile)[0]['properties'].keys() + \
                       #fiona.open(grants_shapefile)[0]['properties'].keys()
        polygons += [s for s in fiona.open(shp)]    
    
    polygons = sc.parallelize(polygons).cache()
        
    #all_images = glob.glob(os.path.join(WORK_DIR, 'Cambria_Raw_Processed_4/*.tif'))
    #all_images = glob.glob(os.path.join('/home/ubuntu/Documents', 'East_Anglia_Stitch_Shift/Shifted/*.tif'.format(region)))
    all_images = glob.glob(os.path.join('/home/ubuntu/Documents', 'East_Anglia/Shifted/*.tif'.format(region)))
    print "all_images is " +str(len(all_images))
    #all_images = glob.glob(os.path.join(WORK_DIR, 'processed_4b/*.tif'))

    # batches of 1 day each
    # dates = [d.strftime('%Y%m%d') for d in pd.date_range(datetime.date(2015,1,1), datetime.date(2016,12,31))]
    # batches of 1 month each
    #dates = ['2016'+"%02d"%m for m in range(1,13)]
    #dates = [year+"%02d"%m for year in ['2016'] for m in range(1,2)]
    #dates = [year+"%02d"%m for year in ['2015', '2016'] for m in range(1,13)]
    dates = ['201605']
    LOCAL_DOWNLOAD = False
    aws_creds =  {'AWS_ACCESS_KEY_ID': os.environ['AWS_ACCESS_KEY_ID'], 
                 'AWS_SECRET_ACCESS_KEY':os.environ['AWS_SECRET_ACCESS_KEY']}
    #aws_creds =  {}
    for date in dates:
        print date
        date_images = [img for img in all_images if date in img]
        if len(date_images)==0:
            continue
        else:
            start = time.time()
            ard_images = sc.parallelize(date_images)
            #ard_images.map(lambda x: rasterio.open(x))

            pairs = ard_images.cartesian(polygons)

            output_rdd = pairs.map(lambda x: (download_path(x[0], region, LOCAL_DOWNLOAD, aws_creds),x[1])).\
                               map(lambda x: (x[0], x[1], geometry2shapely(x[1]['geometry']))).\
                               map(lambda x: (x[0], clip_with_shape(x[0], x[2]), x[1])).\
                               map(lambda x: (get_polygon_id(x[2]), 
                                              save_image(x[0], x[1], x[2], OUTPUT_DIR), 
                                              get_datetime(x[0]),
                                              save_polygon(x[2], all_metadata)))

            lines = output_rdd.map(toCSVLine)
            lines.map(lambda x: write_to_file(x, output_filename)).collect()
            aws_sync_fast(OUTPUT_DIR, S3_DIR)            
            subprocess.check_output('rm -rf {}'.format(OUTPUT_DIR).split(' '))
            print "Date {} took {} seconds".format(date, time.time()-start)

    # write the results to file
    column_names = ['polygon_id','filepath']+all_metadata
    df = lines.map(lambda x: x.split(',')).toDF(column_names).toPandas()
    lines.saveAsTextFile(output_filename)
    # add a header
    header = ','.join(['Polygon_id','Filepath','Date']+list(SortedDict([(m,'') for m in all_metadata]).keys()))
    os.system("sed -i -e '1i{}' {}\\".format(header, output_filename)[:-1])
    os.system('aws s3 cp {} {} --profile poc5'.format(os.path.join(WORK_DIR, output_filename), S3_DIR))
    
    # now copy all this to S3 and delete locally
    print('processing was sucessful!')