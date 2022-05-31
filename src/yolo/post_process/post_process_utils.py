import geopandas as gpd

import os
import shutil
from tqdm import tqdm


def collect_plots(data_src, output_folder, conf = 0.6):
    """
    Collect images from all batches in a single folder.
    
    Params:
        data_src (str): folder containing prediction batch folders
        output_folder (str): folder where all images will be collected
        
    Returns:
        None
    """
    
    batch_list = os.listdir(data_src)
    conf = str(conf).split('.')[-1]
    
    os.makedirs(output_folder, exist_ok = True)

    for batch in tqdm(batch_list):
        src_folder = os.path.join(data_src, batch, f'results/pred_plots_0p{conf}')
        file_list = os.listdir(src_folder)

        for file_to_copy in file_list:
            src = os.path.join(src_folder, file_to_copy)
            dst = os.path.join(output_folder, file_to_copy)

            shutil.copy(src, dst)
            
            
def collect_geojsons(data_src, output_folder, obj_type = 'box', conf = 0.6):
    """
    Collect geojson files from all batches in a single folder.
    
    Params:
        data_src (str): folder containing prediction batch folders
        output_folder (str): folder where all geojson files will be collected
        obj_type (str): type of geojson file to copy. 'box' or 'centroid'
        conf (float): prediction confidence threshold
        
    Returns:
        None
    """
    
    batch_list = os.listdir(data_src)
    conf = str(conf).split('.')[-1]
    
    os.makedirs(output_folder, exist_ok = True)

    for batch in tqdm(batch_list):
        src_folder = os.path.join(data_src, batch, 'results',f'geojsons_geo_0p{conf}')
        file_list = sorted(os.listdir(src_folder))
        
        if obj_type == 'box':
            file_list = [i for i in file_list if 'centroid' not in i]
        elif obj_type == 'centroid':
            file_list = [i for i in file_list if 'centroid' in i]
        
        for file_to_copy in file_list:
            src = os.path.join(src_folder, file_to_copy)
            dst = os.path.join(output_folder, file_to_copy)

            shutil.copy(src, dst)
            
            
def collect_geojsons_pix(data_src, output_folder, conf = 0.6):
    """
    Collect geojson files for image pixel coordinates from all batches in a single folder.
    
    Params:
        data_src (str): folder containing prediction batch folders
        output_folder (str): folder where all geojson files will be collected
        conf (float): prediction confidence threshold
        
    Returns:
        None
    """
    
    batch_list = os.listdir(data_src)
    conf = str(conf).split('.')[-1]

    os.makedirs(output_folder, exist_ok = True)

    for batch in tqdm(batch_list):
        src_folder = os.path.join(data_src, batch, 'results',f'geojsons_pix_0p{conf}')
        file_list = os.listdir(src_folder)

        for file_to_copy in file_list:
            src = os.path.join(src_folder, file_to_copy)
            dst = os.path.join(output_folder, file_to_copy)

            shutil.copy(src, dst)
            
            
def merge_geojsons(geojson_folder, save_dir):
    """
    Collect images from all batches in a single folder.
    
    Params:
        geojson_folder (str): folder containing geojson files to be merged
        save_dir (str): save path for merged geojson file
        
    Returns:
        None
    """

    geojson_list = sorted(os.listdir(geojson_folder))

    assert(len(geojson_list) > 0)

    # read the first geojson file
    df = gpd.read_file(os.path.join(geojson_folder, geojson_list[0]))
    df['image_id'] = geojson_list[0].split('_EPSG')[0]

    for item in tqdm(geojson_list[1:]):

        geojson_dir = os.path.join(geojson_folder, item)
        data = gpd.read_file(geojson_dir)
        data['image_id'] = item.split('_EPSG')[0]

        df = df.append(data, ignore_index = True)

    df.to_file(save_dir)