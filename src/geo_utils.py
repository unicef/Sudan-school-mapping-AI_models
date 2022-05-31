import os
import rasterio
import geopandas as gpd
import shapely
from shapely.geometry import box
from tqdm import tqdm


def parse_txt(txt_dir):
    """
    Read txt file.
    bbox format - xmin, ymin, xmax, ymax (unnormalized).
    
    Params:
        txt_dir (str): path to text file containing bboxes.
        
    Returns: 
        example - [[xmin, ymin, xmax, ymax],
                   [xmin, ymin, xmax, ymax]]
    """
    
    with open(txt_dir, 'r') as file:
        content = [i.strip().split(',') for i in file.readlines()]
        bboxes = [list(map(float, i)) for i in content]

    return bboxes


def parse_geojson(geojson_dir):
    """
    Read geojson file.
    
    Params:
        geojson_dir (str): path to geosjon file containing coordinates and crs system. For geo-referencing.
        
    Returns: 
        image_id (str)
        src_crs (source crs) 
        left (float)
        top (float)
        right (float) 
        bottom (float) 
    """
    
    # read geojson file
    geo_df = gpd.read_file(geojson_dir)
    
    image_id = str(geo_df.iloc[0]['id'].item())

    left = geo_df.iloc[0]['left'].item()
    top = geo_df.iloc[0]['top'].item()
    right = geo_df.iloc[0]['right'].item()
    bottom = geo_df.iloc[0]['bottom'].item()

    src_crs = geo_df.crs
    
    return image_id, src_crs, left, top, right, bottom


def produce_geo_files(model_output_folder, geojson_folder, output_folder):
    """
    Geo-reference bounding boxes(model predictions) from text files and produce geojson files.
    
    Params:
        model_output_folder (str): folder containing model prediction text files
        geojson_folder (str): folder containing geojson files to be used for geo-referencing
        output_folder (str): folder where final geojson files containing geo-referenced model predictions will be produced.
        
    Returns:
        None
    """
    
    txt_file_list = os.listdir(model_output_folder)
    filename_list = [os.path.splitext(i)[0] for i in txt_file_list]

    os.makedirs(output_folder, exist_ok = True)

    # for each text file
    for filename in filename_list:

        # w, h assumed to be 1000x1000
        image_width, image_height = 1000, 1000

        # file dirs
        geojson_dir = os.path.join(geojson_folder, filename + '.geojson')
        txt_dir = os.path.join(model_output_folder, filename + '.txt')

        # get bounding box list from txt file
        bboxes = parse_txt(txt_dir)
        
        # get geo-information for current png image tile
        image_id, src_crs, left, top, right, bottom = parse_geojson(geojson_dir)
        
        # used for mapping image pixel values to geo-coordinates
        affine_tfm = rasterio.transform.from_bounds(west = left, south = bottom, east = right, north = top,
                                                    width = image_width, height = image_height)

        bbox_geom_list, centroid_geom_list = [], []

        # for each bbox in current txt file
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            print('box coords:', xmin, ymin, xmax, ymax)

            # geo-reference bounding box
            bbox_geom = pix_to_geo_coords(affine_tfm, xmin, ymin, xmax, ymax)

            # centroid of bounding box
            bbox_centroid = bbox_geom.centroid

            # append geo-registered bounding box and centroid
            bbox_geom_list.append([bbox_geom])
            centroid_geom_list.append([bbox_centroid])

        # create 2 dataframes - one for bbox and one for centroid
        bbox_geo_df = gpd.GeoDataFrame(bbox_geom_list, columns=['geometry'], crs=src_crs)
        centroid_geo_df = gpd.GeoDataFrame(centroid_geom_list, columns=['geometry'], crs=src_crs)

        # save dirs for 2 dataframes
        bbox_gdf_save_dir = os.path.join(output_folder, filename + '_box' + '.geojson')
        centroid_gdf_save_dir = os.path.join(output_folder, filename + '_centroid' + '.geojson')

        # save 2 dataframes
        bbox_geo_df.to_file(bbox_gdf_save_dir, driver='GeoJSON')
        centroid_geo_df.to_file(centroid_gdf_save_dir, driver='GeoJSON')


def split_geojsons(geojson_dir, output_folder):
    """
    Splitting the original geojson file 'sudan_grid.geojson' (file size around 2.4 Gb).
    The geojson file contains geo-information (e.g. top, left, bottom, right geo-coordinates) for all png tiles.
    After splitting, each geojson file will contain geo-information for only a single png tile.
    
    Params:
        geojson_dir (str): path to the original geojson file 'sudan_grid.geojson'
        output_folder (str): folder where geojson files for each png tile will be produced.
        
    Returns:
        None
    """
    
    os.makedirs(output_folder, exist_ok = True)

    data = gpd.read_file(geojson_dir)

    total_rows = len(data)
    crs = data.crs

    for idx in tqdm(range(total_rows)):
        row = list(data.loc[idx])
        file_id = str(row[0])

        save_dir = os.path.join(output_folder, file_id + '.geojson')
        gdf = gpd.GeoDataFrame([row], columns=['id', 'left', 'top', 'right', 'bottom', 'geometry'], crs=crs)
        gdf.to_file(save_dir, driver='GeoJSON')

        print(save_dir, '  -->   Done.')


def pix_to_geo_coords(affine_tfm, xmin, ymin, xmax, ymax):
    """
    Geo-reference a bounding box.
    
    Params:
        affine_tfm (affine.Affine): used for affine transformation
        xmin (float): x min value of bounding box
        ymin (float): y min value of bounding box
        xmax (float): x max value of bounding box
        ymax (float): y max value of bounding box
        
    Returns:
        geo_box (shapely.geometry.polygon.Polygon)
        
    """
    
    shapely_box = box(xmin, ymin, xmax, ymax)
    geo_box = shapely.affinity.affine_transform(shapely_box, 
                                                  [affine_tfm.a,
                                                   affine_tfm.b,
                                                   affine_tfm.d,
                                                   affine_tfm.e,
                                                   affine_tfm.xoff,
                                                   affine_tfm.yoff])
    
    return geo_box
