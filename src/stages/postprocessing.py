from argparse import Namespace
from src.geo_utils import produce_geo_files


def main(args: Namespace):
    produce_geo_files(model_output_folder = args.model_output,
                  geojson_folder = args.input,
                  output_folder = args.geojson_output)