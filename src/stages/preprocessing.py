from argparse import Namespace
from src.geo_utils import split_geojsons


def main(args: Namespace):
    split_geojsons(geojson_dir=args.input, output_folder=args.output)