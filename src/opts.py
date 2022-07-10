import argparse

ROOT_PATH = "/Users/adam/github/xgboost_plus/"
IMAGE_PATH = "/Users/adam/github/xgboost_plus/data/raw/food101/images/"
META_PATH = "/Users/adam/github/xgboost_plus/data/raw/food101/metadata/"

def process_food_data():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default= ROOT_PATH,
                    help='path to data files')

    parser.add_argument('--img_path', type=str, default= IMAGE_PATH,
                    help='path to image folder')

    parser.add_argument('--meta_path', type=str, default= META_PATH,
                    help='path to metadata folder')
    args = parser.parse_args()
    return args