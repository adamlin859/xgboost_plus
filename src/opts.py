import argparse

ROOT_PATH = "/Users/adam/github/xgboost_plus/"


def process_food_data():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default= ROOT_PATH,
                    help='path to data files')

    args = parser.parse_args()
    return args