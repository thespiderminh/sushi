import argparse
from copy import deepcopy
import json
import os
import os.path as osp
import numpy as np
import pandas as pd
import configparser
import cv2


# Detection and ground truth file formats for MOT17
DET_COL_NAMES = ('frame', 'id', 'name', 'truncation', 'occlusion', 'direction', 'bb_left', 'bb_top', 'bb_right', 'bb_bottom', 'x', 'y', 'z', 'xx', 'yy', 'zz', 'cam')
GT_COL_NAMES = ('0', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'frame_path', 'frame')

def get_refer_kitti_det_df(seq_name, data_root_path, config):
    """
    Load MOT detection file using detection files
    """
    det_file = config.det_file
    seq_path = osp.join(data_root_path, seq_name[-4:])  # Sequence path
    det_file_path = osp.join(osp.dirname(data_root_path), "label_02", seq_name[-4:] + ".txt")  # Detection file

    pd.options.mode.chained_assignment = None  # Tắt hoàn toàn cảnh báo

    # Đếm số frame
    num_frame = 0
    for root, dirs, files in os.walk(seq_path):
        for file in files:
            if (file[-4:] == ".png"):
                num_frame += 1

    det_df = pd.read_csv(det_file_path, sep=' ', header=None)
    det_df = det_df[det_df.columns[:len(DET_COL_NAMES)]]
    det_df.columns = DET_COL_NAMES

    det_df['frame_path'] = det_df['frame'].apply(lambda x: seq_path + "/" + str(x).zfill(6) + ".png")

    # Tạo cột width, height và conf
    det_df['bb_width'] = det_df['bb_right'] - det_df['bb_left']
    det_df['bb_height'] = det_df['bb_bottom'] - det_df['bb_top']
    det_df['conf'] = '1.000'

    # # Tinh chỉnh lại data
    # det_df['bb_left'] = det_df['bb_left'].round(1)
    # det_df['bb_top'] = det_df['bb_top'].round(1)
    # det_df['bb_width'] = det_df['bb_width'].round(1)
    # det_df['bb_height'] = det_df['bb_height'].round(1)

    # Get images' shape
    first_image_path = det_df.loc[0, 'frame_path']
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Build scene info dictionary
    seq_info_dict = {'seq': seq_name,
                     'seq_path': seq_path,
                     'det_file_name': det_file,
                     'frame_height': height,
                     'frame_width': width,
                     'seq_len': num_frame,
                     'fps': 30,
                     'has_gt': True,
                     'is_gt': False}
    
    # Get all text
    text_df = {}
    expression_path = osp.join(osp.dirname(osp.dirname(data_root_path)), "expression", seq_name[-4:])
    for filename in sorted(os.listdir(expression_path)):
        if filename.endswith(".json"):
            file_path = os.path.join(expression_path, filename)
            # Read each file into a list of rows
            with open(file_path, 'r') as file:
                data = json.load(file)        
                text_df[filename[:-5]] = data

    # return det_df, seq_info_dict
    return det_df, seq_info_dict, text_df


def get_refer_gt(seq_name, data_root_path, config):
    """
    Load MOT ground truth file
    """
    det_file = config.det_file
    seq_path = osp.join(data_root_path, seq_name[-4:])  # Sequence path
    gt_folder_path = osp.join(osp.dirname(osp.dirname(data_root_path)), "labels_with_ids", "image_02", seq_name[-4:])  # Detection file

    pd.options.mode.chained_assignment = None  # Tắt hoàn toàn cảnh báo

    # Đếm số frame
    num_frame = 0
    for root, dirs, files in os.walk(seq_path):
        for file in files:
            if (file[-4:] == ".png"):
                num_frame += 1

    gt_df = []
    num_of_frame_having_det = 0
    # Loop through each file in the folder
    for filename in sorted(os.listdir(gt_folder_path)):
        if filename.endswith(".txt") and filename.startswith("0"):
            file_path = os.path.join(gt_folder_path, filename)
            num_of_frame_having_det += 1
            
            # Read each file into a list of rows
            with open(file_path, 'r') as file:
                for line in file:
                    # Split the line by spaces to separate columns
                    row = line.strip().split()

                    # Convert each element to int or float if applicable
                    def convert_value(value):
                        if value.isdigit():
                            return int(value)
                        try:
                            return float(value)
                        except ValueError:
                            return value  # Keep as string if not a number

                    row = [convert_value(element) for element in row]

                    row = row + [osp.join(data_root_path, seq_name[-4:], filename[:-4] + ".png"), int(filename[:-4])]
                    
                    # Convert row data to the correct format and append to data list
                    gt_df.append(row)


    gt_df = pd.DataFrame(data=gt_df, columns=GT_COL_NAMES)

    # Tạo cột conf
    gt_df['conf'] = '1.000'

    # Get images' shape
    first_image_path = gt_df.loc[0, 'frame_path']
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    gt_df['bb_left'], gt_df['bb_width'] = gt_df['bb_left'] * width, gt_df['bb_width'] * width
    gt_df['bb_top'], gt_df['bb_height'] = gt_df['bb_top'] * height, gt_df['bb_height'] * height

    gt_df = gt_df.drop('0', axis=1)

    # Extra bbox values that will be used for id matching
    gt_df['bb_bot'] = (gt_df['bb_top'] + gt_df['bb_height']).values
    gt_df['bb_right'] = (gt_df['bb_left'] + gt_df['bb_width']).values

    # return gt_df
    return gt_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = parser.parse_args()
    config.det_file = 'det'
    a = get_refer_gt(seq_name="refer-0000", 
                            data_root_path="/data/hpc/ngocminh/SUSHI/datasets/KITTI/training/image_02", 
                            config=config)
    print("Refer")
    print(a[a['frame'] == 0][['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height']])
    a, b, text = get_refer_kitti_det_df(seq_name="refer-0000", 
                            data_root_path="/data/hpc/ngocminh/SUSHI/datasets/KITTI/training/image_02", 
                            config=config)
    print("Normal")
    print(a[a['frame'] == 0][['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height']])
    # pd.set_option('display.max_rows', None)
    # print(a[a['frame'] == 463])
    # print(b)