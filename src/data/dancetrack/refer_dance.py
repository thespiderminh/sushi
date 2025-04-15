import argparse
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

def get_refer_dancetrack_det_df_from_det(seq_name, data_root_path, config):
    """
    Load MOT detection file using detection files
    """
    det_file = config.det_file
    seq_path = osp.join(data_root_path, seq_name[6:])  # Sequence path
    gt_folder_path = osp.join(osp.dirname(osp.dirname(data_root_path)), "labels_with_ids", "image_02", seq_name[6:])  # Detection file

    pd.options.mode.chained_assignment = None  # Tắt hoàn toàn cảnh báo

    # Đếm số frame
    num_frame = 0
    for root, dirs, files in os.walk(seq_path):
        for file in files:
            if (file[-4:] == ".png"):
                num_frame += 1

    det_df = []
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

                    row = row + [osp.join(data_root_path, seq_name[6:], filename[:-4] + ".png"), int(filename[:-4])]
                    
                    # Convert row data to the correct format and append to data list
                    det_df.append(row)


    det_df = pd.DataFrame(data=det_df, columns=GT_COL_NAMES)

    # Tạo cột conf
    det_df['conf'] = '1.000'

    # Get images' shape
    first_image_path = det_df.loc[0, 'frame_path']
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    det_df['bb_left'], det_df['bb_width'] = det_df['bb_left'] * width, det_df['bb_width'] * width
    det_df['bb_top'], det_df['bb_height'] = det_df['bb_top'] * height, det_df['bb_height'] * height

    det_df = det_df.drop('0', axis=1)

    # Extra bbox values that will be used for id matching
    det_df['bb_bot'] = (det_df['bb_top'] + det_df['bb_height']).values
    det_df['bb_right'] = (det_df['bb_left'] + det_df['bb_width']).values


    # # Tinh chỉnh lại data
    # det_df['bb_left'] = det_df['bb_left'].round(1)
    # det_df['bb_top'] = det_df['bb_top'].round(1)
    # det_df['bb_width'] = det_df['bb_width'].round(1)
    # det_df['bb_height'] = det_df['bb_height'].round(1)

    # Get images' shape
    first_image_path = det_df.loc[0, 'frame_path']
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape
    
    # Get all text
    text_df = {}
    total_valid_det = 0
    num_text = 0
    expression_path = osp.join(osp.dirname(osp.dirname(osp.dirname(data_root_path))), "expression", seq_name[6:])
    for filename in sorted(os.listdir(expression_path)):
        if filename.endswith(".json"):
            file_path = os.path.join(expression_path, filename)
            # Read each file into a list of rows
            with open(file_path, 'r') as file:
                data = json.load(file)
                for key, value in data['label'].items():
                    total_valid_det += len(value)
                num_text += 1
                text_df[filename[:-5]] = data

    # Build scene info dictionary
    seq_info_dict = {'seq': seq_name,
                     'seq_path': seq_path,
                     'det_file_name': det_file,
                     'frame_height': height,
                     'frame_width': width,
                     'seq_len': num_frame,
                     'total_det': len(det_df),
                     'total_valid_det': total_valid_det,
                     'num_text': num_text,
                     'fps': 30,
                     'has_gt': False,
                     'is_gt': False}

    # return det_df, seq_info_dict
    return det_df, seq_info_dict, text_df


def get_refer_dancetrack_gt(seq_name, data_root_path, config):
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
    # a = get_refer_gt(seq_name="refer-0000", 
    #                         data_root_path="/data/hpc/ngocminh/SUSHI/datasets/KITTI/training/image_02", 
    #                         config=config)
    # print("Refer")
    # print(a[a['frame'] == 0][['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height']])
    det_df, seq_info_dict, text_df = get_refer_dancetrack_det_df_from_det(seq_name="refer-dancetrack0044", 
                            data_root_path="/data/hpc/ngocminh/SUSHI/datasets/REFER-DANCE/DanceTrack/training/image_02", 
                            config=config)
    print(det_df)
    print(seq_info_dict)
    print(text_df.keys())
    # pd.set_option('display.max_rows', None)
    # print(a[a['frame'] == 463])