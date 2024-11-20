import argparse
import os
import os.path as osp
import pandas as pd
import configparser
import cv2


# Detection and ground truth file formats for MOT17
DET_COL_NAMES = ('0', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'frame_path', 'frame')


def get_refer_kitti_det_df(seq_name, data_root_path, config):
    """
    Load MOT detection file using detection files
    """
    det_file = config.det_file
    seq_path = osp.join(data_root_path, seq_name[-4:])  # Sequence path
    det_folder_path = osp.join(osp.dirname(osp.dirname(data_root_path)), "labels_with_ids", "image_02", seq_name[-4:])  # Detection file

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
    for filename in sorted(os.listdir(det_folder_path)):
        if filename.endswith(".txt") and filename.startswith("0"):
            file_path = os.path.join(det_folder_path, filename)
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
                    det_df.append(row)


    det_df = pd.DataFrame(data=det_df, columns=DET_COL_NAMES)

    # Tạo cột conf
    det_df['conf'] = '1.000'

    # Get images' shape
    first_image_path = det_df.loc[0, 'frame_path']
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    det_df['bb_left'], det_df['bb_width'] = det_df['bb_left'] * width, det_df['bb_width'] * width
    det_df['bb_top'], det_df['bb_height'] = det_df['bb_top'] * height, det_df['bb_height'] * height

    det_df = det_df.drop('0', axis=1)

    # Build scene info dictionary
    seq_info_dict = {'seq': seq_name,
                     'seq_path': seq_path,
                     'det_file_name': det_file,
                     'frame_height': height,
                     'frame_width': width,
                     'seq_len': num_frame,
                     'num_of_frame_having_det': num_of_frame_having_det,
                     'fps': 30,
                     'has_gt': osp.exists(osp.join(seq_path, 'gt')),
                     'is_gt': False}

    return det_df, seq_info_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = parser.parse_args()
    config.det_file = 'det'
    a, b = get_refer_kitti_det_df(seq_name="refer-0020", 
                            data_root_path="/data/hpc/ngocminh/SUSHI/datasets/KITTI/training/image_02", 
                            config=config)
    pd.set_option('display.max_rows', None)
    print(a[a['frame'] == 463])
    print(b)