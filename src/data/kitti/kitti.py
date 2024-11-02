import argparse
import os
import os.path as osp
import pandas as pd
import configparser
import cv2


# Detection and ground truth file formats for MOT17
DET_COL_NAMES = ('frame', 'id', 'name', 'use1?', 'use2?', 'direction', 'bb_left', 'bb_top', 'bb_right', 'bb_bottom', 'x', 'y', 'z', 'xx', 'yy', 'zz', 'cam')


def get_kitti_det_df(seq_name, data_root_path, config):
    """
    Load MOT detection file using detection files
    """
    det_file = config.det_file
    seq_path = osp.join(data_root_path, seq_name)  # Sequence path
    det_file_path = osp.join(osp.dirname(data_root_path), "label_02", seq_name + ".txt")  # Detection file

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

    # Coordinates are 1 based
    det_df['bb_left'] -= 1
    det_df['bb_top'] -= 1

    det_df['frame_path'] = det_df['frame'].apply(lambda x: seq_path + "/" + str(x).zfill(6) + ".png")

    # Tạo cột width, height và conf
    det_df['bb_width'] = det_df['bb_right'] - det_df['bb_left']
    det_df['bb_height'] = det_df['bb_bottom'] - det_df['bb_top']
    det_df['conf'] = '1.000'

    # Tinh chỉnh lại data
    det_df['bb_left'] = det_df['bb_left'].round(1)
    det_df['bb_top'] = det_df['bb_top'].round(1)
    det_df['bb_width'] = det_df['bb_width'].round(1)
    det_df['bb_height'] = det_df['bb_height'].round(1)

    # Delete unused columns
    det_df = det_df.drop(columns=['use1?', 'use2?', 'direction', 'bb_right', 'bb_bottom', 'x', 'y', 'z', 'xx', 'yy', 'zz', 'cam'])

    # Get images' shape
    first_image_path = det_df.loc[0, 'frame_path']
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Detele unused detetions
    det_df = det_df[det_df['name'] != 'DontCare']

    # Build scene info dictionary
    seq_info_dict = {'seq': seq_name,
                     'seq_path': seq_path,
                     'det_file_name': det_file,
                     'frame_height': height,
                     'frame_width': width,
                     'seq_len': num_frame,
                     'fps': 30,
                     'has_gt': osp.exists(osp.join(seq_path, 'gt')),
                     'is_gt': False}

    return det_df, seq_info_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = parser.parse_args()
    config.det_file = 'det'
    a, b = get_kitti_det_df(seq_name="0004", 
                            data_root_path="/data/hpc/ngocminh/SUSHI/datasets/KITTI/training/image_02", 
                            config=config)
    pd.set_option('display.max_rows', None)
    print(a[:][:40])
    print(b)