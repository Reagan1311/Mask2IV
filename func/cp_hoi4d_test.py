import pandas as pd
import shutil
import os
import cv2
import numpy as np
from os.path import join as opj

# Load the CSV file
data_path = '/train_data/HOI4D'
test_path = './exp_outputs/test_hoi4d_grasp'
frame_num = 16
os.makedirs(test_path, exist_ok=True)
df = pd.read_csv(opj(data_path, 'test_hoi4d.csv'))
height, width = 320, 512
original_height, original_width = 1080, 1920

# Iterate through each row and check the start and end frames
for index, row in df.iterrows():
    path = row['path']
    object, action = row['object'], row['action']
    start_frame, end_frame = row['start_frame'], row['end_frame']
    video_path = os.path.join(data_path, 'HOI4D_release', path, 'align_rgb')
    mask_path = os.path.join(data_path, 'HOI4D_annotations', path, '2Dseg/shift_mask')
    if not os.path.exists(mask_path):
        mask_path = mask_path.replace('shift_mask', 'mask')
    
    path = path.replace('/', '_')

    # # Copy the start and end frames
    if action.lower() == 'grasp':
        frame = opj(video_path, f'{start_frame:05d}.jpg')
        mask = opj(mask_path, f'{start_frame:05d}.png')
        frame_name = '{}_{}_{}_{}_{}.jpg'.format(path, start_frame, end_frame, object, action)
        mask_name = '{}_{}_{}_{}_{}.png'.format(path, start_frame, end_frame, object, action)
        shutil.copy(frame, opj(test_path, frame_name))
        shutil.copy(mask, opj(test_path, mask_name))
