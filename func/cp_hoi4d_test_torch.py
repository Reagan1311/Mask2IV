import pandas as pd
import shutil
import os
import cv2
import numpy as np

import torch
from os.path import join as opj
import torchvision.io as io
import torchvision.transforms as transforms

# Load the CSV file
data_path = '/train_data/HOI4D'
test_path = './exp_outputs/GT'
frame_num = 16
os.makedirs(test_path, exist_ok=True)
df = pd.read_csv(opj(data_path, 'HOI4D-Instructions', 'train_hoi4d_test.csv'))
video_size = (320, 512)

transform = transforms.Compose([
        transforms.Resize(min(video_size)),
        transforms.CenterCrop(video_size)])

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

    # # Copy the whole clip as mp4.file
    frame_indices = np.round(np.linspace(start_frame, end_frame, frame_num)).astype(int)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    video_name = '{}_{}_{}_{}_{}.mp4'.format(path, start_frame, end_frame, object, action)
    mask_video_name = '{}_{}_{}_{}_{}_mask.mp4'.format(path, start_frame, end_frame, object, action)
    video_name = opj(test_path, video_name)
    mask_video_name = opj(test_path, mask_video_name)

    frames, masks = [], []
    for f in frame_indices:
        # os.makedirs(opj(test_path, path), exist_ok=True)
        frame = opj(video_path, f'{f:05d}.jpg')
        mask = opj(mask_path, f'{f:05d}.png')
        frame = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2RGB)
        frame_tensor = transform(torch.from_numpy(frame).permute(2, 0, 1))
        mask_tensor = transform(torch.from_numpy(mask).permute(2, 0, 1))

        frames.append(frame_tensor)
        masks.append(mask_tensor)
    
    frames = torch.stack(frames).permute(0, 2, 3, 1)
    masks = torch.stack(masks).permute(0, 2, 3, 1)

    io.write_video(video_name, frames, fps=8, video_codec='h264', options={'crf': '10'})
    io.write_video(mask_video_name, masks, fps=8, video_codec='h264', options={'crf': '10'})
    