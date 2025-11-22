import os
import cv2
import csv
import json
from os.path import join as opj
import numpy as np
from tqdm import tqdm


obj_mapping = [
    '', 'ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle',
    'Safe', 'Bowl', 'Bucket', 'Scissors', '', 'Pliers', 'Kettle',
    'Knife', 'TrashCan', '', '', 'Lamp', 'Stapler', '', 'Chair'
]


filter_action = ['rest', 'Stop', 'Reachout', 'carry', 'go', 'Press', 'dump', 'cut', 'paper-cut']
filter_obj = ['StorageFurniture', 'Safe', 'Lamp', 'Stapler']

outlier = []

def generate_csv(data_list, data_path, filter_len=12, sample_len=16):
    head = [['path', 'id', 'duration', 'actual_duration', 'start_frame', 'end_frame', 'object', 'action', 'motion_score']]
    with open(data_list, 'r') as df:
        num_lines = sum(1 for line in df)
    
    with open(data_list, 'r') as df:
        for d in tqdm(df, total=num_lines):
            d = d.rstrip('\n')

            # Check if hands exist in the mask (only videos of chairs occassionally have no hands)
            obj_idx = int(d.split('/')[2][1:])
            obj = obj_mapping[obj_idx]
            
            # filter 
            if obj in filter_obj:
                continue
            
            frame_path = opj(data_path, 'HOI4D_release/{}/align_rgb').format(d)
            anno_path = opj(data_path, 'HOI4D_annotations/{}/action/color.json').format(d)
            mask_path = opj(data_path, 'HOI4D_annotations/{}/2Dseg/shift_mask').format(d)
            if not os.path.exists(mask_path):
                mask_path = mask_path.replace('shift_mask', 'mask')
                if not os.path.exists(mask_path):
                    # only one video has no mask annotations ['ZY20210800004/H4/C5/N15/S56/s02/T1']
                    continue

            with open(anno_path, 'r') as file:
                anno_dir = json.load(file)
            
            actual_dura = len(os.listdir(frame_path)) - 1
            mark_event = False
            if 'events' not in anno_dir.keys():
                outlier.append(d)
                events = anno_dir['markResult']['marks']
                duration = anno_dir['info']['Duration']
                mark_event = True
            else:
                events = anno_dir['events']
                duration = anno_dir['info']['duration']
                # import pdb; pdb.set_trace()
            
            for i, e in enumerate(events):
                action = e['event']
                id = e['id']
                if action in filter_action:
                    continue
                
                # Merge reachout and grasp to one action
                if mark_event:
                    if action == "Grasp" and events[i-1]['event'] == "Reachout":
                        start = (events[i-1]['hdTimeStart'] / duration) * actual_dura
                    else:
                        start = (e['hdTimeStart'] / duration) * actual_dura
                    # start = (e['hdTimeStart'] / duration) * actual_dura
                    end = (e['hdTimeEnd'] / duration) * actual_dura
                else:
                    if action == "Grasp" and events[i-1]['event'] == "Reachout":
                        start = (events[i-1]['startTime'] / duration) * actual_dura
                    else:
                        start = (e['startTime'] / duration) * actual_dura
                    # start = (e['startTime'] / duration) * actual_dura
                    end = (e['endTime'] / duration ) * actual_dura
                
                end = int(min(end, 299))
                start = int(start)

                # Check if frame length
                frame_length = end - start + 1
                
                assert filter_len < sample_len, "Filter length should be less than sample length"

                if frame_length < filter_len:
                    continue
                elif frame_length < sample_len:
                    ext_end = end + (sample_len - frame_length)
                    if ext_end > 299:
                        start = max(0, start - (sample_len - frame_length))
                    else:
                        end = ext_end

                # in rgb format
                rhand_arr = np.array([0, 128, 0])
                if obj == 'Chair':
                    lhand_arr = np.array([128, 128, 0])
                else:
                    lhand_arr = np.array([0, 128, 128])

                # frame_indices = list(range(start, end+1))
                frame_indices = np.round(np.linspace(start, end, sample_len)).astype(int)
                hand_flag, motion_score = hands_ms(mask_path, frame_indices, [lhand_arr, rhand_arr], compute_motion=False)
                if not hand_flag:
                    print('No hands in {}'.format(d))
                    continue
                    
                temp_data = [d, id, duration, actual_dura, start, end, obj, action, motion_score]
                
                head.append(temp_data)
    
    return head


def hands_ms(mask_dir, frame_indices, hands_arrs, compute_motion=True):
    lh_arr, rh_arr = hands_arrs
    hand_exists = False
    hand_masks = []

    for i in frame_indices:
        mask_path = os.path.join(mask_dir, f'{i:05d}.png')  # Adjust the naming format as needed
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
        if mask is None:
            raise Exception(f"Error: Could not read file - {mask_path}")
        
        masks_flatten = mask.reshape(-1, 3)
        hand_matches = (masks_flatten == rh_arr).all(axis=-1) | (masks_flatten == lh_arr).all(axis=-1)
        hand_masks.append(hand_matches)

        rh_exists = (rh_arr == masks_flatten).all(axis=-1).any()
        lh_exists = (lh_arr == masks_flatten).all(axis=-1).any()

        if rh_exists or lh_exists:
            hand_exists = True
            if not compute_motion:
                break
    
    if compute_motion:
        diffs = np.abs(np.diff(hand_masks, axis=0))
        motion_score = np.mean(np.sum(diffs, axis=1))
    else:
        motion_score = 0

    return hand_exists, motion_score


if __name__ == '__main__':
    data_list = '/train_data/HOI4D/HOI4D-Instructions/release.txt'
    HOI4D_path = '/train_data/HOI4D'
    data = generate_csv(data_list, HOI4D_path)
    
    print(len(data), len(outlier))
    
    with open("./exp_outputs/data_hoi4d_update.csv", mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write each row of data
        for row in data:
            writer.writerow(row)
