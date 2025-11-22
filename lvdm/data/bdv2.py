import os
import random
from tqdm import tqdm
import pandas as pd
from decord import VideoReader, cpu
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2

import torchvision.transforms.functional as F

class BDV2(Dataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def __init__(self,
                 meta_path,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 random_dilation_erosion=False,
                 dilation_erosion_rate=0,
                 first_mask_path=None,
                 ):
        self.meta_path = meta_path
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self.first_mask_path = first_mask_path
        self.random_dilation_erosion = random_dilation_erosion
        self.dilation_erosion_rate = dilation_erosion_rate
        self._load_metadata()
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                    ])            
            elif spatial_transform == "resize_center_crop":
                # assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = v2.Compose([
                    transforms.Resize(min(self.resolution), antialias=True),
                    transforms.CenterCrop(self.resolution),
                    ])
            elif spatial_transform == "resize":
                self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None
            
        self.spatial_transform_mask = v2.Compose([
                    transforms.Resize(min(self.resolution), antialias=True, 
                                      interpolation=transforms.InterpolationMode.NEAREST),
                    transforms.CenterCrop(self.resolution),
                    ])

    def _load_metadata(self):
        # df = pd.read_csv('/workspace/exp_outputs/train_bdv2.csv')   # 18308
        metadata = pd.read_csv(self.meta_path)
        # metadata = df[(df['duration'] <= 50) & (df['confidence'] >= 0.4) & (~df['object'].isnull())] # 15140
        
        # filter the path with valid masks folder
        metadata = metadata[metadata['path'].apply(lambda p: os.path.exists(os.path.join(p, 'masks')))]  # 13781
        metadata = metadata[metadata['path'].apply(lambda p: len(os.listdir(os.path.join(p, 'masks'))) > 0 )]

        print(f'>>> {len(metadata)} data samples loaded.')
        
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)

        self.metadata = metadata
        # self.metadata.dropna(inplace=True)
    
    def read_frames(self, frame_dir, mask_dir, frame_indices, raw_res=True, first_mask=False):
        frames, masks = [], []
        for idx, i in enumerate(frame_indices):
            frame_path = os.path.join(frame_dir, f'im_{i:d}.jpg')  # Adjust the naming format as needed
            mask_path = os.path.join(mask_dir, f'{idx}.png') if first_mask else os.path.join(mask_dir, f'mask_{i:d}.png')
            frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

            if not raw_res:
                frame = cv2.resize(frame, (300, 350))
                mask = cv2.resize(mask, (300, 350))
            if frame is None or mask is None:
                print(f"Error: Could not read frame {frame_path}")
                continue
            
            # if np.all(mask == [0, 0, 0]):
            #     raise ValueError(f"Something is wrong with the annotations in {mask_path}, there are no hands or objects in the mask")

            frames.append(frame)
            masks.append(mask)
        
        masks = np.array(masks)
        if first_mask:
            masks = np.where(masks >= 128, 255, 0)

        return np.array(frames), masks
    
    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride

        ## get frames until success
        while True:
            # print('========================')
            index = index % len(self.metadata)
            sample = self.metadata.iloc[index]
            video_path = os.path.join(sample['path'], 'images0')
            if self.first_mask_path:
                mask_path = os.path.join(self.first_mask_path, 'val', str(index))
            else:
                mask_path = os.path.join(sample['path'], 'masks')
            
            caption = sample['caption'].lower()
            caption = "a robot gripper " + caption
            object = str(sample['object']).lower()

            frame_num = len(os.listdir(video_path))

            if frame_num < self.video_length:
                print(f"video ({video_path}) length ({frame_num}) is smaller than target length({self.video_length})")
                index += 1
                continue
            
            fps_ori = 15
            frame_stride = frame_num // self.video_length
            # start_idx = random.randint(0, 3) if frame_num >= self.video_length + 3 else random.randint(0, frame_num - self.video_length)

            ## calculate frame indices
            frame_indices = np.round(np.linspace(0, frame_num-1, self.video_length)).astype(int)

            try:
                
                frames, masks = self.read_frames(video_path, mask_path, frame_indices, 
                                                    raw_res=self.load_raw_resolution,
                                                    first_mask=self.first_mask_path is not None)
                
                if len(frames) == 0 or len(masks) == 0:
                    raise ValueError("Length of frames or masks is 0")
                
                if frames.shape[0] != self.video_length:
                    raise ValueError(f"Length of frames is {frames.shape[0]}, which does not match the target length {self.video_length}")
                
                ## process data
                assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
                frames = torch.tensor(frames).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
                masks = torch.tensor(masks).permute(3, 0, 1, 2).float()
                if self.first_mask_path is None:
                    masks = format_mask(masks)

                # divide the mask into 2 parts: hand and object
                masks_f = masks.flatten(1)
                bg_tensor = torch.tensor([0, 0, 0]).unsqueeze(-1)
                bg_matches = (masks_f == bg_tensor).all(dim=0)
                
                # Use mask_attn in either 1 or 2
                
                # 1. get a mask attn with foreground only
                mask_attn = ~bg_matches
                mask_attn = mask_attn.reshape(masks.shape[1:]).unsqueeze(0)

                break

            except KeyboardInterrupt:
                print("Interrupted by user. Exiting...")
                break  # Exit the loop when Ctrl + C is pressed
            
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue
        
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            if not self.first_mask_path:
                masks = self.spatial_transform_mask(masks)

            mask_attn = self.spatial_transform_mask(mask_attn)

            if self.random_dilation_erosion:
                if random.random() < self.dilation_erosion_rate:
                    C, N, _, _ = masks.shape  # (3, 16, H, W)
                    mask_np = masks.numpy().astype(np.uint8)
                    operation = random.choice(["dilate", "erode"])
                    kernel_size = random.choice([3, 5, 7])
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    
                    for c in range(C):
                        for n in range(N):
                            if operation == "dilate":
                                mask_np[c, n] = cv2.dilate(mask_np[c, n], kernel, iterations=1)
                            else:
                                mask_np[c, n] = cv2.erode(mask_np[c, n], kernel, iterations=1)
                    masks = torch.from_numpy(mask_np).float()
        
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        
        ## turn frames tensors to [-1,1]
        frames = (frames / 255 - 0.5) * 2
        masks = (masks / 255 - 0.5) * 2
        
        mask_copy = masks.clone().flatten(1)
        hand_condition = (mask_copy[0] == -1) & (mask_copy[1] == 1) & (mask_copy[2] == -1)
        obj_condition = (mask_copy[0] == 1) & (mask_copy[1] == -1) & (mask_copy[2] == -1)
    
        hand_masks = hand_condition.reshape(masks.shape[1:]).numpy().astype(np.uint8)  # t x h x w
        obj_masks = obj_condition.reshape(masks.shape[1:]).numpy().astype(np.uint8)

        contact_maps = np.zeros_like(hand_masks, dtype=np.uint8)  # t x h x w
        kernel = np.ones((7, 7), np.uint8)
        for i in range(contact_maps.shape[0]):
            dilate_hand_masks = cv2.dilate(hand_masks[i], kernel, iterations=1)
            dilate_obj_masks = cv2.dilate(obj_masks[i], kernel, iterations=1)
            contact_maps[i] = (hand_masks[i] & dilate_obj_masks) | (obj_masks[i] & dilate_hand_masks)
        
        contact_maps = (hand_masks | obj_masks).astype(np.uint8)

        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max

        data = {'video': frames, 'mask': masks, 'obj_mask': obj_masks, 'object': object, 'caption': caption, 
                'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask_attn': mask_attn, 'contact_maps': contact_maps}
        return data
    
    def __len__(self):
        return len(self.metadata)


def format_mask(masks):
    mask_f = masks.flatten(1)
    condition = (mask_f[0] == 255) & (mask_f[1] == 255) & (mask_f[2] == 255)
    mask_f[:, condition] = torch.tensor([[255], [0], [0]], dtype=mask_f.dtype)
    
    condition = (mask_f[0] == 128) & (mask_f[1] == 128) & (mask_f[2] == 128)
    mask_f[:, condition] = torch.tensor([[0], [255], [0]], dtype=mask_f.dtype)
    
    masks = mask_f.reshape(masks.shape)
    return masks
