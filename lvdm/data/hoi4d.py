import os
import random
from tqdm import tqdm
import pandas as pd
from decord import VideoReader, cpu
import cv2
import numpy as np
from scipy.ndimage import label

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2

import torchvision.transforms.functional as F
# 17 objects & 14 actions in HOI4D
OBJECTS = ['ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle', 'Safe', 'Bowl', 'Bucket', 
           'Scissors', 'Pliers', 'Kettle', 'Knife', 'TrashCan', 'Lamp', 'Stapler', 'Chair']
ACTIONS = ['Grasp', 'Pickup', 'putdown', 'dump', 'open', 'close', 'carry', 'Carrywithbothhands', 
           'binding', 'turn&on&the&switch', 'Press', 'paper-cut', 'push', 'pull', 'cut']

class HOI4D(Dataset):
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
                 data_dir,
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
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self.random_dilation_erosion = random_dilation_erosion
        self.dilation_erosion_rate = dilation_erosion_rate
        self._load_metadata()
        self.spatial_transform = spatial_transform
        self.first_mask_path = first_mask_path
        
        self.object_transform = transforms.Compose([
            transforms.Resize((320, 512), antialias=True),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(30),
        ])
        
    def bottom_aligned_center_crop(self, image: torch.Tensor, crop_height: int, crop_width: int):
        """
        Crop the image in the center horizontally but aligned to the bottom.

        Args:
            image (torch.Tensor): Input image tensor of shape (C, H, W).
            crop_height (int): Desired height of the cropped image.
            crop_width (int): Desired width of the cropped image.

        Returns:
            torch.Tensor: Cropped image.
        """
        _, _, H, W = image.shape  # Get original height and width
        
        top = (H - crop_height) // 4 * 3  # Align bottom of the crop with image bottom
        left = (W - crop_width) // 2  # Center horizontally

        return F.crop(image, top, left, crop_height, crop_width)

    def _load_metadata(self):
        df = pd.read_csv(self.meta_path, dtype=str) # 20010
        metadata = df.loc[(df['action'] != 'carry') & (~df['object'].isin(['StorageFurniture', 'Safe', 'Lamp', 'Stapler']))] # 12535

        print(f'>>> {len(metadata)} data samples loaded.')
        
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)

        self.metadata = metadata
        self.metadata.dropna(inplace=True)
    
    def read_frames(self, frame_dir, mask_dir, frame_indices, raw_res=True, first_mask=False):
        frames, masks = [], []
        for idx, i in enumerate(frame_indices):
            frame_path = os.path.join(frame_dir, f'{i:05d}.jpg')  # Adjust the naming format as needed
            mask_path = os.path.join(mask_dir, f'{idx}.png') if first_mask else os.path.join(mask_dir, f'{i:05d}.png')
            frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

            if not raw_res:
                frame = cv2.resize(frame, (300, 350))
                mask = cv2.resize(mask, (300, 350))
            if frame is None or mask is None:
                print(f"Error: Could not read frame {frame_path}")
                continue
            
            if np.all(mask == [0, 0, 0]):
                raise ValueError(f"Something is wrong with the annotations in {mask_path}, there are no hands & objects in the mask")

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

            index = index % len(self.metadata)
            sample = self.metadata.iloc[index]
            video_path = os.path.join(self.data_dir, 'HOI4D_release', sample['path'], 'align_rgb')
            if self.first_mask_path:
                mask_path = os.path.join(self.first_mask_path, 'val', str(index))
            else:
                mask_path = os.path.join(self.data_dir, 'HOI4D_annotations', sample['path'], '2Dseg/shift_mask')

            object, action = sample['object'].lower(), format_action(sample['action'])
            start_frame, end_frame = int(sample['start_frame']), int(sample['end_frame'])
            frame_num = (end_frame - start_frame) + 1

            if frame_num < self.video_length:
                print(f"video ({action, object}) length ({frame_num}) is smaller than target length({self.video_length})")
                index += 1
                continue
            
            fps_ori = 15
            frame_stride = frame_num // self.video_length
            frame_indices = np.round(np.linspace(start_frame, end_frame, self.video_length)).astype(int)
            
            try:
                if not os.path.exists(mask_path):
                    mask_path = mask_path.replace('shift_mask', 'mask')
                frames, masks = self.read_frames(video_path, mask_path, frame_indices, 
                                                    raw_res=self.load_raw_resolution, first_mask=self.first_mask_path is not None)
                
                if len(frames) == 0 or len(masks) == 0:
                    raise ValueError("Length of frames or masks is 0")
                
                if frames.shape[0] != self.video_length:
                    raise ValueError(f"Length of frames is {frames.shape[0]}, which does not match the target length {self.video_length}")
                
                ## process data
                assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
                frames = torch.tensor(frames).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
                masks = torch.tensor(masks).permute(3, 0, 1, 2).float()
                
                # divide the mask into 2 parts: hand and object
                masks_f = masks.flatten(1)
                bg_tensor = torch.tensor([0, 0, 0]).unsqueeze(-1)
                bg_matches = (masks_f == bg_tensor).all(dim=0)
                
                # Use mask_attn in either 1 or 2
                mask_attn = ~bg_matches
                mask_attn = mask_attn.reshape(masks.shape[1:]).unsqueeze(0)
                
                if self.first_mask_path is None:
                    masks = format_mask(masks, object)
                caption = format_template(action, object, masks_f)
                
                hand_tensor = torch.tensor([0, 255, 0]).unsqueeze(-1)
                hand_matches = (masks_f == hand_tensor).all(dim=0)
                
                obj_part0_tensor = torch.tensor([255, 0, 0]).unsqueeze(-1)
                obj_part1_tensor = torch.tensor([0, 0, 255]).unsqueeze(-1)
                obj_matches = (masks_f == obj_part0_tensor).all(dim=0) | (masks_f == obj_part1_tensor).all(dim=0)
                
                mask_attn = torch.zeros_like(bg_matches).int()
                mask_attn[hand_matches == True] = 1
                mask_attn[obj_matches == True] = 2
                mask_attn = mask_attn.reshape(masks.shape[1:]).unsqueeze(0)

                hand_mask = hand_matches.reshape(masks.shape[1:]).unsqueeze(0)
                obj_mask = obj_matches.reshape(masks.shape[1:]).unsqueeze(0)    # 1 x t x h x w
                
                if self.first_mask_path is not None:
                    obj_mask = F.resize(obj_mask, (frames.shape[2], frames.shape[3]), 
                                        interpolation=transforms.InterpolationMode.NEAREST) 

                # randomly choose one object image as condition image and apply augmentation 
                if (obj_mask == False).all():
                    print('================' + video_path + str(start_frame) + '-' + str(end_frame))
                    index += 1
                    continue
                
                obj_crops = []
                for num_f, obj_mask_f in enumerate(obj_mask[0]):
                    nz_inds = torch.nonzero(obj_mask_f, as_tuple=False)
                    if nz_inds.shape[0] == 0:
                        obj_crops.append(torch.zeros((3, 224, 224), dtype=frames.dtype))  # 3 x 224 x 224
                    
                    else:
                        y_min, x_min = torch.min(nz_inds, dim=0).values
                        y_max, x_max = torch.max(nz_inds, dim=0).values
                        obj_crop = transforms.functional.crop(frames[:, num_f] * obj_mask_f, 
                                                            y_min, x_min, y_max-y_min, x_max-x_min)
                        obj_crops.append(self.object_transform(obj_crop))
                obj_crops = torch.stack(obj_crops).transpose(0, 1)  # 3 x 16 x 224 x 224
                         
                
                if self.spatial_transform is not None:

                    frames = F.resize(frames, int(min(self.resolution) * 1.2), interpolation=transforms.InterpolationMode.BILINEAR)
                    if self.first_mask_path is None:
                        masks = F.resize(masks, int(min(self.resolution) * 1.2), interpolation=transforms.InterpolationMode.NEAREST)  # Use NEAREST for masks
                    obj_mask = F.resize(obj_mask, int(min(self.resolution) * 1.2), interpolation=transforms.InterpolationMode.NEAREST)
                    hand_mask = F.resize(hand_mask, int(min(self.resolution) * 1.2), interpolation=transforms.InterpolationMode.NEAREST)

                    mask_attn = F.resize(mask_attn, int(min(self.resolution) * 1.2), interpolation=transforms.InterpolationMode.NEAREST)
                    
                    # Generate random crop parameters
                    if self.spatial_transform == 'resize_random_crop':

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

                        max_tries = 10
                        for idx in range(max_tries):
                            i, j, h, w = transforms.RandomCrop.get_params(frames, output_size=self.resolution)

                            # Apply the same crop to both image and mask
                            obj_mask_crop = F.crop(obj_mask, i, j, h, w)
                            hand_mask_crop = F.crop(hand_mask, i, j, h, w)
                            if (obj_mask_crop == 0).all() or (hand_mask_crop == 0).all():
                                continue
                            
                            # Define the borders
                            top_border = obj_mask_crop[:, :, 0, :].flatten(1)
                            bottom_border = obj_mask_crop[:, :, -1, :].flatten(1)
                            left_border = obj_mask_crop[:, :, :, 0].flatten(1)
                            right_border = obj_mask_crop[:, :, :, -1].flatten(1)

                            # Check if any pixel in the border is not [0, 0, 0]
                            if object == 'chair':
                                border_allzero = (left_border == 0).all() & (right_border == 0).all()
                            else:
                                border_allzero = (top_border == 0).all() & (bottom_border == 0).all() & (left_border == 0).all() & (right_border == 0).all()

                            if border_allzero:
                                frames = F.crop(frames, i, j, h, w)
                                masks = F.crop(masks, i, j, h, w)
                                mask_attn = F.crop(mask_attn, i, j, h, w)
                                break
                    
                        frames = F.center_crop(frames, self.resolution)
                        masks = F.center_crop(masks, self.resolution)
                        mask_attn = F.center_crop(mask_attn, self.resolution)    
                        
                    elif self.spatial_transform == 'resize_center_crop':
                        frames = F.center_crop(frames, self.resolution)
                        masks = F.center_crop(masks, self.resolution)
                        mask_attn = F.center_crop(mask_attn, self.resolution)

                    else:
                        raise NotImplementedError(f"{self.spatial_transform} is not supported")  
                        
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
        
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        
        ## turn frames tensors to [-1,1]
        frames = (frames / 255 - 0.5) * 2
        masks = (masks / 255 - 0.5) * 2
            
        mask_copy = masks.clone().flatten(1)
        hand_condition = (mask_copy[0] == -1) & (mask_copy[1] == 1) & (mask_copy[2] == -1)
        obj_condition_0 = (mask_copy[0] == 1) & (mask_copy[1] == -1) & (mask_copy[2] == -1)
        obj_condition_1 = (mask_copy[0] == -1) & (mask_copy[1] == -1) & (mask_copy[2] == 1)
        obj_condition = obj_condition_0 | obj_condition_1
    
        hand_masks = hand_condition.reshape(masks.shape[1:]).numpy().astype(np.uint8)  # t x h x w
        obj_masks = obj_condition.reshape(masks.shape[1:]).numpy().astype(np.uint8)
        
        contact_maps = np.zeros_like(hand_masks, dtype=np.uint8)  # t x h x w
        kernel = np.ones((7, 7), np.uint8)
        for i in range(contact_maps.shape[0]):
            dilate_hand_masks = cv2.dilate(hand_masks[i], kernel, iterations=1)
            dilate_obj_masks = cv2.dilate(obj_masks[i], kernel, iterations=1)
            contact_maps[i] = (hand_masks[i] & dilate_obj_masks) | (obj_masks[i] & dilate_hand_masks)
        
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max

        obj_crops = (obj_crops / 255 - 0.5) * 2
        
        # fg
        contact_maps = (hand_masks | obj_masks).astype(np.uint8)
        
        data = {'video': frames, 'mask': masks, 'obj_mask': obj_masks, 'object': object, 'caption': caption, 
                'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask_attn': mask_attn, 'obj_img': obj_crops, 'contact_maps': contact_maps}

        return data
    
    def __len__(self):
        return len(self.metadata)

def format_action(action):
    if '&' in action:
        action = action.replace('&', ' ')
    elif '-' in action:
        action = action.replace('-', ' ')
    elif action == 'Carrywithbothhands':
        action = 'Carry with both hands'
    return action.lower()

def format_mask(masks, object):
    #     mask_f --> flatten mask
    #     array([[  0,   0,   0],
    #            [128,   0,   0],
    #            [  0, 128,   0],
    #            [128, 128,   0],
    #            [  0,   0, 128],
    #            [128,   0, 128],
    #            [  0, 128, 128]], dtype=uint8)
    mask_f = masks.flatten(1)
    if object in ['mug', 'bottle', 'kettle', 'knife']:
        condition = (mask_f[0] == 128) & (mask_f[1] == 128) & (mask_f[2] == 0)
        mask_f[:, condition] = torch.tensor([[0], [0], [0]], dtype=mask_f.dtype)
    elif object == 'laptop':
        condition1 = (mask_f[0] == 128) & (mask_f[1] == 0) & (mask_f[2] == 0)
        condition2 = (mask_f[0] == 128) & (mask_f[1] == 128) & (mask_f[2] == 0)
        mask_f[:, condition1] = torch.tensor([[128], [128], [0]], dtype=mask_f.dtype)
        mask_f[:, condition2] = torch.tensor([[128], [0], [0]], dtype=mask_f.dtype)
    elif object == 'bucket':
        condition1 = (mask_f[0] == 0) & (mask_f[1] == 0) & (mask_f[2] == 128)
        condition2 = (mask_f[0] == 128) & (mask_f[1] == 0) & (mask_f[2] == 128)
        condition = condition1 | condition2
        mask_f[:, condition] = torch.tensor([[0], [0], [0]], dtype=mask_f.dtype)
    elif object == 'trashcan':
        condition = (mask_f[0] == 0) & (mask_f[1] == 0) & (mask_f[2] == 128)
        mask_f[:, condition] = torch.tensor([[0], [0], [0]], dtype=mask_f.dtype)
    elif object == 'chair':
        # curr_lhand_tensor = torch.tensor([128, 128, 0]).unsqueeze(-1)
        condition = (mask_f[0] == 128) & (mask_f[1] == 128) & (mask_f[2] == 0)
        mask_f[:, condition] = torch.tensor([[0], [128], [128]], dtype=mask_f.dtype)
    elif object in ['scissors', 'pliers']:
        # remove the object part in the mask
        condition = (mask_f[0] == 128) & (mask_f[1] == 128) & (mask_f[2] == 0)
        mask_f[:, condition] = torch.tensor([[128], [0], [0]], dtype=mask_f.dtype)
    
    # left hand color to right hand color
    condition = (mask_f[0] == 0) & (mask_f[1] == 128) & (mask_f[2] == 128)
    mask_f[:, condition] = torch.tensor([[0], [255], [0]], dtype=mask_f.dtype)
    
    # Yellow to Blue
    condition = (mask_f[0] == 128) & (mask_f[1] == 128) & (mask_f[2] == 0)
    mask_f[:, condition] = torch.tensor([[0], [0], [255]], dtype=mask_f.dtype)
    
    mask_f[[mask_f == 128]] = 255
    masks = mask_f.reshape(masks.shape)
    return masks

def format_template(action, object, mask_f):
    condition = (mask_f[0] == 0) & (mask_f[1] == 128) & (mask_f[2] == 128)
    two_hands = True if condition.sum() > 0 else False

    if action in ['grasp', 'dump', 'open', 'close', 'carry', 'press', 'push', 'pull']:
        action = action[:-1] + 'ing' if action[-1] == 'e' else action + 'ing'
        action_template = f'a hand {action}'
    elif action == 'binding':
        action_template = 'a hand binding with'
    elif action == 'pickup':
        action_template = 'a hand picking up'
    elif action == 'putdown':
        action_template = 'a hand putting down'
    elif 'hands' in action:
        action_template = 'two hands carrying'
    # for paper-cut and cut
    elif 'cut' in action:
        action_template = 'a hand cutting with'
    elif 'turn on' in action:
        action_template = 'a hand turning on the switch of'
    else:
        action_template = f'a hand {action}ing'
    
    if object == 'storagefurniture':
        object = 'storage furniture'
    elif object == 'trashcan':
        object = 'trash can'
    
    if object[-1] == 's':
        template = f'{action_template} {object}'
    else:
        if object[0].lower() in ['a', 'e', 'i', 'o', 'u']:
            template = f'{action_template} an {object}'
        else:
            template = f'{action_template} a {object}'
    
    template = template.replace('a hand', 'two hands') if two_hands else template
    
    return template

def remove_small_regions(mask: torch.Tensor, min_size: int) -> torch.Tensor:
    """
    Remove small connected components from a binary mask.
    
    Args:
        mask (torch.Tensor): HxW binary mask tensor (values 0 or 1).
        min_size (int): Minimum size of connected components to keep.
        
    Returns:
        torch.Tensor: Cleaned mask with small components removed.
    """
    mask_np = mask.cpu().numpy()  # Convert to NumPy
    labeled_array, num_features = label(mask_np)  # Label connected components
    
    # Count the size of each component
    component_sizes = np.bincount(labeled_array.ravel())

    # Ignore background (component 0)
    largest_component = np.argmax(component_sizes[1:]) + 1  

    # Keep only the largest region
    mask_largest = (labeled_array == largest_component).astype(np.uint8)
    
    return torch.tensor(mask_largest, dtype=torch.uint8, device=mask.device)


if __name__== "__main__":
    
    meta_path = "/train_data/HOI4D/HOI4D-Instructions/train_data.csv" ## path to the meta file
    data_dir = "/train_data/HOI4D" ## path to the data directory
    save_dir = "/workspace/save_dir" ## path to the save directory
    print('hello world'); exit()
    
    dataset = HOI4D(meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256,448],
                 frame_stride=1,
                 spatial_transform="resize_center_crop",
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=True
                 )
    dataloader = DataLoader(dataset,
                    batch_size=1,
                    num_workers=0,
                    shuffle=False)

    
    import sys
    sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
    from utils.save_video import tensor_to_mp4
    for i, batch in tqdm(enumerate(dataloader), desc="Data Batch"):
        video = batch['video']
        name = batch['path'][0].split('videos/')[-1].replace('/','_')
        tensor_to_mp4(video, save_dir+'/'+name, fps=8)

