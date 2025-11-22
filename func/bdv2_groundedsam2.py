import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
import pandas as pd
import shutil
from tqdm import tqdm



def annotate_frame(img, masks, object_ids, ID_TO_OBJECTS):
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
        mask=masks, # (n, h, w)
        class_id=np.array(object_ids, dtype=np.int32),
    )
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    return annotated_frame

"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)


# init grounding dino model from huggingface
model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


# ===================================================
# load bridgedataV2 csv
df = pd.read_csv('/workspace/exp_outputs/bdv2_data.csv')
df_filter = df[(df['duration'] <= 50) & (df['confidence']>=0.4)]

# df_filter = pd.read_csv('/workspace/exp_outputs/bdv2_valid_data.csv')

gripper_name = 'black robot gripper'
# gripper_name = 'robot gripper'

# temp_save = '/workspace/exp_outputs/bd_seg_test'
# os.makedirs(temp_save, exist_ok=True)
# valid_save = 0

# Random Sampling
# df_filter = df_filter.sample(frac=1, random_state=42).reset_index(drop=True)
for index, row in tqdm(df_filter.iterrows(), total=df_filter.shape[0]):
    path, duration, caption, obj_name, confidence = row['path'], row['duration'], \
        row['caption'], row['object'], row['confidence']
    caption = caption.lower()
    
    cloth_condition = 'fold' in caption or 'move the cloth' in caption or 'moved the cloth' in caption or 'move cloth' in caption or 'moved cloth' in caption 
    
    if cloth_condition:
        obj_name = ['cloth']
        # obj_name = ['towel']
    elif 'drawer' in caption and len(caption.split()) <=4:
        obj_name = ['drawer', 'drawer handle']
    else:
        obj_name = ['objects']
        # obj_name = ['objects', 'vegetables', 'fruits']

    text = ".".join([gripper_name] + obj_name) + "."
    video_dir = os.path.join(path, 'images0')
    save_dir = os.path.join(path, 'masks')

    # ===================================================

    # # setup the input image and text prompt for SAM 2 and Grounding DINO
    # # VERY important: text queries need to be lowercased + end with a dot
    # text = "car. black robot grapper."

    # # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`  

    # video_dir = "notebooks/videos/car"

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0][3:]))

    # init video predictor state
    inference_state = video_predictor.init_state(video_path=video_dir)

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    """
    Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame
    """

    # prompt grounding dino to get the box coordinates on specific frame
    img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
    image = Image.open(img_path)

    # run Grounding DINO on the image
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    
    # Get indices of non-empty labels
    result = results[0]
    valid_indices = [i for i, label in enumerate(result['labels']) if label != ""]

    # Filter dictionary entries using valid indices
    results = [{
        'scores': result['scores'][valid_indices],
        'boxes': result['boxes'][valid_indices],
        'text_labels': [result['text_labels'][i] for i in valid_indices],
        'labels': [result['labels'][i] for i in valid_indices]
    }]

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))

    # process the detection results
    input_boxes = results[0]["boxes"].cpu().numpy()
    OBJECTS = results[0]["labels"]
    
    if len(input_boxes) == 0:
        continue

    # prompt SAM 2 image predictor to get the mask for the object
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # convert the mask shape to (n, H, W)
    if masks.ndim == 3:
        masks = masks[None]
        scores = scores[None]
        logits = logits[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)

    """
    Step 3: Register each object's positive points to video predictor with seperate add_new_points call
    """

    PROMPT_TYPE_FOR_VIDEO = "box" # or "point"
    # PROMPT_TYPE_FOR_VIDEO = "point" # or "point"

    assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

    # If you are using point prompts, we uniformly sample positive points based on the mask
    if PROMPT_TYPE_FOR_VIDEO == "point":
        # sample the positive points from mask for each objects
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

        for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )
    # Using box prompt
    elif PROMPT_TYPE_FOR_VIDEO == "box":
        for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    # Using mask prompt is a more straightforward way
    elif PROMPT_TYPE_FOR_VIDEO == "mask":
        for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
            labels = np.ones((1), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                mask=mask
            )
    else:
        raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")


    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    """
    Step 5: Visualize the segment results across the video and save them
    """

    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=0)}
    IOU_THRESH = 0.75
    LARGE_IOU_THRESH = 0.90
    GRIPPER_IOU_THRESH = 0.65
    IOU_SPARSE_THRESH = 0.80
    gripper_ids = [key for key, value in ID_TO_OBJECTS.items() if value == gripper_name]
    obj_ids = [key for key, value in ID_TO_OBJECTS.items() if value in obj_name]

    if len(obj_ids) == 0 or len(gripper_ids) == 0:
        print('No gripper or objects detected')
        shutil.rmtree(save_dir, ignore_errors=True)
        continue
    
    masks_list = []
    for frame_idx, segments in video_segments.items():
        img = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))
        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)
        masks_list.append(masks)

    # compute IOU with every STEP frames
    iou_check_dense = []
    iou_check_sparse = []
    STEP = len(masks_list) // 10
    for id, value in ID_TO_OBJECTS.items():
        iou_check_sub = []
        frames_idx = list(range(0, len(masks_list), STEP))
        for i in range(len(frames_idx) - 1):
            curr_m = masks_list[frames_idx[i]][id]
            next_m = masks_list[frames_idx[i+1]][id]
            intersection = np.logical_and(curr_m, next_m).sum()
            union = np.logical_or(curr_m, next_m).sum()
            
            # remove objects occupying too large areas
            area = curr_m.sum() / curr_m.size
            if area > 0.5:
                iou = 1
            else:
                iou = intersection / union if union > 0 else 1
            iou_check_sub.append(iou)
        iou_check_dense.append(np.mean(iou_check_sub))
        
        first_m = masks_list[frames_idx[0]][id]
        last_m = masks_list[frames_idx[-1]][id]
        intersection = np.logical_and(first_m, last_m).sum()
        union = np.logical_or(first_m, last_m).sum()
        iou_check_sparse.append(intersection / union if union > 0 else 1)

    iou_check_dense = np.array(iou_check_dense)
    iou_check_sparse = np.array(iou_check_sparse)

    # remove the gripper that is not moving
    gripper_ids = [i for i in gripper_ids if iou_check_dense[i] < GRIPPER_IOU_THRESH]
    
    if len(gripper_ids) > 1:
        gripper_ids = [gripper_ids[results[0]['scores'][gripper_ids].argmax()]]
    elif len(gripper_ids) == 0:
        print('No moving gripper')
        shutil.rmtree(save_dir, ignore_errors=True)
        continue
  
    grip_first_m = np.zeros_like(masks_list[0][0])
    grip_last_m = np.zeros_like(masks_list[0][0])
    for i in gripper_ids:
        grip_first_m |= masks_list[0][i]
        grip_first_m |= masks_list[-1][i]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_grip_first_m = cv2.dilate(grip_first_m.astype(np.uint8), kernel)
    dilated_grip_last_m = cv2.dilate(grip_last_m.astype(np.uint8), kernel)
    
    for i in obj_ids[:]:
        iou_dense = iou_check_dense[i]
        iou_sparse = iou_check_sparse[i]
        thresh = LARGE_IOU_THRESH if ID_TO_OBJECTS[i] in ['drawer', 'cloth'] else IOU_THRESH
        obj_first_m = masks_list[0][i]
        obj_last_m = masks_list[-1][i]
        overlap_first = (obj_first_m & dilated_grip_first_m).sum()
        overlap_last = (obj_last_m & dilated_grip_first_m).sum()
        
        if ID_TO_OBJECTS[i] in ['drawer', 'cloth', 'drawer handle']:
            if iou_dense > thresh:
                obj_ids.remove(i)
        else:
            if iou_dense > thresh or iou_sparse > LARGE_IOU_THRESH or overlap_first > 0 or overlap_last > 0:
                obj_ids.remove(i)
                
    if len(obj_ids) == 0:
        print('No valid objects left')
        shutil.rmtree(save_dir, ignore_errors=True)
        continue
    
    if not ('cloth' in obj_name or 'drawer' in obj_name):
        obj_id = obj_ids[np.argmin(iou_check_dense[obj_ids])]
        obj_ids = [obj_id]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    for f_num, mask in enumerate(masks_list):
        save_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
        for idx, m in enumerate(mask):
            if idx in obj_ids:
                save_mask[m==True] = (255, 255, 255)
            elif idx in gripper_ids:
                save_mask[m==True] = (128, 128, 128)
        cv2.imwrite(os.path.join(save_dir, f"mask_{f_num:d}.png"), save_mask)


# """
# Step 6: Convert the annotated frames to video
# """

# output_video_path = "./children_tracking_demo_video.mp4"
# create_video_from_images(save_dir, output_video_path)
