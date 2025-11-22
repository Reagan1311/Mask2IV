import clip
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Resize, CenterCrop

from PIL import Image
import cv2
from utils.metrics.viclip import get_viclip, retrieve_text, _frame_from_video, get_vid_feat, frames2tensor, get_text_feat_dict

model_cfgs = {
    'viclip-l-internvid-10m-flt': {
        'size': 'l',
        'pretrained': '/workspace/exp_outputs/ViCLIP-L_InternVid-FLT-10M.pth',
    },
    'viclip-l-internvid-200m': {
        'size': 'l',
        'pretrained': 'xxx/ViCLIP-L_InternVid-200M.pth',
    },
    'viclip-b-internvid-10m-flt': {
        'size': 'b',
        'pretrained': 'xxx/ViCLIP-B_InternVid-FLT-10M.pth',
    },
    'viclip-b-internvid-200m': {
        'size': 'b',
        'pretrained': 'xxx/ViCLIP-B_InternVid-200M.pth',
    },
}

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
clip_mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1,1,3)
clip_std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1,1,3)

# device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = model_cfgs['viclip-l-internvid-10m-flt']
viclip_model = get_viclip(cfg['size'], cfg['pretrained'])
clip_model, preprocess = clip.load("ViT-B/32")


def norm(data, mean, std, device):
    mean = torch.from_numpy(mean).view(1,1,3,1,1).to(device)
    std = torch.from_numpy(std).view(1,1,3,1,1).to(device)
    if isinstance(data, torch.Tensor):
        data = (data - mean) / std
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = (data[i] - mean) / std
    return data
    

def resize(data, size=224, center_crop=False):
    # Normalize the data to the range [0, 1]
    b, t, c = data.shape[:3]
    data = data.flatten(0, 1)
    if not center_crop:
        data = F.interpolate(data, size=(size, size), mode='bilinear', align_corners=False)
    else:
        data = Resize(size)(data)
        data = CenterCrop(size)(data)
    data = data.view(b, t, c, size, size)
    return data


# def calculate_viclip(gen_v, gt_v, captions):
#     results = {}

#     gen_v = resize(gen_v, center_crop=True)
#     gt_v = resize(gt_v, center_crop=True)
#     gen_v, gt_v = norm([gen_v, gt_v], v_mean, v_std, device=gen_v.device)
#     gen_v_clip, gt_v_clip = norm([gen_v, gt_v], clip_mean, clip_std, device=gen_v.device)
        
#     gt_vt_score, gen_vt_score, gen_gt_score = viclip_scores(viclip_model, gen_v, gt_v, captions)
    
#     gen_fc_score = frame_consistency(clip_model, gen_v_clip)
#     gt_fc_score = frame_consistency(clip_model, gt_v_clip)
        
#     results['gt_vt_score'] = gt_vt_score.mean()
#     results['gen_vt_score'] = gen_vt_score.mean()
#     results['gen_gt_score'] = gen_gt_score.mean()
#     results['gen_fc_score'] = gen_fc_score.mean()
#     results['gt_fc_score'] = gt_fc_score.mean()
        
#     return results


def calculate_viclip(gen_v, gt_v, captions, max_bs=8):
    results = {}

    gen_v = resize(gen_v, center_crop=True)
    gt_v = resize(gt_v, center_crop=True)
    gen_v, gt_v = norm([gen_v, gt_v], v_mean, v_std, device=gen_v.device)
    gen_v_clip, gt_v_clip = norm([gen_v, gt_v], clip_mean, clip_std, device=gen_v.device)
    
    gt_vt_scores, gen_vt_scores, gen_gt_scores = [], [], []
    gen_fc_scores, gt_fc_scores = [], []
    for i in range(0, gen_v.shape[0], max_bs):
        gen_v_batch = gen_v[i:i + max_bs]
        gt_v_batch = gt_v[i:i + max_bs]
        gen_v_clip_batch = gen_v_clip[i:i + max_bs]
        gt_v_clip_batch = gt_v_clip[i:i + max_bs]
        caption = captions[i:i + max_bs]
        
        gt_vt_score, gen_vt_score, gen_gt_score = viclip_scores(viclip_model, gen_v_batch, gt_v_batch, caption)
        
        gen_fc_score = frame_consistency(clip_model, gen_v_clip_batch)
        gt_fc_score = frame_consistency(clip_model, gt_v_clip_batch)
        
        gt_vt_scores.extend(gt_vt_score)
        gen_vt_scores.extend(gen_vt_score)
        gen_gt_scores.extend(gen_gt_score)
        gen_fc_scores.extend(gen_fc_score)
        gt_fc_scores.extend(gt_fc_score)
        
    gt_vt_score = np.mean(gt_vt_scores)
    gen_vt_score = np.mean(gen_vt_scores)
    gen_gt_score = np.mean(gen_gt_scores)
    gen_fc_score = np.mean(gen_fc_scores)
    gt_fc_score = np.mean(gt_fc_scores)
        
    results['viclip-t-gt'] = gt_vt_score
    results['viclip-t'] = gen_vt_score
    results['viclip-v'] = gen_gt_score
    results['gen_fc'] = gen_fc_score
    results['gt_fc'] = gt_fc_score
        
    return results

def viclip_scores(models, gen_frames_tensor, frames_tensor, texts):
    clip, tokenizer = models['viclip'], models['tokenizer']
    clip = clip.to(gen_frames_tensor.device)
    with torch.no_grad():
        vid_feat = get_vid_feat(frames_tensor.float(), clip)
        gen_vid_feat = get_vid_feat(gen_frames_tensor.float(), clip)
        text_feat = clip.encode_text(texts)
    
    gt_vt_score = F.cosine_similarity(vid_feat, text_feat).tolist()
    gen_vt_score = F.cosine_similarity(gen_vid_feat, text_feat).tolist()
    gen_gt_score = F.cosine_similarity(gen_vid_feat, vid_feat).tolist()

    return gt_vt_score, gen_vt_score, gen_gt_score
    

def frame_consistency(model, videos, step=3):
    # videos shape: (B, T, C, H, W)
    model = model.to(videos.device)
    frame_features = []
    num_frames = videos.shape[1]
    for f in range(0, num_frames, step):
        with torch.no_grad():
            feature = model.encode_image(videos[:, f, :, :, :])
        frame_features.append(feature)

    consistencies = []
    for i in range(len(frame_features) - 1):
        sim = F.cosine_similarity(frame_features[i], frame_features[i+1])
        consistencies.append(sim)
    consistencies = torch.stack(consistencies)  # T-1 x b
    average_consistency = consistencies.sum(0) / len(consistencies)
    return average_consistency.tolist()


def get_text_feat_list(texts, clip, tokenizer, text_feat_l=[]):
    for t in texts:
        feat = clip.get_text_features(t, tokenizer, text_feat_l)
        text_feat_l.append(feat)
    return text_feat_l