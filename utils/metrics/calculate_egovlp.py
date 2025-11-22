import os
import transformers
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Resize, CenterCrop
from torchvision.transforms._transforms_video import NormalizeVideo

from PIL import Image
import egovlp.model as module_arch
from egovlp.model import sim_matrix


egovlp_args = {
    # ego4d
    # "video_params": {
    #     "model": "SpaceTimeTransformer",
    #     "arch_config": "base_patch16_224",
    #     "num_frames": 4,
    #     "pretrained": True,
    #     "time_init": "zeros"
    # },
    # "text_params": {
    #     "model": "distilbert-base-uncased",
    #     "pretrained": True,
    #     "input": "text"
    # },
    # "projection": "minimal",
    # "load_checkpoint" : "/workspace/exp_outputs/checkpoints/egovlp/pretrained/egovlp.pth"
    
    # epickitchen
    "video_params": {
        "model": "SpaceTimeTransformer",
        "arch_config": "base_patch16_224",
        "num_frames": 16,
        "pretrained": True,
        "time_init": "zeros"
    },
    "text_params": {
        # "model": "/workspace/exp_outputs/checkpoints/distilbert-base-uncased.zip",
        "model": "distilbert-base-uncased",
        "pretrained": True,
        "input": "text"
    },
    "projection": "minimal",
    "load_checkpoint" : "/workspace/exp_outputs/checkpoints/egovlp/epic_mir_plus.pth"
}


egovlp_model = getattr(module_arch, "FrozenInTime")(**egovlp_args)
egovlp_model = egovlp_model.cuda()
egovlp_model.eval().requires_grad_(False)

tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='/workspace/exp_outputs/checkpoints/egovlp/distilbert-base-uncased', TOKENIZERS_PARALLELISM=False)

video_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def video_trans(data, transforms, size=224):
    # Normalize the data to the range [0, 1]
    b, t, c = data.shape[:3]
    data = transforms(data.flatten(0, 1))
    data = data.view(b, t, c, size, size)
    return data


def calculate_egovlp(gen_v, gt_v, captions, max_bs=8):
    results = {}
    
    gen_v = video_trans(gen_v, video_transforms)
    gt_v = video_trans(gt_v, video_transforms)    

    gt_vt_scores, gen_vt_scores, gen_gt_scores = [], [], []
    for i in range(0, gen_v.shape[0], max_bs):
        gen_v_batch = gen_v[i:i + max_bs]
        gt_v_batch = gt_v[i:i + max_bs]
        caption = captions[i:i + max_bs]
        gt_vt_score, gen_vt_score, gen_gt_score = run_egovlp(egovlp_model, gen_v_batch, gt_v_batch, caption)

        gt_vt_scores.extend(gt_vt_score)
        gen_vt_scores.extend(gen_vt_score)
        gen_gt_scores.extend(gen_gt_score)
    
    gt_vt_score = np.mean(gt_vt_scores)
    gen_vt_score = np.mean(gen_vt_scores)
    gen_gt_score = np.mean(gen_gt_scores)

    results['egovlp-t-gt'] = gt_vt_score
    results['egovlp-t-gen'] = gen_vt_score
    results['egovlp-v'] = gen_gt_score
    
    return results

def run_egovlp(model, gen_v, gt_v, caption):
    model = model.to(gen_v.device)

    gen_v_feats = model({'video': gen_v}, video_only=True, return_embeds=True)
    gt_v_feats = model({'video': gt_v}, video_only=True, return_embeds=True)

    token_desc = tokenizer(caption, return_tensors='pt', padding=True, truncation=True)
    token_desc = {key: val.cuda() for key, val in token_desc.items()}
    text_embed = model.compute_text(token_desc)

    sim_test_gt = F.cosine_similarity(gt_v_feats, text_embed).tolist()
    sim_text_gen = F.cosine_similarity(gen_v_feats, text_embed).tolist()
    sim_video = F.cosine_similarity(gen_v_feats, gt_v_feats).tolist()

    return sim_test_gt, sim_text_gen, sim_video


    # pre_image = transforms.ToTensor()(pre_image) if pre_image is not None else None
    # image_0 = transforms.ToTensor()(image_0)
    # image_1 = transforms.ToTensor()(image_1)

    # if dataset == 'epickitchen' and pre_image is not None:
    #     if pre_image.size() != image_1.size():  # In epickitchens, there could be a minor difference in the dimension like (3, 256, 456) vs. (3, 256, 455)
    #         image_1 = F.resize(image_1, pre_image.shape[1:])
    
    # if dataset == 'ego4d':
    #     images_0 = torch.stack([image_0, image_0, image_0, image_0], dim=0) if pre_image is None else torch.stack([pre_image, pre_image, image_0, image_0], dim=0)
    # elif dataset == 'epickitchen':
    #     images_0 = torch.stack([image_0] * 16, dim=0) if pre_image is None else torch.stack([pre_image] * 8 + [image_0] * 8, dim=0)
    # else:
    #     raise NotImplementedError
    # images_0 = images_0.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
    # images_0 = image_transforms(images_0)
    # images_0 = images_0.transpose(0, 1)  # recover
    # images_0 = images_0.unsqueeze(0)

    # if dataset == 'ego4d':
    #     images_1 = torch.stack([image_1, image_1, image_1, image_1], dim=0) if pre_image is None else torch.stack([pre_image, pre_image, image_1, image_1], dim=0)
    # elif dataset == 'epickitchen':
    #     images_1 = torch.stack([image_1] * 16, dim=0) if pre_image is None else torch.stack([pre_image] * 8 + [image_1] * 8, dim=0)
    # else:
    #     raise NotImplementedError
    # images_1 = images_1.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
    # images_1 = image_transforms(images_1)
    # images_1 = images_1.transpose(0, 1)  # recover
    # images_1 = images_1.unsqueeze(0)

    # images_0 = images_0.cuda()
    # images_1 = images_1.cuda()
    # images_0_features = model({'video': images_0}, video_only=True, return_embeds=True)
    # images_1_features = model({'video': images_1}, video_only=True, return_embeds=True)
    
    # sim_image = sim_matrix(images_0_features, images_1_features)

    # return sim_image


def compute_egovlp(gt_path, gen_path, use_context=False):
    module_args = egovlp_args
    model = getattr(module_arch, "FrozenInTime")(**module_args)
    model = model.cuda()
    model.eval().requires_grad_(False)

    count = 0
    avg_sim_image = 0

    for clip_id in tqdm(os.listdir(gt_path)):
        for action_id in os.listdir(os.path.join(gt_path, clip_id)):
            for frame in os.listdir(os.path.join(gt_path, clip_id, action_id)):
                assert os.path.exists(os.path.join(gen_path, clip_id, action_id, frame))
                gt = Image.open(os.path.join(gt_path, clip_id, action_id, frame))
                gen = Image.open(os.path.join(gen_path, clip_id, action_id, frame))

                # load first image
                if use_context:
                    pre_frame = sorted(os.listdir(os.path.join(gt_path.replace('val_gt_for_metric', 'val'), clip_id, action_id)))[0]
                    assert frame != pre_frame
                    pre_image = Image.open(os.path.join(gt_path.replace('val_gt_for_metric', 'val'), clip_id, action_id, pre_frame))
                else:
                    pre_image = None
                
                sim_image = run_egovlp(dataset, model, gt, gen, pre_image=pre_image)
                sim_image = sim_image.cpu().numpy().tolist()[0][0]
                avg_sim_image += sim_image
                count += 1

    avg_sim_image /= count

    return avg_sim_image
