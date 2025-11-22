import argparse, os, sys, glob
import datetime, time
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from PIL import Image
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from utils.utils import instantiate_from_config
import random

from lvdm.data.hoi4d import format_action, format_mask, format_template


def get_filelist(data_dir, postfixes):
    patterns = [os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list

def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k,v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]]=state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model

def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list

def bottom_aligned_center_crop(image: torch.Tensor, crop_height: int, crop_width: int):
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

def load_data_prompts(data_dir, save_dir, video_size=(256,256), video_frames=16, interp=False, second_stage=False):
    transform = transforms.Compose([
        transforms.Resize(int(min(video_size) * 1.2), antialias=True),
        transforms.CenterCrop(video_size),
        # transforms.Lambda(lambda x: bottom_aligned_center_crop(x, video_size[0], video_size[1])),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    
    ## load video
    file_list = get_filelist(data_dir, ['jpg'])
    prompt_list = [format_action(f.split('_')[-1].split('.')[0]) for f in file_list]
    obj_list = [f.split('_')[-2].lower() for f in file_list]
    
    # fs_list = [(int(f.split('_')[-3]) - int(f.split('_')[-4]) + 1) // 16 for f in file_list]  # frame stride
    # fps_list = [15 // f for f in fs_list]

    # constant fps
    fs_list = [1] * len(file_list)
    fps_list = [15] * len(file_list)

    # assert len(file_list) == n_samples, "Error: data and prompts are NOT paired!"
    data_list = []
    mask_list = []
    filename_list = []
    n_samples = len(file_list)
    for idx in range(n_samples):
        if interp:
            image1 = Image.open(file_list[2*idx]).convert('RGB')
            image_tensor1 = transform(image1).unsqueeze(1) # [c,1,h,w]
            image2 = Image.open(file_list[2*idx+1]).convert('RGB')
            image_tensor2 = transform(image2).unsqueeze(1) # [c,1,h,w]
            frame_tensor1 = repeat(image_tensor1, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
            frame_tensor2 = repeat(image_tensor2, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
            frame_tensor = torch.cat([frame_tensor1, frame_tensor2], dim=1)
            _, filename = os.path.split(file_list[idx*2])
        else:
            # image = Image.open(file_list[idx]).convert('RGB')
            image = cv2.cvtColor(cv2.imread(file_list[idx]), cv2.COLOR_BGR2RGB)
            image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(1).float()
            # image_tensor = transform(image).unsqueeze(1) # [c,1,h,w]
            frame_tensor = repeat(image_tensor, 'c t h w -> c (repeat t) h w', repeat=video_frames)
            frame_tensor = transform(frame_tensor)
            if not second_stage:
                # mask = Image.open(file_list[idx].replace('jpg', 'png')).convert('RGB')
                mask = cv2.cvtColor(cv2.imread(file_list[idx].replace('jpg', 'png')), cv2.COLOR_BGR2RGB)
                mask_tensor = torch.tensor(mask).permute(2, 0, 1).unsqueeze(1).float()
                mask_tensor = repeat(mask_tensor, 'c t h w -> c (repeat t) h w', repeat=video_frames)
                mask_tensor = format_mask(mask_tensor, obj_list[idx])
                mask_tensor = transform(mask_tensor)
            else:
                mask_traj_path = os.path.join(save_dir, os.path.basename(file_list[idx]).split('.')[0]+'_sample0_mask.mp4')
                mask_trajs = cv2.VideoCapture(mask_traj_path)
                mask_trajs = [transform(torch.tensor(x).permute(2, 0, 1).unsqueeze(1).float()) for x in _frame_from_video(mask_trajs, rgb=True)]
                # mask_trajs = [transform(Image.fromarray(x)).unsqueeze(1) for x in _frame_from_video(mask_trajs, rgb=True)]
                mask_tensor = torch.cat(mask_trajs, dim=1)

            _, filename = os.path.split(file_list[idx])
            filename = filename.split('.')[0]

        frame_tensor = (frame_tensor / 255 - 0.5) * 2
        mask_tensor = (mask_tensor / 255 - 0.5) * 2
        data_list.append(frame_tensor)
        mask_list.append(mask_tensor)
        filename_list.append(filename)

    prompt_list = [format_template(action, obj_list[i], mask_list[i].flatten(1)) for i, action in enumerate(prompt_list)]
    
    return filename_list, data_list, mask_list, prompt_list, fps_list, fs_list


def save_results(prompt, samples, filename, fakedir, fps=8, loop=False):
    filename = filename.split('.')[0]+'.mp4'
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        if loop:
            video = video[:-1,...]
        
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, h, n*w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        path = os.path.join(savedirs[idx], filename)
        torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'}) ## crf indicates the quality


def save_results_seperate(prompt, samples, filename, fakedir, fps=10, loop=False, mask=False):
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        if loop: # remove the last frame
            video = video[:,:,:-1,...]
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        for i in range(n):
            grid = video[i,...]
            grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(1, 2, 3, 0) #thwc
            filename = f'{filename.split(".")[0]}_sample{i}_mask.mp4' if mask \
                else f'{filename.split(".")[0]}_sample{i}.mp4'
            path = os.path.join(savedirs[idx].replace('samples', 'samples_separate'), filename)
            torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'})

def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def image_guided_synthesis(model, prompts, videos, masks, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                        unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False, multiple_cond_cfg=False, 
                        loop=False, interp=False, timestep_spacing='uniform', guidance_rescale=0.0, first_stage=False, **kwargs):
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    if isinstance(fs, int):
        fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)
    else:
        fs = torch.tensor(fs, dtype=torch.long, device=model.device)

    if not text_input or not first_stage:
        prompts = [""]*batch_size

    img = masks[:,:,0] if first_stage else videos[:,:,0]  #bchw
    first_frame = videos[:, :, 0]
    img_emb = model.embedder(first_frame) ## blc
    img_emb = model.image_proj_model(img_emb)

    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}
    # if model.model.conditioning_key == 'hybrid':
    z = get_latent_z(model, videos) # b c t h w
    # First stage generation
    if first_stage:
        z_mask = get_latent_z(model, masks)
        z = torch.cat([z_mask, z], dim=1)
    else:
        cond.update({"c_control": [masks]})

    if loop or interp:
        img_cat_cond = torch.zeros_like(z)
        img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
        img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
    else:
        img_cat_cond = z[:,:,:1,:,:]
        img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
        # # for the baseline concatenation model
        if not first_stage:
            z_extra = get_latent_z(model, masks)
            img_cat_cond = torch.cat([img_cat_cond, z_extra], dim=1)

    cond["c_concat"] = [img_cat_cond] # b c 1 h w
    
    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img)) ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}
        # if model.model.conditioning_key == 'hybrid':
        uc["c_concat"] = [img_cat_cond]
        if not first_stage:
            uc.update({"c_control": [masks]})
    else:
        uc = None

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    z0 = None
    cond_mask = None

    batch_variants = []
    for _ in range(n_samples):

        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:

            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=batch_size,
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            cfg_img=cfg_img, 
                                            mask=cond_mask,
                                            # attn_mask=attn_mask,
                                            x0=cond_z0,
                                            fs=fs,
                                            timestep_spacing=timestep_spacing,
                                            guidance_rescale=guidance_rescale,
                                            **kwargs
                                            )

        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)


def load_model(args, gpu_no, config, ckpt):
    config = OmegaConf.load(config)
    model_config = config.pop("model", OmegaConf.create())
    
    ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    model.perframe_ae = args.perframe_ae
    assert os.path.exists(ckpt), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, ckpt)
    model.eval()
    return model


def _frame_from_video(video, rgb=False):
    while video.isOpened():
        success, frame = video.read()
        if success:
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame
        else:
            break
    
    
def run_inference(args, gpu_num, gpu_no):
    ## Load models from two stages
    if not args.second_stage_only:
        model1 = load_model(args, gpu_no, args.config1, args.ckpt_path1)
    model2 = load_model(args, gpu_no, args.config2, args.ckpt_path2)
    ## run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"
    ## latent noise shape
    h, w, channels = args.height // 8, args.width // 8, 4
    # channels = model2.model.out_channels
    # channels = model2.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'Inference with {n_frames} frames')
    noise_shape = [args.bs, channels, n_frames, h, w]

    fakedir = os.path.join(args.savedir, "samples")
    fakedir_separate = os.path.join(args.savedir, "samples_separate")

    # os.makedirs(fakedir, exist_ok=True)
    os.makedirs(fakedir_separate, exist_ok=True)

    ## prompt file setting
    assert os.path.exists(args.prompt_dir), "Error: prompt file Not Found!"
    filename_list, data_list, mask_list, prompt_list, fps_list, fs_list = load_data_prompts(args.prompt_dir, fakedir_separate, video_size=(args.height, args.width), video_frames=n_frames, interp=args.interp, second_stage=args.second_stage_only)
    
    if args.fps_cond:
        fs_list = fps_list
    
    num_samples = len(prompt_list)
    samples_split = num_samples // gpu_num
    print('Prompts testing [rank:%d] %d/%d samples loaded.'%(gpu_no, samples_split, num_samples))
    #indices = random.choices(list(range(0, num_samples)), k=samples_per_device)
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    prompt_list_rank = [prompt_list[i] for i in indices]
    data_list_rank = [data_list[i] for i in indices]
    mask_list_rank = [mask_list[i] for i in indices]
    filename_list_rank = [filename_list[i] for i in indices]
    fs_list_rank = [fs_list[i] for i in indices]

    if not args.second_stage_only:
        # First stage inference: first frame & mask & action text --> motion trajectories in the format of mask
        first_stage_masks = []
        start = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast():
            for idx, indice in tqdm(enumerate(range(0, len(prompt_list_rank), args.bs)), desc='Sample Batch'):
                prompts = prompt_list_rank[indice:indice+args.bs]
                videos = data_list_rank[indice:indice+args.bs]
                masks = mask_list_rank[indice:indice+args.bs]
                filenames = filename_list_rank[indice:indice+args.bs]
                fss = fs_list_rank[indice:indice+args.bs]

                if isinstance(videos, list):
                    videos = torch.stack(videos, dim=0).to("cuda")
                else:
                    videos = videos.unsqueeze(0).to("cuda")

                if isinstance(masks, list):
                    masks = torch.stack(masks, dim=0).to("cuda")
                else:
                    masks = masks.unsqueeze(0).to("cuda")

                # shape: 1, 1, 3, 16, 320, 512
                batch_samples = image_guided_synthesis(model1, prompts, videos, masks, noise_shape, args.n_samples, args.ddim_steps, args.ddim_eta,
                                    args.unconditional_guidance_scale, args.cfg_img, fss, args.text_input, args.multiple_cond_cfg, args.loop, 
                                    args.interp, args.timestep_spacing, args.guidance_rescale, first_stage=True)
                for sample in batch_samples:
                    temp = torch.where(sample[0] >=0, 1, -1).float()
                    first_stage_masks.append(temp)

                ## save each example individually
                for nn, samples in enumerate(batch_samples):
                    ## samples : [n_samples,c,t,h,w] -> 1, 3, 16, 320, 512
                    prompt = prompts[nn]
                    filename = filenames[nn]
                    # save_results(prompt, samples, filename, fakedir, fps=8, loop=args.loop)
                    save_results_seperate(prompt, samples, filename, fakedir, fps=8, loop=args.loop, mask=True)

        print(f"Saved in {args.savedir}. Time used in the first stage: {(time.time() - start):.2f} seconds")
        mask_list_rank = first_stage_masks
    
    # Second stage inference: first frame & mask trajectories & action text --> HOI interaction video
    start = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for idx, indice in tqdm(enumerate(range(0, len(prompt_list_rank), args.bs)), desc='Sample Batch'):
            prompts = prompt_list_rank[indice:indice+args.bs]
            videos = data_list_rank[indice:indice+args.bs]
            masks = mask_list_rank[indice:indice+args.bs]
            filenames = filename_list_rank[indice:indice+args.bs]
            fss = fs_list_rank[indice:indice+args.bs]

            if isinstance(videos, list):
                videos = torch.stack(videos, dim=0).to("cuda")
            else:
                videos = videos.unsqueeze(0).to("cuda")

            if isinstance(masks, list):
                masks = torch.stack(masks, dim=0).to("cuda")
            else:
                masks = masks.unsqueeze(0).to("cuda")

            batch_samples = image_guided_synthesis(model2, prompts, videos, masks, noise_shape, args.n_samples, args.ddim_steps, args.ddim_eta, 
                                                   args.unconditional_guidance_scale, args.cfg_img, fss, args.text_input, args.multiple_cond_cfg, 
                                                   args.loop, args.interp, args.timestep_spacing, args.guidance_rescale)
            
            
            ## save each example individually
            for nn, samples in enumerate(batch_samples):
                ## samples : [n_samples,c,t,h,w]
                prompt = prompts[nn]
                filename = filenames[nn]
                # save_results(prompt, samples, filename, fakedir, fps=8, loop=args.loop)
                save_results_seperate(prompt, samples, filename, fakedir, fps=8, loop=args.loop)

    print(f"Saved in {args.savedir}. Time used in the second stage: {(time.time() - start):.2f} seconds")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path1", type=str, default=None, help="checkpoint path")
    parser.add_argument("--ckpt_path2", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config1", type=str, help="config (yaml) path")
    parser.add_argument("--config2", type=str, help="config (yaml) path")
    parser.add_argument("--prompt_dir", type=str, default=None, help="a data dir containing videos and prompts")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference, should be one")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--fps_cond", action='store_false', default=True, help="using fps / frame_stride as conditional input")
    # parser.add_argument("--frame_stride", type=int, default=3, help="frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=123, help="seed for seed_everything")
    parser.add_argument("--video_length", type=int, default=16, help="inference video length")
    parser.add_argument("--negative_prompt", action='store_true', default=False, help="negative prompt")
    parser.add_argument("--text_input", action='store_true', default=False, help="input text to I2V model or not")
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False, help="use multi-condition cfg or not")
    parser.add_argument("--cfg_img", type=float, default=None, help="guidance scale for image conditioning")
    parser.add_argument("--timestep_spacing", type=str, default="uniform", help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.")
    parser.add_argument("--guidance_rescale", type=float, default=0.0, help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)")
    parser.add_argument("--perframe_ae", action='store_true', default=False, help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")

    parser.add_argument("--second_stage_only", action='store_true', default=False, help="implement second stage inference only")

    ## currently not support looping video and generative frame interpolation
    parser.add_argument("--loop", action='store_true', default=False, help="generate looping videos or not")
    parser.add_argument("--interp", action='store_true', default=False, help="generate generative frame interpolation or not")
    return parser


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@DynamiCrafter cond-Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()

    seed = args.seed
    if seed < 0:
        seed = random.randint(0, 2 ** 31)
    seed_everything(seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)