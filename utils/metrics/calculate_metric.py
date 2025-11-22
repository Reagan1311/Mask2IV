import os
import sys
import numpy as np
import torch
from torch.nn import functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as T
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips
from calculate_viclip import calculate_viclip
from calculate_egovlp import calculate_egovlp
# from torchvision.models.optical_flow import raft_large

# ps: pixel value should be in [0, 1]!
# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# pitfall: fid will hang forever, if not specified "sync_on_compute=False" for DDP training
fid = FrechetInceptionDistance(feature=2048, normalize=True, sync_on_compute=False)
fid_transform = T.Compose([
    T.Resize(299),
    T.CenterCrop(299)
])

# raft = raft_large(pretrained=True, progress=False)
# raft = raft.eval()


def calculate_metric(gen_v, gt_v, caption, only_final=True, max_bs=5):
    # video shape: [N_test, timestamps, 3, h, w], range: [0, 1]
    start_time = time.time()
    print('Calculating metrics for gen_v {} and gt_v {}'.format(gen_v.shape, gt_v.shape))
    results = {}
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # # remove the top
    # gen_v = gen_v[:, :, :, 40:, :] 
    # gt_v = gt_v[:, :, :, 40:, :]
    
    results['fid'] = calculate_fid(gen_v, gt_v, fid_transform, fid, device, max_bs=max_bs * 10)
    results['fvd'] = calculate_fvd(gen_v, gt_v, device, method='styleganv', only_final=only_final, max_bs=max_bs)
    # results['fvd'] = calculate_fvd(videos1, videos2, device, method='videogpt', only_final=only_final)
    results['ssim'] = calculate_ssim(gen_v.cpu(), gt_v.cpu(), only_final=only_final)['value'][0]
    results['psnr'] = calculate_psnr(gen_v.cpu(), gt_v.cpu(), only_final=only_final)['value'][0]
    results['lpips'] = calculate_lpips(gen_v, gt_v, device, only_final=only_final)['value'][0]
    
    # gen_ad, gt_ad = calculate_ad(gen_v, gt_v, raft, device, max_bs=max_bs)
    # results['gen_ad'], results['gt_ad'] = gen_ad, gt_ad
    
    vt_results = calculate_viclip(gen_v, gt_v, caption, max_bs=max_bs)
    egovlp_results = calculate_egovlp(gen_v, gt_v, caption, max_bs=max_bs)
    
    results.update(vt_results)
    results.update(egovlp_results)

    print("Calculating metrics done in {:.2f} seconds.".format(time.time() - start_time))

    return results


def calculate_fid(gen_v, gt_v, transform, fid, device, max_bs):
    print("Calculating FID...")
    fid = fid.to(device)
    gen_v_f, gt_v_f = gen_v.flatten(0, 1).to(device), gt_v.flatten(0, 1).to(device)
    gen_v_f, gt_v_f = transform(gen_v_f), transform(gt_v_f)
    
    for i in range(0, gen_v.shape[0], max_bs):
        gen_v_b = gen_v_f[i:i + max_bs]
        gt_v_b = gt_v_f[i:i + max_bs]
        fid.update(gen_v_b, real=False)
        fid.update(gt_v_b, real=True)
    fid_score = fid.compute().item()
    fid.reset()
    return fid_score

# average displacement
def calculate_ad(gen_v, gt_v, model, device, max_bs):
    model = model.to(device)
    # [0, 1] -> [-1, 1], [N_test, timestamps, 3, h, w]
    gen_v, gt_v = (gen_v.to(device) - 0.5) * 2.0, (gt_v.to(device) - 0.5) * 2.0
    num_frames = gen_v.shape[1]
    
    gen_flows_all, gt_flows_all = [], []
    for i in range(0, gen_v.shape[0], max_bs):
        gen_flows, gt_flows = [], []
        gen_v_b = gen_v[i:i + max_bs]
        gt_v_b = gt_v[i:i + max_bs]
        for f in range(num_frames-1):
            flow_gen = model(gen_v_b[:, f], gen_v_b[:, f + 1])[-1].cpu().numpy() # bs x 2 x h x w
            flow_gt = model(gt_v_b[:, f], gt_v_b[:, f + 1])[-1].cpu().numpy()
            gen_flows.append(flow_gen)
            gt_flows.append(flow_gt)
        
        gen_flows_all.append(np.array(gen_flows)) # T-1 x bs x 2 x h x w
        gt_flows_all.append(np.array(gt_flows))

    gen_flows_all = np.concatenate(gen_flows_all, axis=1)  # T-1 x N_test x 2 x h x w
    gt_flows_all = np.concatenate(gt_flows_all, axis=1)

    gen_dis = np.linalg.norm(np.array(gen_flows_all), axis=2)   # T-1 x N_test x h x w
    gt_dis = np.linalg.norm(np.array(gt_flows_all), axis=2)
    gen_dis = gen_dis.mean()
    gt_dis = gt_dis.mean()

    return gen_dis, gt_dis
    
    