import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import polynomial_kernel

def polynomial_mmd(X, Y):
    m = X.shape[0]
    n = Y.shape[0]
    # compute kernels
    K_XX = polynomial_kernel(X)
    K_YY = polynomial_kernel(Y)
    K_XY = polynomial_kernel(X, Y)
    # compute mmd distance
    K_XX_sum = (K_XX.sum() - np.diagonal(K_XX).sum()) / (m * (m - 1))
    K_YY_sum = (K_YY.sum() - np.diagonal(K_YY).sum()) / (n * (n - 1))
    K_XY_sum = K_XY.sum() / (m * n)
    mmd = K_XX_sum + K_YY_sum - 2 * K_XY_sum
    return mmd

def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 

    return x

def calculate_fvd(videos1, videos2, device, method='styleganv', only_final=False, max_bs=8):

    if method == 'styleganv':
        from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
    elif method == 'videogpt':
        from fvd.videogpt.fvd import load_i3d_pretrained, frechet_distance
        from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats

    print("calculate_fvd...")

    # videos [batch_size, timestamps, channel, h, w]
    
    assert videos1.shape == videos2.shape

    i3d = load_i3d_pretrained(device=device)
    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = []

    if only_final:

        assert videos1.shape[2] >= 10, "for calculate FVD, each clip_timestamp must >= 10"

        # videos_clip [batch_size, channel, timestamps, h, w]
        videos_clip1 = videos1
        videos_clip2 = videos2

        # get FVD features
        feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device, bs=max_bs)
        feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device, bs=max_bs)

        # calculate FVD
        # fvd_results.append(frechet_distance(feats1, feats2))
        fvd_results = frechet_distance(feats1, feats2)
        # if method == 'styleganv':
        #     kvd_results = polynomial_mmd(feats1, feats2)
        # elif method == 'videogpt':
        #     kvd_results = polynomial_mmd(feats1.cpu().numpy(), feats2.cpu().numpy())
    
    else:

        # for calculate FVD, each clip_timestamp must >= 10
        for clip_timestamp in tqdm(range(10, videos1.shape[-3]+1)):
        
            # get a video clip
            # videos_clip [batch_size, channel, timestamps[:clip], h, w]
            videos_clip1 = videos1[:, :, : clip_timestamp]
            videos_clip2 = videos2[:, :, : clip_timestamp]

            # get FVD features
            feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
            feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)
        
            # calculate FVD when timestamps[:clip]
            fvd_results.append(frechet_distance(feats1, feats2))

    # result = {
    #     "fvd": fvd_results,
    #     # 'kvd': kvd_results,
    # }

    return fvd_results

# test code / using example

def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 30
    CHANNEL = 3
    SIZE = 64
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")
    # device = torch.device("cpu")

    result = calculate_fvd(videos1, videos2, device, method='videogpt', only_final=False)
    print("[fvd-videogpt ]", result["value"])

    result = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=False)
    print("[fvd-styleganv]", result["value"])

if __name__ == "__main__":
    main()
