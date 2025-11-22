# version=$1 ##1024, 512, 256
seed=42

ckpt1=""
config1=configs/inference_512_bdv2_first.yaml

# maskcat
ckpt2=""
config2=configs/inference_512_bdv2_maskcat.yaml

prompt_dir=prompts/bdv2
res_dir="./exp_outputs/Mask2IV-inference/maskcat/bdv2"


CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/inference_bdv2_2stage.py \
--seed ${seed} \
--ckpt_path1 $ckpt1 \
--ckpt_path2 $ckpt2 \
--config1 $config1 \
--config2 $config2 \
--savedir $res_dir \
--n_samples 1 \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 1.0 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--text_input \
--video_length 16 \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae \
# --second_stage_only
