import os, re
from omegaconf import OmegaConf
import logging
mainlogger = logging.getLogger('mainlogger')

import torch
from collections import OrderedDict

def init_workspace(name, logdir, model_config, lightning_config, rank=0):
    workdir = os.path.join(logdir, name)
    ckptdir = os.path.join(workdir, "checkpoints")
    cfgdir = os.path.join(workdir, "configs")
    loginfo = os.path.join(workdir, "loginfo")

    # Create logdirs and save configs (all ranks will do to avoid missing directory error if rank:0 is slower)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    os.makedirs(loginfo, exist_ok=True)

    if rank == 0:
        if "callbacks" in lightning_config and 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
            os.makedirs(os.path.join(ckptdir, 'trainstep_checkpoints'), exist_ok=True)
        OmegaConf.save(model_config, os.path.join(cfgdir, "model.yaml"))
        OmegaConf.save(OmegaConf.create({"lightning": lightning_config}), os.path.join(cfgdir, "lightning.yaml"))
    return workdir, ckptdir, cfgdir, loginfo

def check_config_attribute(config, name):
    if name in config:
        value = getattr(config, name)
        return value
    else:
        return None

def get_trainer_callbacks(lightning_config, config, logdir, ckptdir, logger):
    default_callbacks_cfg = {
        "model_checkpoint": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch}",
                "verbose": True,
                "save_last": False,
            }
        },
        "batch_logger": {
            "target": "callbacks.ImageLogger",
            "params": {
                "save_dir": logdir,
                "batch_frequency": 1000,
                "max_images": 4,
                "clamp": True,
            }
        },    
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                "log_momentum": False
            }
        },
        "cuda_callback": {
            "target": "callbacks.CUDACallback"
        },
    }

    ## optional setting for saving checkpoints
    monitor_metric = check_config_attribute(config.model.params, "monitor")
    if monitor_metric is not None:
        mainlogger.info(f"Monitoring {monitor_metric} as checkpoint metric.")
        default_callbacks_cfg["model_checkpoint"]["params"]["monitor"] = monitor_metric
        default_callbacks_cfg["model_checkpoint"]["params"]["save_top_k"] = 3
        default_callbacks_cfg["model_checkpoint"]["params"]["mode"] = "min"

    if 'model_checkpoint' in lightning_config.callbacks:
        default_callbacks_cfg["model_checkpoint"]["params"]["dirpath"] = ckptdir
    
    if 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
        mainlogger.info('Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint': {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                                                   'params': {
                                                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                                                        "filename": "{epoch}-{step}",
                                                        "verbose": True,
                                                        'save_top_k': -1,
                                                        'every_n_train_steps': 10000,
                                                        'save_weights_only': True
                                                    }
                                                }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    return callbacks_cfg

def get_trainer_logger(lightning_config, logdir, on_val):
    default_logger_cfgs = {
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "save_dir": logdir,
                "name": "tensorboard",
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.CSVLogger",
            "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
    }
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    if not on_val:
        os.makedirs(os.path.join(logdir, "tensorboard"), exist_ok=True)
        default_logger_cfg = default_logger_cfgs["tensorboard"]
    else:
        default_logger_cfg = default_logger_cfgs["testtube"]
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    return logger_cfg

def get_trainer_strategy(lightning_config):
    default_strategy_dict = {
        "target": "pytorch_lightning.strategies.DDPShardedStrategy"
    }
    if "strategy" in lightning_config:
        strategy_cfg = lightning_config.strategy
        return strategy_cfg
    else:
        strategy_cfg = OmegaConf.create()

    strategy_cfg = OmegaConf.merge(default_strategy_dict, strategy_cfg)
    return strategy_cfg

def load_checkpoints(model, model_cfg, ignore_keys=None, strict=True):
    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        assert os.path.exists(pretrained_ckpt), "Error: Pre-trained checkpoint NOT found at:%s"%pretrained_ckpt
        mainlogger.info(">>> Load weights from pretrained checkpoint")

        pl_sd = torch.load(pretrained_ckpt, map_location="cpu")
        try:
            if 'state_dict' in pl_sd.keys():
                if getattr(model_cfg.params, 'control_stage_config', None) is not None:
                    strict = False
                    remove_keys = []
                    add_keys = []
                    for k in pl_sd["state_dict"].keys():
                        if 'model.diffusion_model' in k:
                            k_new = k.replace('model.diffusion_model', 'model')
                            remove_keys.append((k, k_new))
                        
                    for k in model.state_dict().keys():
                        if 'control_model' in k and 'zero_convs' not in k and 'input_hint_block' not in k and 'middle_block_out' not in k and 'attn2_mask' not in k and 'norm2_2' not in k and 'obj_attn' not in k and 'mask_mlp' not in k and 'obj_mlp' not in k:
                            add_keys.append(k)
                    
                    for (k, k_new) in remove_keys:
                        pl_sd["state_dict"][k_new] = pl_sd["state_dict"][k].clone()
                        del pl_sd['state_dict'][k]
                        
                    for k in add_keys:
                        pl_sd["state_dict"][k] = pl_sd["state_dict"][k.replace('control_model', 'model')].clone()
                
                if ignore_keys is not None:
                    strict = False
                    for key in ignore_keys:
                        model.state_dict()[key].zero_()
                        model.state_dict()[key][:, :8, :, :].copy_(pl_sd["state_dict"][key])
                        pl_sd["state_dict"].pop(key, None)
                                            
                # remove keys from pretrained checkpoint if the embedder is not CLIP
                remove_keys = []
                if 'CLIP' not in model_cfg.params.img_cond_stage_config.target:
                    strict = False
                    for k in pl_sd['state_dict'].keys():
                        if 'image_proj_model.proj_in' in k or 'embedder' in k:
                            remove_keys.append(k)

                if getattr(model_cfg.params, 'obj_cond_stage_config', None) is not None:
                    strict = False
                    for k in pl_sd['state_dict'].keys():
                        if 'object_proj_model' in k or 'obj_embedder' in k:
                            remove_keys.append(k)                    

                for k in remove_keys:
                    del pl_sd['state_dict'][k]
                
                model.load_state_dict(pl_sd["state_dict"], strict=strict)
                mainlogger.info(">>> Loaded weights from pretrained checkpoint: %s"%pretrained_ckpt)
            else:
                # deepspeed
                new_pl_sd = OrderedDict()
                for key in pl_sd['module'].keys():
                    new_pl_sd[key[16:]]=pl_sd['module'][key]
                model.load_state_dict(new_pl_sd, strict=True)
        except:
            model.load_state_dict(pl_sd)
    else:
        mainlogger.info(">>> Start training from scratch")

    return model

def set_logger(logfile, name='mainlogger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s-%(levelname)s: %(message)s"))
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger