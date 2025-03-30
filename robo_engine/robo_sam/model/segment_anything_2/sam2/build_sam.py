# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import hydra
import torch
import pdb
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):
    
    hydra_overrides_extra_org = hydra_overrides_extra.copy()
    try:
        if apply_postprocessing:
            hydra_overrides_extra = hydra_overrides_extra.copy()
            hydra_overrides_extra += [
                # dynamically fall back to multi-mask if the single mask is not stable
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            ]
            
        cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
        _load_checkpoint(model, ckpt_path)
    except Exception as e:
        # print(e)
        if apply_postprocessing:
            hydra_overrides_extra += [
                # dynamically fall back to multi-mask if the single mask is not stable
                "++robo_sam.model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++robo_sam.model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++robo_sam.model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            ]
        config_file_name = config_file.split('/')[-1]
        config_file_name = '.'.join(config_file_name.split('.')[:-1])
        config_file = '/'.join(config_file.split('/')[:-1] + [config_file_name+"_engine.yaml"])
        config_file = "robo_sam/model/segment_anything_2/sam2_configs" + "/" + config_file
        cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
        _load_checkpoint(model, ckpt_path)
    

    if device:
        model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):  
    if True:
        hydra_overrides = [
            "++model._target_=robo_engine.robo_sam.model.segment_anything_2.sam2.sam2_video_predictor.SAM2VideoPredictor",
        ]
        if apply_postprocessing:
            hydra_overrides_extra = hydra_overrides_extra.copy()
            hydra_overrides_extra += [
                "++robo_engine.robo_sam.model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++robo_engine.robo_sam.model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++robo_engine.robo_sam.model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                "++robo_engine.robo_sam.model.binarize_mask_from_pts_for_mem_enc=true",
                "++robo_engine.robo_sam.model.fill_hole_area=8",
            ]
        config_file_name = config_file.split('/')[-1]
        config_file_name = '.'.join(config_file_name.split('.')[:-1])
        config_file = '/'.join(config_file.split('/')[:-1] + [config_file_name+"_engine.yaml"])
        config_file = "robo_sam/model/segment_anything_2/sam2_configs" + "/" + config_file
        hydra_overrides.extend(hydra_overrides_extra)     # config_path = dir(infer_engine.py) + dir(hydra_init) + config_file
        cfg = compose(config_name=config_file, overrides=hydra_overrides)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)  # instantiate_path = dir(package_base) + model._target_, where package_base is the "pip install -e ." directory
        _load_checkpoint(model, ckpt_path)

    if device:
        model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
