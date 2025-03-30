import numpy as np
import os
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
import pdb
from .pipeline_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline
from PIL import Image, ImageOps
import requests
from io import BytesIO
from transformers import AutoTokenizer, PretrainedConfig
import torch
# from transparent_background import Remover


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def get_sd_pipeline(sd_base_dir, controlnet_dir, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(
        sd_base_dir, subfolder="tokenizer", use_fast=False,
    )
    controlnet = ControlNetModel.from_pretrained(controlnet_dir)

    sd_inpainting_model_name = sd_base_dir
    text_encoder_cls = import_model_class_from_model_name_or_path(sd_inpainting_model_name, None)

    # noise_scheduler = DDPMScheduler.from_pretrained(sd_inpainting_model_name, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        sd_inpainting_model_name, subfolder="text_encoder", revision=None
    )
    vae = AutoencoderKL.from_pretrained(sd_inpainting_model_name, subfolder="vae", revision=None)
    unet = UNet2DConditionModel.from_pretrained(
        sd_inpainting_model_name, subfolder="unet", revision=None
    )

    weight_dtype = torch.float32
    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        sd_inpainting_model_name,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=None,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def resize_with_padding(img, expected_size):
    width, height = img.size
    if width > height:
        img_resized = img.resize((expected_size[0], int(height * expected_size[0] / width)))
    else:
        img_resized = img.resize((int(width * expected_size[1] / height), expected_size[1]))
    resized_width, resized_height = img_resized.size
    img_padded = ImageOps.pad(img_resized, expected_size, color=(0, 0, 0), centering=(0.5, 0.5))
    padding_left = (expected_size[0] - resized_width) // 2
    padding_top = (expected_size[1] - resized_height) // 2
    image_org_pos = (padding_left, padding_top, padding_left + resized_width, padding_top + resized_height)
    return img_padded, image_org_pos

def recover_resize_with_padding(img_padded, image_org_pos, target_size):
    left, top, right, bottom = image_org_pos
    cropped_img = img_padded.crop((left, top, right, bottom))
    img = cropped_img.resize(target_size)
    return img


def infer_image(image, mask, prompt, pipeline, 
                cond_scale=1.0, 
                num_images_per_prompt=1,
                use_fg_remover=False, 
                num_inference_steps=10,
                seed=42):
    # img: PIL image
    # mask: np.array, range [0, 1], 1 for foreground, 0 for background
    org_size = image.size
    mask = 1.0 - mask
    mask = Image.fromarray((np.stack([mask]*3, axis=-1) * 255).astype('uint8'))
    img_padded, image_org_pos = resize_with_padding(image, (512, 512))
    mask_padded, _ = resize_with_padding(mask, (512, 512))
    device = pipeline.device
    generator = torch.Generator(device=device).manual_seed(seed)

    controlnet_image = pipeline(
        prompt=prompt, image=img_padded, mask_image=mask_padded, control_image=mask_padded, 
        num_images_per_prompt=num_images_per_prompt, generator=generator, num_inference_steps=num_inference_steps, guess_mode=False, controlnet_conditioning_scale=cond_scale
    ).images    
    controlnet_image_rec = [recover_resize_with_padding(img, image_org_pos, org_size) for img in controlnet_image]
    return controlnet_image_rec


def infer_image_batch(image_list, mask_list, prompt_list, pipeline, 
                      cond_scale=1.0, 
                      num_images_per_prompt=1,
                      use_fg_remover=False, 
                      num_inference_steps=10,
                      seed=42):
    # img: PIL image
    # mask: np.array, range [0, 1], 1 for foreground, 0 for background
    org_size = image_list[0].size
    mask_list = [1.0 - mask for mask in mask_list]
    mask_list = [Image.fromarray((np.stack([mask]*3, axis=-1) * 255).astype('uint8')) for mask in mask_list]
    img_padded, image_org_pos = resize_with_padding(image_list[0], (512, 512))
    img_padded_list = [resize_with_padding(img, (512, 512))[0] for img in image_list]
    mask_padded_list = [resize_with_padding(mask, (512, 512))[0] for mask in mask_list]
    device = pipeline.device
    generator = torch.Generator(device=device).manual_seed(seed)

    controlnet_image = pipeline(
        prompt=prompt_list, image=img_padded_list, mask_image=mask_padded_list, control_image=mask_padded_list, 
        num_images_per_prompt=num_images_per_prompt, generator=generator, num_inference_steps=num_inference_steps, guess_mode=False, controlnet_conditioning_scale=cond_scale
    ).images    
    controlnet_image_rec = [recover_resize_with_padding(img, image_org_pos, org_size) for img in controlnet_image]
    return controlnet_image_rec


if __name__ == "__main__":

    sd_base_dir = "/cephfs/shared/yuanchengbo/roboaug/diffusion_ckpt/stabilityai/stable-diffusion-2-inpainting"
    controlnet_dir = "/cephfs/shared/yuanchengbo/roboaug/diffusion_ckpt/controlnet-model-mix-data-resize-with-padding"
    pipeline = get_sd_pipeline(sd_base_dir, controlnet_dir, device='cuda')
    
    image_path = "/root/project/robo_engine/robo_engine/robot_example_2.png"
    mask_path = "/root/project/robo_engine/robo_engine/mask.png"
    image = Image.open(image_path)
    mask = np.array(Image.open(mask_path).convert('L'))
    mask = (mask > 0).astype(np.float32)
    prompt = "robot arm laboratory with tubes and beakers."

    image = [image] * 3
    mask = [mask] * 3
    prompt = [prompt] * 3
    
    # prompt = "robot arm clean laboratory without any objects."
    aug_images = infer_image_batch(image, mask, prompt, pipeline, cond_scale=1.0, num_images_per_prompt=1, use_fg_remover=False, seed=42)    
    
    pdb.set_trace()
    x = 0
    aug_images[x].save(f"{x}_{prompt}.png")