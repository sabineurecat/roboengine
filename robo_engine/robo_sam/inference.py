import argparse
import os
import sys

import cv2
import numpy as np
import torch
import pdb
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig

from model.evf_sam2 import EvfSam2Model



def parse_args(args):
    parser = argparse.ArgumentParser(description="Robo-EVF infer")
    parser.add_argument("--sam_version_tokenizer", default="YxZhang/evf-sam2")
    parser.add_argument("--sam_version", required=True)
    parser.add_argument("--image_size", default=224, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--sam_load_in_8bit", action="store_true", default=False)
    parser.add_argument("--sam_load_in_4bit", action="store_true", default=False)
    parser.add_argument("--image_path", type=str, default="vis_test/d3.png")
    parser.add_argument("--prompt", type=str, default="[robot]")
    parser.add_argument("--save_dir", type=str, default=".")
    
    return parser.parse_args(args)


def sam_preprocess(
    x: np.ndarray,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024) -> torch.Tensor:
    assert img_size==1024, \
        "both SAM and Effi-SAM receive images of size 1024^2, don't change this setting unless you're sure that your employed model works well with another size."
    x = torch.from_numpy(x).permute(2,0,1).contiguous()
    x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
    x = (x - pixel_mean) / pixel_std
    resize_shape = None
    
    return x, resize_shape

def beit3_preprocess(x: np.ndarray, img_size=224) -> torch.Tensor:
    beit_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=None), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return beit_preprocess(x)


def init_models(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.sam_version_tokenizer,
        padding_side="right",
        use_fast=False,
    )
    torch_dtype = torch.float32
    kwargs = {"torch_dtype": torch_dtype}

    if args.sam_load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    sam_load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
            }
        )
    elif args.sam_load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    sam_load_in_8bit=True,
                ),
            }
        )

    model = EvfSam2Model.from_pretrained(
        args.sam_version, low_cpu_mem_usage=True, **kwargs
    )

    if (not args.sam_load_in_4bit) and (not args.sam_load_in_8bit):
        model = model.cuda()
    model.eval()

    return tokenizer, model


def sam_infer(args, image_np, prompt, tokenizer, model):
    # clarify IO
    # if not os.path.exists(image_path):
    #     print("File not found in {}".format(image_path))
    #     exit()

    # preprocess
    original_size_list = [image_np.shape[:2]]

    image_beit = beit3_preprocess(image_np, args.image_size).to(dtype=model.dtype, device=model.device)

    image_sam, resize_shape = sam_preprocess(image_np)
    image_sam = image_sam.to(dtype=model.dtype, device=model.device)

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=model.device)

    # infer
    pred_mask = model.inference(
        image_sam.unsqueeze(0),
        image_beit.unsqueeze(0),
        input_ids,
        resize_list=[resize_shape],
        original_size_list=original_size_list,
    )
    pred_mask = pred_mask.detach().cpu().numpy()[0]
    pred_mask = pred_mask > 0
    # base_name = '.'.join(image_path.split('/')[-1].split('.')[:-1])
    # save_path = os.path.join(args.save_dir, f"mask_{base_name}_{prompt}.png")
    return pred_mask

def save_visualization(image_np, pred_mask, save_fp):
    masked_image = np.copy(image_np)
    masked_image[~pred_mask] = 255 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title("Pred Mask")
    axes[1].axis('off')
    axes[2].imshow(masked_image)
    axes[2].set_title("Masked Image")
    axes[2].axis('off')
    plt.savefig(save_fp, dpi=150)


def main(args):
    args = parse_args(args)
    # use float16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(args.save_dir, exist_ok=True)
    tokenizer, model = init_models(args)

    image_path = args.image_path 
    prompt = "robot"
    image_path = "/root/project/robo_engine/7.jpg"
    image_np = np.array(Image.open(image_path))
    pred_mask = sam_infer(args, image_np, prompt, tokenizer, model)
    save_visualization(image_np, pred_mask, "vis.png")
    
    pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv[1:])

# CUDA_VISIBLE_DEVICES=0 python inference.py --sam_version /cephfs/shared/yuanchengbo/roboaug/seg_ckpt/mt_1e-5_all_pt30_1229170254 --image_path /root/project/robo_engine/1.jpg