import os
import pdb
import cv2
import numpy as np
from PIL import Image 

import torch
from diffusers import (
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
)
from einops import rearrange
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import resize

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class StableFreeDiffusion:

    def __init__(self, device):
        self.device = device
        self.inpainter = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16,
            safety_checker=None, revision=None,
        ).to(device)
        self.inpainter.scheduler = UniPCMultistepScheduler.from_config(
            self.inpainter.scheduler.config
        )


    def gen(self, image_np, prompt, num_inference_steps=10, resize_shape=None, disable_resize=False):  # recommended for good generalization
        if resize_shape is None:
            resize_shape = image_np.shape[:2]

        # pdb.set_trace()
        out = self.inpainter(prompt, num_inference_steps=num_inference_steps).images[0]
        if disable_resize is False:
            out = resize(out, resize_shape, interpolation=Image.BICUBIC)
        return out


    def gen_batch(self, image_np_list, prompt_list, num_inference_steps=10, resize_shape=None, disable_resize=False, seed=42):  
        if resize_shape is None:
            resize_shapes = [image_np.shape[:2] for image_np in image_np_list]
        else:
            resize_shapes = [resize_shape]*len(prompt_list)    

        # pdb.set_trace()
        device = self.inpainter.device
        generator = torch.Generator(device=device).manual_seed(seed)
        output_images = self.inpainter(prompt_list, num_inference_steps=num_inference_steps, generator=generator).images

        if disable_resize is False:
            output_images = [resize(output_images[i], resize_shapes[i], interpolation=Image.BICUBIC) for i in range(len(prompt_list))]
            
        return output_images   


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inpainter = StableFreeDiffusion(device)

    image_np = np.array(Image.open("/root/project/robo_engine/robo_engine/robot_example.png"))
    prompt = "lab station"
    prompt_list = [prompt]*4
    image_np_list = [image_np]*4
    # inpainted_image = inpainter.gen(image_np, prompt=prompt, cond_scale=0.0, num_inference_steps=20)
    # inpainted_image.save("inpainted_image.png")
    # print("Inpainting Successful.")

    generated_images =  inpainter.gen_batch( image_np_list, prompt_list)
    # for i in range(len(generated_images)):
    #     generated_images[i].save(f"{i}_gen.png")

    pdb.set_trace()