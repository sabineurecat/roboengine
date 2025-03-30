import os
import pdb
import cv2
import numpy as np
from PIL import Image 

import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from einops import rearrange
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import resize
from transformers import (
    DPTForDepthEstimation,
    DPTImageProcessor,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class StableControlDiffusion:

    def __init__(self, device):
        self.device = device
        self.depth_estimator = DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-hybrid-midas"
        ).to(device)
        self._feature_extractor = DPTImageProcessor.from_pretrained(
            "Intel/dpt-hybrid-midas", do_rescale=False
        )
        controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-normal", torch_dtype=torch.float16
        ).to(device)
        self.inpainter = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16,
            safety_checker=None, revision=None,
        ).to(device)
        self.inpainter.scheduler = UniPCMultistepScheduler.from_config(
            self.inpainter.scheduler.config
        )


    def get_normal_map(self, images):
        device = images.device
        images = self._feature_extractor(
            images=images, return_tensors="pt"
        ).pixel_values.to(device)
        with torch.no_grad():
            depth_map = self.depth_estimator(images).predicted_depth.to(device)
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(512, 512),
            mode="bicubic",
            align_corners=False,
        )
        normal_image = depth_map[0,0].cpu().numpy()

        image_depth = normal_image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)

        # bg_threhold = 0.4
        # pdb.set_trace()
        x = cv2.Sobel(normal_image, cv2.CV_32F, 1, 0, ksize=3)

        y = cv2.Sobel(normal_image, cv2.CV_32F, 0, 1, ksize=3)

        z = np.ones_like(x) * np.pi * 2.0

        normal_image = np.stack([x, y, z], axis=2)
        normal_image /= np.sum(normal_image ** 2.0, axis=2, keepdims=True) ** 0.5
        normal_image = (normal_image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        normal_image = Image.fromarray(normal_image).resize((512,512))

        return normal_image

    
    def gen(self, image_np, prompt, image_normal=None, cond_scale=0.8, num_inference_steps=10, resize_shape=None, disable_resize=False):  # recommended for good generalization

        if image_normal is None:
            image_torch = torch.as_tensor(image_np, device=self.device).permute(2, 0, 1)[None] / 255.0
            image_normal = self.get_normal_map(image_torch)
        if resize_shape is None:
            resize_shape = image_np.shape[:2]

        # pdb.set_trace()
        out = self.inpainter(prompt, image_normal, num_inference_steps=num_inference_steps, controlnet_conditioning_scale=cond_scale).images[0]
        if disable_resize is False:
            out = resize(out, resize_shape, interpolation=Image.BICUBIC)
        return out


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inpainter = StableControlDiffusion(device)

    image_np = np.array(Image.open("/root/project/robo_engine/robot_example.png"))
    prompt = "A clean room."
    inpainted_image = inpainter.gen(image_np, prompt=prompt, cond_scale=0.0, num_inference_steps=20)
    inpainted_image.save("inpainted_image.png")
    print("Inpainting Successful.")
    pdb.set_trace()