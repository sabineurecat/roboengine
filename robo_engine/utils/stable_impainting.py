import os
import pdb
import cv2
import numpy as np
from PIL import Image 

import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
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

class StableImpainting:

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
        self.inpainter = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16,
            safety_checker=None, revision=None,
        ).to(device)
        # (chengbo yuan, 2025.1.4) We do find adding little normal control (cond_scale~0.1) improve the augmentation quality a lot. 
        # self.inpainter = StableDiffusionInpaintPipeline.from_pretrained(
        #     "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16,
        #     safety_checker=None,
        # ).to(device)
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

    
    def gen(self, image_np, mask_np, prompt, image_normal=None, cond_scale=0.8, num_inference_steps=10, resize_shape=None, disable_resize=False):  # recommended for good generalization

        if image_normal is None:
            image_torch = torch.as_tensor(image_np, device=self.device).permute(2, 0, 1)[None] / 255.0
            image_normal = self.get_normal_map(image_torch)
        if resize_shape is None:
            resize_shape = image_np.shape[:2]
        
        image_np = cv2.resize(image_np, (512, 512))
        mask_np = cv2.resize(mask_np, (512, 512))
        image_np = image_np/255.0
        mask_np = mask_np*255


        out = self.inpainter(
            prompt,
            num_inference_steps=num_inference_steps,
            image=image_np,
            control_image=image_normal,
            mask_image=mask_np,
            controlnet_conditioning_scale=cond_scale, 
        ).images[0]

        # pdb.set_trace()
        if disable_resize is False:
            out = resize(out, resize_shape, interpolation=Image.BICUBIC)
        return out

    def gen_batch(self, image_np_list, mask_np_list, prompt_list, normal_images_list=None, cond_scale=0.8,num_images_per_prompt=1, num_inference_steps=10, resize_shape=None, disable_resize=False, seed=42):  

        if normal_images_list is None:
            normal_images_list=[]
            for image_np in image_np_list:
                image_torch = torch.as_tensor(image_np, device=self.device).permute(2, 0, 1)[None] / 255.0
                image_normal = self.get_normal_map(image_torch)
                normal_images_list.append(image_normal)

        if resize_shape is None:
            resize_shape = image_np_list[0].shape[:2]
        
        image_np_list = [cv2.resize(image_np, (512, 512)) for image_np in image_np_list]
        mask_np_list = [cv2.resize(mask_np, (512, 512)) for mask_np in mask_np_list]
        mask_np_list = [mask_np*255 for mask_np in mask_np_list]
        image_np_list = [image_np/255.0 for image_np in image_np_list]

        device = self.inpainter.device
        generator = torch.Generator(device=device).manual_seed(seed)
        output_images = self.inpainter(
            prompt_list,
            num_inference_steps=num_inference_steps,
            image=image_np_list,
            control_image=normal_images_list,
            mask_image=mask_np_list,
            num_images_per_prompt=num_images_per_prompt,
            controlnet_conditioning_scale=cond_scale, 
            generator=generator
        ).images
        
        if disable_resize is False:
            output_images = [resize(output_image, resize_shape, interpolation=Image.BICUBIC) for output_image in output_images]
        return output_images    


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inpainter = StableImpainting(device)


    image_np = np.array(Image.open("/root/project/robo_engine/robot_example.png"))
    image = torch.as_tensor(image_np, device=device).permute(2, 0, 1) / 255.0
    mask = torch.zeros_like(image).to(device)
    mask = mask[0]
    mask[0:120, 40:120] = 1.0
    prompt = "A red cup."
    inpainted_image = inpainter.gen(image, mask, prompt=prompt, num_inference_steps=10)
    Image.fromarray((255*mask.detach().cpu().numpy()).astype(np.uint8)).save("mask.png")
    Image.fromarray((255*inpainted_image[0].permute(1,2,0).detach().cpu().numpy()).astype(np.uint8)).save("inpainted_image.png")
    print(inpainted_image.shape)
    print(inpainted_image)
    print("Inpainting Successful.")
    pdb.set_trace()