import os
import cv2
import json
from tqdm import tqdm
from utils import resize_with_padding

path='/nfs_gaoyang/shared_datasets/MSCOCO/photo_background_generation'
output_path='/nfs_gaoyang/shared_datasets/MSCOCO/image_mask_prompt_resize_with_padding'

image_path=os.path.join(path, "images")
mask_path=os.path.join(path, "masks")
prompt_path=os.path.join(path, "prompts")

output_train=os.path.join(output_path, "data", "train")
output_val=os.path.join(output_path, "data", "val")
os.makedirs(output_train, exist_ok=True)
os.makedirs(output_val, exist_ok=True)

train_img_output=os.path.join(output_train, "image")
train_mask_output=os.path.join(output_train, "conditioning_image")
train_text_output=os.path.join(output_train, "text")

val_img_output=os.path.join(output_val, "image")
val_mask_output=os.path.join(output_val, "conditioning_image")
val_text_output=os.path.join(output_val, "text")

os.makedirs(train_img_output, exist_ok=True)
os.makedirs(train_mask_output, exist_ok=True)
os.makedirs(train_text_output, exist_ok=True)

os.makedirs(val_img_output, exist_ok=True)
os.makedirs(val_mask_output, exist_ok=True)
os.makedirs(val_text_output, exist_ok=True)

train_count=0
val_count=0

image_path_list=os.listdir(image_path)
for image in tqdm(image_path_list):
    if "train" in image:
        im=cv2.imread(os.path.join(image_path, image))
        im=resize_with_padding(im, 512)
        cv2.imwrite(os.path.join(train_img_output, str(train_count)+'.png'), im)

        mask=cv2.imread(os.path.join(mask_path, image))
        mask=resize_with_padding(mask, 512)
        cv2.imwrite(os.path.join(train_mask_output, str(train_count)+'.png'), mask)

        prompt_file=image.replace('png', 'json')
        with open(os.path.join(prompt_path, prompt_file), 'r') as file:
            data = json.load(file)
            
        with open(os.path.join(train_text_output, str(train_count)+".txt"), 'w') as file:
            file.write(str(data))
        train_count+=1    
        
    elif "val" in image:
        im=cv2.imread(os.path.join(image_path, image))
        im=resize_with_padding(im, 512)
        cv2.imwrite(os.path.join(val_img_output, str(val_count)+'.png'), im)

        mask=cv2.imread(os.path.join(mask_path, image))
        mask=resize_with_padding(mask, 512)
        cv2.imwrite(os.path.join(val_mask_output, str(val_count)+'.png'), mask)

        prompt_file=image.replace('png', 'json')
        with open(os.path.join(prompt_path, prompt_file), 'r') as file:
            data = json.load(file)
            
        with open(os.path.join(val_text_output, str(val_count)+".txt"), 'w') as file:
            file.write(str(data))
        val_count+=1
