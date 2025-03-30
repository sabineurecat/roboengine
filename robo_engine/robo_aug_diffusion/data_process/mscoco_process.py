from __future__ import print_function
from pycocotools.coco import COCO
import os, sys, zipfile
import urllib.request
import shutil
import pdb
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import json
from PIL import Image
from tqdm import tqdm


def coco_photo_background_generation(instances_fp, captions_fp, image_base_dir, save_dir, commit="default"):
    
    coco=COCO(instances_fp)
    coco_caps = COCO(captions_fp)

    img_ids = coco.getImgIds()
    for img_id in tqdm(img_ids, desc=commit):
        img_info = coco.loadImgs(img_id)[0]
        img_file_path = os.path.join(image_base_dir, img_info['file_name'])
        if not os.path.exists(img_file_path):
            print(f"Image file not found: {img_file_path}")
            continue
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        masks = {}
        for ann in anns:
            category_id = ann['category_id']
            category_info = coco.loadCats([category_id])
            category_name = category_info[0]['name']
            mask = coco.annToMask(ann)
            # print(f"{img_id}: ", category_name)
            if category_name not in masks:
                masks[category_name] = mask
                masks[category_name+"_num"] = 1
            else:
                masks[category_name] = np.maximum(masks[category_name], mask)
                masks[category_name+"_num"] = masks[category_name+"_num"] + 1
        
        caption_ids = coco_caps.getAnnIds(imgIds=img_id)
        captions = coco_caps.loadAnns(caption_ids) 
        captions = [caption['caption'].strip().lower() for caption in captions]

        for category_name, mask in masks.items():
            if category_name.endswith("_num"):
                continue
            size_ratio = mask.sum() / (mask.shape[0] * mask.shape[1])
            if size_ratio < 0.1:
                continue
            if masks[category_name+"_num"] > 1:
                continue
            # print(category_name)
            Image.open(img_file_path).save(os.path.join(save_dir, "images", f"{img_id}_{category_name}_{commit}.png"))
            Image.fromarray((mask*255).astype(np.uint8)).save(os.path.join(save_dir, "masks", f"{img_id}_{category_name}_{commit}.png"))
            with open(os.path.join(save_dir, "prompts", f"{img_id}_{category_name}_{commit}.json"), "w") as f:
                json.dump(captions, f)



if __name__ == "__main__":
    base_dir = "/nfs_gaoyang/shared_datasets/MSCOCO"
    save_dir = os.path.join(base_dir, "photo_background_generation")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "prompts"), exist_ok=True)

    instances_fp = os.path.join(base_dir, "train/annotations/instances_train2017.json")
    captions_fp = os.path.join(base_dir, "train/annotations/captions_train2017.json")
    image_base_dir = os.path.join(base_dir, "train/train2017")
    coco_photo_background_generation(instances_fp, captions_fp, image_base_dir, save_dir, commit='train')
    
    instances_fp = os.path.join(base_dir, "val/annotations/instances_val2017.json")
    captions_fp = os.path.join(base_dir, "val/annotations/captions_val2017.json")
    image_base_dir = os.path.join(base_dir, "val/val2017")
    coco_photo_background_generation(instances_fp, captions_fp, image_base_dir, save_dir, commit='val')

    # pdb.set_trace()
    print("Done")
