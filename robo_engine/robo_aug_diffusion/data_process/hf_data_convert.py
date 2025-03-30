from datasets import DatasetDict, Dataset, Features, Image as HFImage, Value
import os
from tqdm import tqdm

dataset_root = "/root/project/robo_engine/robo_seg_dataset/mask_data_img_resize_padding"
# dataset_root = "/nfs_gaoyang/shared_datasets/MSCOCO/image_mask_prompt_resize_with_padding"
splits = ["train", "val", "test"]

def load_dataset(split):
    data_dir = os.path.join(dataset_root, "data", split)
    image_dir = os.path.join(data_dir, "image")
    conditioning_image_dir = os.path.join(data_dir, "conditioning_image")
    text_dir = os.path.join(data_dir, "text")

    data = []
    for filename in tqdm(sorted(os.listdir(image_dir))):
        image_path = os.path.join(image_dir, filename)
        conditioning_image_path = os.path.join(conditioning_image_dir, filename)
        text_path = os.path.join(text_dir, filename.replace('.png', '.txt'))

        if os.path.exists(image_path) and os.path.exists(conditioning_image_path) and os.path.exists(text_path):
            data.append({
                "image": image_path,
                "conditioning_image": conditioning_image_path,
                "text": open(text_path, 'r', encoding='utf-8').read()
            })
    return data

features = Features({
    "image": HFImage(),
    "conditioning_image": HFImage(),
    "text": Value("string")
})

dataset_dict = DatasetDict({
    split: Dataset.from_list(load_dataset(split)).cast(features) for split in splits
})

# dataset_dict.save_to_disk("/nfs_gaoyang/shared_datasets/MSCOCO/image_mask_prompt_resize_with_padding_hf")
dataset_dict.save_to_disk("/root/project/robo_engine/robo_seg_dataset/mask_data_hf")

print(dataset_dict)
