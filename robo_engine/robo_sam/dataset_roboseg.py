import glob
import os
import random

import cv2
import numpy as np
import torch
import pdb
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import transforms
import json
import spacy
from PIL import Image, ImageEnhance
from torchvision.transforms.functional import resize, to_pil_image


DATA_INFO_DICT = {
    ## Train
    "asu_table_top_converted_externally_to_rlds/EYE_image": [["robot", "UR", "UR and Robotiq"], 9, "train"],
    "austin_buds_dataset_converted_externally_to_rlds/EYE_image": [["robot", "Franka"], 8, "train"],
    "austin_sailor_dataset_converted_externally_to_rlds/EYE_image": [["robot", "Franka"], 32, "train"],
    "austin_sirius_dataset_converted_externally_to_rlds/EYE_image": [["robot", "Franka"], 16, "train"],
    "bc_z/EYE_image": [["robot", "EveryDay Robot"], 464, "train"],
    "berkeley_autolab_ur5/EYE_image": [["robot", "UR"], 32, "train"],
    "berkeley_cable_routing/EYE_image": [["robot", "Franka"], 8, "train"],
    "berkeley_cable_routing/EYE_top_image": [["robot", "Franka"], 6, "train"],
    "berkeley_fanuc_manipulation/EYE_image": [["robot", "FANUC Mate"], 231, "train"],
    "bridge/EYE_image_0": [["robot", "WindowX"], 655, "train"],
    "bridge/EYE_image_1": [["robot", "WindowX"], 336, "train"],
    "bridge/EYE_image_2": [["robot", "WindowX"], 320, "train"],
    "cmu_franka_exploration_dataset_converted_externally_to_rlds/EYE_highres_image": [["robot", "Franka"], 6, "train"],
    "cmu_stretch/EYE_image": [["robot", "Hello Robot"], 40, "train"],
    "dlr_edan_shared_control_converted_externally_to_rlds/EYE_image": [["robot", "Light-Weight Robot "], 48, "train"],
    "droid/EYE_image": [["robot", "Franka", "Franka and Robotiq"], 324, "train"],
    "droid_franka_robotiq/EYE_image": [["Franka", "Franka and Robotiq"], 37, "train"],
    "droid_multirobot/EYE_image": [["robot", "all robots"], 37, "train"],
    "extra_sawyer/EYE_image": [["Sawyer"], 8, "train"],
    "extra_tiago/EYE_image": [["Tiago"], 2, "train"],
    "extra_ur/EYE_image": [["UR"], 9, "train"],
    "extra_windowx/EYE_image": [["WindowX"], 15, "train"],
    "extra_xarm/EYE_image": [["xArm"], 3, "train"],
    "fractal20220817_data/EYE_image": [["robot", "EveryDay Robot"], 304, "train"],
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds/EYE_image": [["robot", "Franka"], 40, "train"],
    "jaco_play/EYE_image": [["robot", "Jaco"], 88, "train"],
    "kuka/EYE_image": [["robot", "KUKA"], 6, "train"],
    "language_table/EYE_rgb": [["robot", "xArm"], 8, "train"],
    "nyu_franka_play_dataset_converted_externally_to_rlds/EYE_image": [["robot", "Franka"], 8, "train"],
    "nyu_rot_dataset_converted_externally_to_rlds/EYE_image": [["robot", "xArm"], 45, "train"],
    "rdt/EYE_image": [["robot", "WindowX"], 113, "train"],
    "rekep/Image_View": [["robot", "Franka", "Franka and UMI"], 24, "train"],
    "roboexp/Image_view": [["robot", "xArm", "xArm and UMI"], 20, "train"],
    "roboturk/EYE_front_rgb": [["robot", "Sawyer"], 24, "train"],
    "stanford_hydra_dataset_converted_externally_to_rlds/EYE_image": [["robot", "Franka"], 16, "train"],
    "taco_play/EYE_rgb_static": [["robot", "Franka"], 11, "train"],
    "tokyo_u_lsmo_converted_externally_to_rlds/EYE_image": [["robot", "xArm"], 10, "train"],
    "toto/EYE_image": [["robot", "Franka"], 8, "train"],
    "ucsd_kitchen_dataset_converted_externally_to_rlds/EYE_image": [["robot", "xArn", "xArm and Robotiq"], 50, "train"],
    "umi/Image_View": [["robot", "UR", "UR and UMI"], 18, "train"],
    "utaustin_mutex/EYE_image": [["robot", "Franka"], 56, "train"],
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds/EYE_image": [["robot", "PR"], 8, "train"],
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds/EYE_image": [["robot", "PR"], 18, "train"],
    "utokyo_xarm_bimanual_converted_externally_to_rlds/EYE_image": [["robot", "xArm"], 16, "train"],
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds/EYE_image": [["robot", "xArm"], 8, "train"],
    "viola/Eye_agentview_rgb": [["robot", "Franka"], 16, "train"],
    
    ## Val
    "bc_z/EYE_image_val": [["robot", "EveryDay Robot"], 11, "val"],
    "berkeley_fanuc_manipulation/EYE_image_val": [["robot", "FANUC Mate"], 9, "val"],
    "bridge/EYE_image_val": [["robot", "WindowX"], 24, "val"],
    "droid/EYE_image_val": [["Franka", "Franka and Robotiq"], 20, "val"],
    "rdt/EYE_image_val": [["robot", "WindowX"], 8, "val"],
    "rekep/Image_View_val": [["robot", "Franka", "Franka and UMI"], 8, "val"],
    "stanford_hydra_dataset_converted_externally_to_rlds/EYE_image_val": [["robot", "Franka"], 8, "val"],
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds/EYE_image2": [["robot", "xArm"], 8, "val"],

    ## Test
    "bridge/EYE_image_test": [["robot", "WindowX"], 32, "test"],
    "droid/EYE_image_test": [["Franka", "Franka and Robotiq"], 20, "test"],
    "fractal20220817_data/EYE_view_test": [["robot", "EveryDay Robot"], 16, "test"],
    "nyu_franka_play_dataset_converted_externally_to_rlds/EYE_image_additional_view": [["robot", "Franka"], 8, "test"],
    "rdt/EYE_image_test": [["robot", "WindowX"], 8, "test"],
    "roboexp/Image_view_test": [["robot", "xArm", "xArm and UMI"], 5, "test"],
    "viola/Eye_test": [["robot", "Franka"], 8, "test"],

    ## Zero
    "internet_testbench/Default_view": [["robot"], 45, "zero"],
}



def extract_noun_phrases_with_adjectives(text, nlp):
    doc = nlp(text)
    noun_phrases = []
    for nph in doc.noun_chunks:
        adjectives = [token.text for token in nph if token.pos_ == "ADJ"]
        filtered_nph = [token.text for token in nph if token.pos_ != "PRON"]
        if filtered_nph:
            noun_phrases.append(" ".join(filtered_nph))
        # if adjectives:
        #     noun_phrases.append((nph.text, adjectives))
        # else:
        #     noun_phrases.append((nph.text, None))
    
    return noun_phrases


class Resize:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        return np.array(resize(to_pil_image(image), (self.target_length, self.target_length), antialias=None))


def apply_random_crop(image, mask, max_crop_size_ratio=0.6):
    h, w = image.shape[:2]
    crop_ratio = random.uniform(max_crop_size_ratio, 1.0)
    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    top = random.randint(0, max(0, h - crop_h))
    left = random.randint(0, max(0, w - crop_w))
    cropped_image = image[top:top + crop_h, left:left + crop_w]
    cropped_mask = mask[top:top + crop_h, left:left + crop_w]
    return cropped_image, cropped_mask


def apply_color_jitter(image, brightness=0.3, contrast=0.3, saturation=0.3):
    pil_image = Image.fromarray(image)
    if brightness > 0:
        enhancer = ImageEnhance.Brightness(pil_image)
        factor = 1 + random.uniform(-brightness, brightness)
        pil_image = enhancer.enhance(factor)
    if contrast > 0:
        enhancer = ImageEnhance.Contrast(pil_image)
        factor = 1 + random.uniform(-contrast, contrast)
        pil_image = enhancer.enhance(factor)
    if saturation > 0:
        enhancer = ImageEnhance.Color(pil_image)
        factor = 1 + random.uniform(-saturation, saturation)
        pil_image = enhancer.enhance(factor)

    return np.array(pil_image)


def apply_random_flip(image, mask, horizontal=True, vertical=False):
    if horizontal and random.random() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    return image, mask


def apply_random_rotation(image, mask, max_angle=30):
    angle = random.uniform(-max_angle, max_angle)
    rotated_image = Image.fromarray(image).rotate(angle, resample=Image.BICUBIC, fillcolor=(0, 0, 0))
    rotated_mask = Image.fromarray(mask).rotate(angle, resample=Image.NEAREST, fillcolor=0)  # Use NEAREST for segmentation mask
    return np.array(rotated_image), np.array(rotated_mask)


def deit_3_augmentation(image):
    def apply_grayscale(image):
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
        return grayscale_image

    def apply_solarization(image, threshold=128):
        pil_image = Image.fromarray(image)
        solarized_image = pil_image.point(lambda p: p if p < threshold else 255 - p)
        return np.array(solarized_image)

    def apply_gaussian_blur(image, kernel_size=(3, 3), sigma=1):
        blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
        return blurred_image
    
    choice = random.choice([0, 1, 2, 3])
    if choice == 0:
        image = apply_grayscale(image)
    elif choice == 1:
        image = apply_solarization(image)
    elif choice == 2:
        image = apply_gaussian_blur(image)
    else:
        pass
    return image


def apply_base_augmentation(image, raw_mask):
    image, raw_mask = apply_random_crop(image, raw_mask)
    image = apply_color_jitter(image)
    image = deit_3_augmentation(image)
    image, raw_mask = apply_random_flip(image, raw_mask)
    image, raw_mask = apply_random_rotation(image, raw_mask)
    return image, raw_mask


def collate_fn(
    batch, tokenizer=None, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_evf_list = []
    masks_list = []
    label_list = []
    resize_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        image_path,
        images,
        images_evf,
        masks,
        label,
        resize,
        sampled_classes,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_evf_list.append(images_evf)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        sampled_classes_list.extend(sampled_classes)
        cnt += len(sampled_classes)
        offset_list.append(cnt)
        inferences.append(inference)

    try:
        input_ids = [
            tokenizer(prompt, return_tensors="pt").input_ids[0]
            for prompt in sampled_classes_list
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
    # TinyCLIP
    except TypeError:
        input_ids = [
            tokenizer(prompt) for prompt in sampled_classes_list
        ]
        input_ids = torch.cat(input_ids, dim=0)

    
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_evf": torch.stack(images_evf_list, dim=0),
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "sampled_classes_list": sampled_classes_list,  # not used
        "inference": inferences[0],
    }


class RoboSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        mode,
        split,
        base_dataset_dir,
        image_size=224,
        transform=Resize(1024),
        renew_meta=False,
        apply_base_augmentation=False,
    ):  
        self.mode = mode
        self.split = split
        self.base_dataset_dir = base_dataset_dir
        if self.mode in ['object', 'task', "mix"]:
            self.nlp = spacy.load("en_core_web_sm")

        if os.path.exists(f"metadata_{split}_{mode}.json") and (renew_meta is False):
            with open(f"metadata_{split}_{mode}.json", "r") as f:
                meta_list = json.load(f)
        else:
            meta_list = []
            if mode == "mix":
                mode_list = ['robot', 'object', 'task']
                for mmode in mode_list:
                    meta_list = meta_list + self.get_mode_meta(mmode, base_dataset_dir, split)
            else:
                meta_list = meta_list + self.get_mode_meta(mode, base_dataset_dir, split)
            random.shuffle(meta_list)
            with open(f"metadata_{split}.json", "w") as f:
                json.dump(meta_list, f)
        
        print(f"Split[{split}]: {len(meta_list)} prompted-images.")
        self.meta_list = meta_list
        self.transform = transform
        self.image_preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), interpolation=3, antialias=None), 
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.apply_base_augmentation = apply_base_augmentation


    def get_mode_meta(self, mode, base_dataset_dir, split):

        meta_list = []
        for robo_dataset in tqdm(os.listdir(os.path.join(base_dataset_dir))):
            for robo_view in os.listdir(os.path.join(base_dataset_dir, robo_dataset)):
                if DATA_INFO_DICT[robo_dataset+"/"+robo_view][2] != split:
                    continue
                # if robo_dataset+"/"+robo_view not in DATA_INFO_DICT:
                #     continue
                robo_prompt_list = []

                # pdb.set_trace()
                robo_prompt_list = DATA_INFO_DICT[robo_dataset+"/"+robo_view][0]
                if split == "train":
                    if "robot" in robo_prompt_list:
                        robo_prompt_list = robo_prompt_list + ["robot"] * (len(robo_prompt_list) - 2)   # 50%, 50%
                    else:
                        robo_prompt_list = 2 * robo_prompt_list   # virtual 50%, 50%
                    if ("droid" in robo_dataset) or ("extra" in robo_dataset) or (robo_dataset in ['umi', 'rekep', 'roboexp']):
                        robo_prompt_list = robo_prompt_list + 2 * ["robot"]   # more attention on DROID datasets
                else:
                    robo_prompt_list = ["robot"]  # only evaluate general robot prompt
                robo_prompt_list = [x.strip().lower()+"." for x in robo_prompt_list]

                base_dir = os.path.join(base_dataset_dir, robo_dataset, robo_view)
                img_list = os.listdir(base_dir)
                name_list = [x for x in img_list if x.startswith("img_")]
                name_list = [".".join(x[4:].split(".")[:-1]) for x in name_list]
                    
                for name in name_list:
                    insta_robo_prompt_list = robo_prompt_list.copy()
                    if (mode == "object") or (mode == "task"):
                        text = ' '.join(name.split('_')[3:]).strip().lower()
                        noun_list = extract_noun_phrases_with_adjectives(text, self.nlp)
                        if mode == "object":
                            if len(noun_list) > 0:
                                noun_list = ". ".join(noun_list).strip().lower() + "."
                                insta_robo_prompt_list = [noun_list for x in insta_robo_prompt_list]
                            else:
                                insta_robo_prompt_list = ["robot manipulated objects."] * len(insta_robo_prompt_list)
                        elif mode == "task":
                            if len(noun_list) > 0:
                                noun_list = ". ".join(noun_list).strip().lower() + "."
                                insta_robo_prompt_list = [(x + " " + noun_list).strip() for x in insta_robo_prompt_list]
                    for robot_prompt in insta_robo_prompt_list:      
                        meta = {
                            "mode": mode,
                            "image": os.path.join(base_dir, "img_"+name+".png"),
                            "raw_mask": os.path.join(base_dir, "mask_"+name+".png"),
                            "instruction": f"[{mode}] " + robot_prompt,
                            # "instruction": robot_prompt,
                        }
                        meta_list.append(meta)
        return meta_list



    def __len__(self):
        return len(self.meta_list)


    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        return x


    def __getitem__(self, idx):

        meta = self.meta_list[idx]
        image_path = meta['image']
        raw_mask_path = meta['raw_mask']
        instruction = meta['instruction']

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_mask = np.array(Image.open(raw_mask_path))   # (h, w, 5)  

        if self.apply_base_augmentation is True:
            image, raw_mask = apply_base_augmentation(image, raw_mask)

        if meta['mode'] == 'robot':
            masks = [((raw_mask == 1.0) | (raw_mask == 2.0)).astype(np.float16)]
        elif meta['mode'] == 'object':
            masks = [((raw_mask == 3.0)).astype(np.float16)]
        elif meta['mode'] == 'task':
            masks = [((raw_mask == 1.0) | (raw_mask == 2.0) | (raw_mask == 3.0)).astype(np.float16)]
        else:
            raise ValueError(f"Invalid mode [{mode}] when loading data.")
        sents = [instruction]

        # preprocess image for evf
        image_evf = self.image_preprocessor(image)

        # preprocess image for sam
        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if not isinstance(masks, torch.Tensor):
            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image,
            image_evf,
            masks,
            labels,
            resize,
            sents,
            inference,
        )        
