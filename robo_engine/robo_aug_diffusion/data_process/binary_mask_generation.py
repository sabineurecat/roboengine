from PIL import Image
import cv2
import os
import json
from tqdm import tqdm
from utils import resize_with_padding

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
    "roboturk/EYE_front_rgb": [["robot", "Sawyer"], 24, "train"],
    "stanford_hydra_dataset_converted_externally_to_rlds/EYE_image": [["robot", "Franka"], 16, "train"],
    "taco_play/EYE_rgb_static": [["robot", "Franka"], 11, "train"],
    "tokyo_u_lsmo_converted_externally_to_rlds/EYE_image": [["robot", "xArm"], 10, "train"],
    "toto/EYE_image": [["robot", "Franka"], 8, "train"],
    "ucsd_kitchen_dataset_converted_externally_to_rlds/EYE_image": [["robot", "xArn", "xArm and Robotiq"], 50, "train"],
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
    "stanford_hydra_dataset_converted_externally_to_rlds/EYE_image_val": [["robot", "Franka"], 8, "val"],
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds/EYE_image2": [["robot", "xArm"], 8, "val"],

    ## Test
    "bridge/EYE_image_test": [["robot", "WindowX"], 32, "test"],
    "droid/EYE_image_test": [["Franka", "Franka and Robotiq"], 20, "test"],
    "fractal20220817_data/EYE_view_test": [["robot", "EveryDay Robot"], 16, "test"],
    "nyu_franka_play_dataset_converted_externally_to_rlds/EYE_image_additional_view": [["robot", "Franka"], 8, "test"],
    "rdt/EYE_image_test": [["robot", "WindowX"], 8, "test"],
    "viola/Eye_test": [["robot", "Franka"], 8, "test"],

    ## Zero
    "internet_testbench/Default_view": [["robot"], 45, "zero"],
}


root_dir='/root/project/robo_engine/robo_seg_dataset/data'
output_dir='/root/project/robo_engine/robo_seg_dataset/mask_data_img_resize_padding/data'

train_count=0
val_count=0
test_count=0
for key, value in tqdm(DATA_INFO_DICT.items()):
    if "extra" in key:
        continue
    # if "droid" not in key:
    #     continue
    # else:
    #     multiple=False
    
    if value[-1]=="train":
        output_img_dir=os.path.join(output_dir, "train", "image")
        output_mask_dir=os.path.join(output_dir, "train", "conditioning_image")
        output_text_dir=os.path.join(output_dir, "train", "text")
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        os.makedirs(output_text_dir, exist_ok=True)
        path=os.path.join(root_dir, key)
        for file in os.listdir(path):
            if (file.startswith('img')) and (file.endswith('.png')):
                mask_file=file.replace('img', 'mask')
                json_file=file.replace('img', 'desc').replace('png', 'json')
                image=cv2.imread(os.path.join(path,file))
                mask=cv2.imread(os.path.join(path,mask_file))
                image=resize_with_padding(image, 512)
                mask=resize_with_padding(mask, 512)
                mask=(mask>0)*255
                cv2.imwrite(os.path.join(output_img_dir, str(train_count)+".png"), image)
                cv2.imwrite(os.path.join(output_mask_dir, str(train_count)+".png"), mask)
                
                print(os.path.join(path, json_file))
                with open(os.path.join(path, json_file), 'r') as file:
                    # print(os.path.join(path, json_file))
                    data = json.load(file)
                data=["robot arm "+data_.lower() for data_ in data]

                with open(os.path.join(output_text_dir, str(train_count)+".txt"), 'w') as file:
                    file.write(str(data))
                
                train_count+=1
    if value[-1]=="val":
        output_img_dir=os.path.join(output_dir, "val", "image")
        output_mask_dir=os.path.join(output_dir, "val", "conditioning_image")
        output_text_dir=os.path.join(output_dir, "val", "text")
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        os.makedirs(output_text_dir, exist_ok=True)
        path=os.path.join(root_dir, key)
        for file in os.listdir(path):
            if file.startswith('img'):
                mask_file=file.replace('img', 'mask')
                json_file=file.replace('img', 'desc').replace('png', 'json')
                image=cv2.imread(os.path.join(path,file))
                mask=cv2.imread(os.path.join(path,mask_file))
                image=resize_with_padding(image, 512)
                mask=resize_with_padding(mask, 512)
                mask=(mask>0)*255
                cv2.imwrite(os.path.join(output_img_dir, str(val_count)+".png"), image)
                cv2.imwrite(os.path.join(output_mask_dir, str(val_count)+".png"), mask)
                
                with open(os.path.join(path,json_file), 'r') as file:
                    data = json.load(file)
                    
                data=["robot arm "+data_.lower() for data_ in data]
                with open(os.path.join(output_text_dir, str(val_count)+".txt"), 'w') as file:
                    file.write(str(data))
                    
                val_count+=1
    if value[-1]=="test":
        output_img_dir=os.path.join(output_dir, "test", "image")
        output_mask_dir=os.path.join(output_dir, "test", "conditioning_image")
        output_text_dir=os.path.join(output_dir, "test", "text")
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        os.makedirs(output_text_dir, exist_ok=True)
        path=os.path.join(root_dir, key)
        for file in os.listdir(path):
            if file.startswith('img'):
                mask_file=file.replace('img', 'mask')
                json_file=file.replace('img', 'desc').replace('png', 'json')
                image=cv2.imread(os.path.join(path,file))
                mask=cv2.imread(os.path.join(path,mask_file))
                image=resize_with_padding(image, 512)
                mask=resize_with_padding(mask, 512)
                mask=(mask>0)*255
                cv2.imwrite(os.path.join(output_img_dir, str(test_count)+".png"), image)
                cv2.imwrite(os.path.join(output_mask_dir, str(test_count)+".png"), mask)
                
                with open(os.path.join(path,json_file), 'r') as file:
                    data = json.load(file)
                    
                data=["robot arm "+data_.lower() for data_ in data]
                with open(os.path.join(output_text_dir, str(test_count)+".txt"), 'w') as file:
                    file.write(str(data))
                    
                test_count+=1
                

for key, value in tqdm(DATA_INFO_DICT.items()):
    if "extra" in key:
        continue
    if "droid" not in key:
        continue
    
    if value[-1]=="train":
        output_img_dir=os.path.join(output_dir, "train", "image")
        output_mask_dir=os.path.join(output_dir, "train", "conditioning_image")
        output_text_dir=os.path.join(output_dir, "train", "text")
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        os.makedirs(output_text_dir, exist_ok=True)
        path=os.path.join(root_dir, key)
        for file in os.listdir(path):
            if file.startswith('img'):
                mask_file=file.replace('img', 'mask')
                json_file=file.replace('img', 'desc').replace('png', 'json')
                image=cv2.imread(os.path.join(path,file))
                mask=cv2.imread(os.path.join(path,mask_file))
                image=resize_with_padding(image, 512)
                mask=resize_with_padding(mask, 512)
                mask=(mask>0)*255
                cv2.imwrite(os.path.join(output_img_dir, str(train_count)+".png"), image)
                cv2.imwrite(os.path.join(output_mask_dir, str(train_count)+".png"), mask)
                
                with open(os.path.join(path, json_file), 'r') as file:
                    # print(os.path.join(path, json_file))
                    data = json.load(file)
                    
                data=["robot arm "+data_.lower() for data_ in data]
                with open(os.path.join(output_text_dir, str(train_count)+".txt"), 'w') as file:
                    file.write(str(data))
                
                train_count+=1
    if value[-1]=="val":
        output_img_dir=os.path.join(output_dir, "val", "image")
        output_mask_dir=os.path.join(output_dir, "val", "conditioning_image")
        output_text_dir=os.path.join(output_dir, "val", "text")
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        os.makedirs(output_text_dir, exist_ok=True)
        path=os.path.join(root_dir, key)
        for file in os.listdir(path):
            if file.startswith('img'):
                mask_file=file.replace('img', 'mask')
                json_file=file.replace('img', 'desc').replace('png', 'json')
                image=cv2.imread(os.path.join(path,file))
                mask=cv2.imread(os.path.join(path,mask_file))
                image=resize_with_padding(image, 512)
                mask=resize_with_padding(mask, 512)
                mask=(mask>0)*255
                cv2.imwrite(os.path.join(output_img_dir, str(val_count)+".png"), image)
                cv2.imwrite(os.path.join(output_mask_dir, str(val_count)+".png"), mask)
                
                with open(os.path.join(path,json_file), 'r') as file:
                    data = json.load(file)
                    
                data=["robot arm "+data_.lower() for data_ in data]
                with open(os.path.join(output_text_dir, str(val_count)+".txt"), 'w') as file:
                    file.write(str(data))
                    
                val_count+=1
    if value[-1]=="test":
        output_img_dir=os.path.join(output_dir, "test", "image")
        output_mask_dir=os.path.join(output_dir, "test", "conditioning_image")
        output_text_dir=os.path.join(output_dir, "test", "text")
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        os.makedirs(output_text_dir, exist_ok=True)
        path=os.path.join(root_dir, key)
        for file in os.listdir(path):
            if file.startswith('img'):
                mask_file=file.replace('img', 'mask')
                json_file=file.replace('img', 'desc').replace('png', 'json')
                image=cv2.imread(os.path.join(path,file))
                mask=cv2.imread(os.path.join(path,mask_file))
                image=resize_with_padding(image, 512)
                mask=resize_with_padding(mask, 512)
                mask=(mask>0)*255
                cv2.imwrite(os.path.join(output_img_dir, str(test_count)+".png"), image)
                cv2.imwrite(os.path.join(output_mask_dir, str(test_count)+".png"), mask)
                
                with open(os.path.join(path,json_file), 'r') as file:
                    data = json.load(file)
                    
                data=["robot arm "+data_.lower() for data_ in data]
                with open(os.path.join(output_text_dir, str(test_count)+".txt"), 'w') as file:
                    file.write(str(data))
                    
                test_count+=1
                