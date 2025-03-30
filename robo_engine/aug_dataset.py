import cv2
import os
from tqdm import tqdm
import numpy as np
import shutil
import argparse
import imageio
from utils.utils import refine_mask, save_np_array_list_video, save_video
from infer_engine import  RoboEngineRobotSegmentation, RoboEngineObjectSegmentation, RoboEngineAugmentation


# import multiprocessing
# from concurrent.futures import ProcessPoolExecutor
# multiprocessing.set_start_method('spawn', force=True)

# def generate_masks(robo_seg, obj_seg, image_np_list, instruction, seg_anchor_frequency):
#     with ProcessPoolExecutor() as executor:
#         robo_future = executor.submit(robo_seg.gen_video, image_np_list, "robot", seg_anchor_frequency)
#         obj_future = executor.submit(obj_seg.gen_video, image_np_list, instruction, seg_anchor_frequency)
#         robo_masks = robo_future.result()
#         obj_masks = obj_future.result()
#     return robo_masks, obj_masks

def aug_video(
    input_video_path, 
    target_videos_dir_name,
    instruction, 
    robo_seg, 
    obj_seg, 
    robo_aug, 
    dataset_seg_masks_dir,
    aug_batch_size,
    num_augmentations,
    seg_anchor_frequency=16,
    tabletop=True,
    num_inference_steps=8,
    verbose=False):

    video_file_name = (input_video_path.split("/")[-1]).split(".")[0]
    video_file_parent_dir = os.path.dirname(input_video_path)
    if dataset_seg_masks_dir is not None:
        seg_masks_file_path = os.path.join(dataset_seg_masks_dir, f"{video_file_name}_aug_0.mp4")
    else: 
        seg_masks_file_path = None

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {input_video_path}.")
        exit()

    image_np_list = []

    fps = 30 
    while True:

        ret, frame = cap.read() 
        if not ret:
            break    
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        image_np = np.array(frame)
        image_np_list.append(image_np)
    cap.release()

    if seg_masks_file_path is not None:
        masks_np_list = []
        print(f"Reading video from:::::: {seg_masks_file_path}")
        if os.path.isfile(seg_masks_file_path):
            cap = cv2.VideoCapture(seg_masks_file_path)

            if cap.isOpened():
                while True:
                    ret, frame = cap.read() 
                    if not ret:
                        break  
                    mask_np = np.array(frame)
                    masks_np_list.append(mask_np)
                cap.release() 
        masks_np_list = [(mask.sum(axis=-1) > 6.0).astype(np.float32) for mask in masks_np_list]

    if len(masks_np_list)!=len(image_np_list):
        masks_np_list=[]        

    if len(masks_np_list)==0:

        robo_masks = robo_seg.gen_video(image_np_list, "robot", seg_anchor_frequency)
        obj_masks = obj_seg.gen_video(image_np_list, instruction, seg_anchor_frequency) 
    
        masks = ((robo_masks+obj_masks)>0).astype(np.float32)
        masks_np_list = [refine_mask(masks[i]) for i in range(len(masks))]


    for i in range(num_augmentations):
        if verbose is True:
            print(f"Processing Augmentation {i} for {input_video_path}")
            print(f"Target Video Directory: {target_videos_dir_name}")

        video_writer = imageio.get_writer(os.path.join(target_videos_dir_name, f"{video_file_name}_aug_{i}.mp4"), fps=fps)
        if robo_aug.aug_method in ["engine", "inpainting", "background"]:
            aug_images = robo_aug.gen_image_batch(image_np_list, masks_np_list, batch_size=aug_batch_size, num_inference_steps=num_inference_steps, tabletop=tabletop, verbose=verbose)

            for frame in aug_images:
                video_writer.append_data(frame)
        else:
            for idx in range(len(image_np_list)):
                image_np = image_np_list[idx]
                mask = masks_np_list[idx]
                aug_image = robo_aug.gen_image(image_np, mask)  
                video_writer.append_data(aug_image)      
        video_writer.close()

    if verbose:
        print(f"Finished processing all frames for {input_video_path}.")   


def augment_dataset(
    source_dataset_dir, 
    target_dataset_dir, 
    instruction,
    robo_seg,
    obj_seg,
    robo_aug, 
    source_dataset_seg_masks_dir= None,
    num_augmentations=5,
    seg_anchor_frequency=16,
    aug_batch_size=16,
    aug_prompt_tabletop=False,
    aug_diffusion_num_inference_steps=5,
    aug_verbose=False,
    div_num_idx=None,
    div_start_idx=None,
    ):

    
    folder_original_name = source_dataset_dir.split("/")[-1]

    source_videos_dir_name = f'{source_dataset_dir}/{folder_original_name}_videos'
    source_actions_dir_name = f'{source_dataset_dir}/{folder_original_name}_action'
    source_ee_dir_name = f'{source_dataset_dir}/{folder_original_name}_ee'
    target_videos_dir_name = f'{target_dataset_dir}/{folder_original_name}_aug_videos'
    target_actions_dir_name = f'{target_dataset_dir}/{folder_original_name}_aug_action'
    target_ee_dir_name = f'{target_dataset_dir}/{folder_original_name}_aug_ee'

    os.makedirs(target_dataset_dir, exist_ok=True)
    os.makedirs(target_videos_dir_name, exist_ok=True)
    os.makedirs(target_actions_dir_name, exist_ok=True)
    os.makedirs(target_ee_dir_name, exist_ok=True)

    if source_dataset_seg_masks_dir is None:
        source_dataset_seg_masks_dir = f'{source_dataset_dir}/{folder_original_name}_masks'
    os.makedirs(source_dataset_seg_masks_dir, exist_ok=True)


    source_videos_fp_list = os.listdir(source_videos_dir_name)
    source_videos_fp_list.sort()
    n_data = len(source_videos_fp_list)

    ## Divide the data into div_num_idx parts and process only the div_start_idx part
    if div_num_idx is None:
        div_num_idx = 1 
    if div_start_idx is None:
        div_start_idx = 0
    n_data_per_div = n_data//div_num_idx
    start_idx = div_start_idx*n_data_per_div
    end_idx = (div_start_idx+1)*n_data_per_div
    if div_start_idx == div_num_idx-1:
        end_idx = n_data   

    # import pdb; pdb.set_trace()

    for idx in tqdm(range(start_idx, end_idx), desc="Augmenting Data"):
        video_full_path = os.path.join(source_videos_dir_name, source_videos_fp_list[idx])
        real_idx = video_full_path.split("/")[-1].split("_")[0]
        action_file_full_path = os.path.join(source_actions_dir_name, str(real_idx)+"_action.npy")
        ee_file_full_path = os.path.join(source_ee_dir_name, str(real_idx)+"_ee.npy")

        # r_idx = int(source_videos_fp_list[idx].split("_")[0])
        # print(r_idx)
        # if r_idx not in [121, 143, 144]:
        #     continue

        aug_video(video_full_path, target_videos_dir_name, instruction, robo_seg, obj_seg, robo_aug,
                  source_dataset_seg_masks_dir,
                  aug_batch_size=aug_batch_size,
                  num_augmentations=num_augmentations,
                  seg_anchor_frequency=seg_anchor_frequency,
                  tabletop=aug_prompt_tabletop, 
                  num_inference_steps=aug_diffusion_num_inference_steps, 
                  verbose=aug_verbose) 

        action_file = str(real_idx)+"_action.npy"
        ee_file = str(real_idx)+"_ee.npy"
        for i in range(num_augmentations):
            action_array_save_filename = action_file.split(".npy")[0]+f"_aug_{i}.npy"
            shutil.copy(action_file_full_path, os.path.join(target_actions_dir_name, action_array_save_filename))
            ee_array_save_filename = ee_file.split(".npy")[0]+f"_aug_{i}.npy"
            shutil.copy(ee_file_full_path, os.path.join(target_ee_dir_name, ee_array_save_filename))



if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Augment dataset')
    parser.add_argument('--to_be_augmented_data_dir', type=str, help='Directory containing the dataset to be augmented')
    parser.add_argument('--instruction', type=str, help='Instruction for the dataset')
    parser.add_argument('--num_augmentations', type=int, help='Number of augmentations to be performed')
    parser.add_argument('--seg_anchor_frequency', type=int, default=16, help='Segmentation anchor frequency')
    parser.add_argument('--aug_batch_size', type=int, default=32, help='Segmentation anchor frequency')
    parser.add_argument('--robo_seg_method', default=["robo_sam_video"], help='Robo segmentation method')
    parser.add_argument('--obj_seg_method', default=["evf_sam_video"], help='Object segmentation method')
    parser.add_argument('--seg_masks_dir', type=str, default=None, help='Directory containing the segmentation masks of dataset to be augmented, if no segmentation masks exist for the dataset, masks will be saved in this path')
    parser.add_argument('--aug_method', default="texture", help='Augmentation method')
    parser.add_argument('--config_dir', default="./cfg", help='Configuration directory')
    parser.add_argument('--start_idx', default=None,type=int, help='Index of starting trajectory to be augmented in the dataset')
    parser.add_argument('--div_num_idx', default=None,type=int, help='Number of trajectories starting from start_index to be augmented in the dataset')
    parser.add_argument('--commit', default="",type=str)
    args = parser.parse_args()

    robo_seg = RoboEngineRobotSegmentation(args.config_dir, args.robo_seg_method)
    obj_seg = RoboEngineObjectSegmentation(args.config_dir, args.obj_seg_method)
    robo_aug = RoboEngineAugmentation(args.config_dir, aug_method=args.aug_method)

    augment_dataset(
        source_dataset_dir=args.to_be_augmented_data_dir, 
        target_dataset_dir=args.to_be_augmented_data_dir+f"_aug_{args.commit}", 
        instruction=args.instruction,
        robo_seg=robo_seg,
        obj_seg=obj_seg,
        robo_aug=robo_aug, 
        source_dataset_seg_masks_dir= args.seg_masks_dir,
        num_augmentations=args.num_augmentations,
        seg_anchor_frequency=args.seg_anchor_frequency,
        aug_batch_size=args.aug_batch_size, 
        aug_prompt_tabletop=True,
        aug_diffusion_num_inference_steps=5,
        aug_verbose=True,
        div_num_idx=args.div_num_idx,
        div_start_idx=args.start_idx,
    )

 
# CUDA_VISIBLE_DEVICES=0 python aug_dataset.py --seg_masks_dir /cephfs/shared/yuanchengbo/roboengine/data/proc_data/task_put_mouse/feb_13_put_mouse_lab_100_aug_black/feb_13_put_mouse_lab_100_aug_videos --to_be_augmented_data_dir /cephfs/shared/yuanchengbo/roboengine/data/proc_data/task_put_mouse/feb_13_put_mouse_lab_100 --instruction "put mouse on the purple pad" --num_augmentations 1  --aug_method "imagenet" --start_idx 0 --div_num_idx 1 --commit imagenet 
# CUDA_VISIBLE_DEVICES=1 python aug_dataset.py --seg_masks_dir /cephfs/shared/yuanchengbo/roboengine/data/proc_data/task_put_mouse/feb_13_put_mouse_lab_100_aug_black/feb_13_put_mouse_lab_100_aug_videos --to_be_augmented_data_dir /cephfs/shared/yuanchengbo/roboengine/data/proc_data/task_put_mouse/feb_13_put_mouse_lab_100 --instruction "put mouse on the purple pad" --num_augmentations 1  --aug_method "texture" --start_idx 0 --div_num_idx 1 --commit texture 

# CUDA_VISIBLE_DEVICES=0 python aug_dataset.py --to_be_augmented_data_dir /cephfs/shared/yuanchengbo/roboengine/data/proc_data/task_put_mouse/feb_13_put_mouse_lab_100 --instruction "put mouse on the purple pad" --num_augmentations 1  --aug_method "black" --start_idx 0 --div_num_idx 4 --commit black
# CUDA_VISIBLE_DEVICES=1 python aug_dataset.py --to_be_augmented_data_dir /cephfs/shared/yuanchengbo/roboengine/data/proc_data/task_put_mouse/feb_13_put_mouse_lab_100 --instruction "put mouse on the purple pad" --num_augmentations 1  --aug_method "black" --start_idx 1 --div_num_idx 4 --commit black
# CUDA_VISIBLE_DEVICES=2 python aug_dataset.py --to_be_augmented_data_dir /cephfs/shared/yuanchengbo/roboengine/data/proc_data/task_put_mouse/feb_13_put_mouse_lab_100 --instruction "put mouse on the purple pad" --num_augmentations 1  --aug_method "black" --start_idx 2 --div_num_idx 4 --commit black
# CUDA_VISIBLE_DEVICES=3 python aug_dataset.py --to_be_augmented_data_dir /cephfs/shared/yuanchengbo/roboengine/data/proc_data/task_put_mouse/feb_13_put_mouse_lab_100 --instruction "put mouse on the purple pad" --num_augmentations 1  --aug_method "black" --start_idx 3 --div_num_idx 4 --commit black