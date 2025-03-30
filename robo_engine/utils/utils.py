import os
import numpy as np
import torch
import pdb
import matplotlib.pyplot as plt
from PIL import Image
import cv2  # type: ignore
import imageio



def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
):
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    correct_holes_array = np.full_like(mask, correct_holes, dtype=bool)
    # Perform the bitwise XOR operation
    working_mask = (correct_holes_array ^ mask).astype(np.uint8)
    # working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        # return mask, False
        return mask
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask


def refine_mask(mask, min_mask_region_area=25):
    mask = mask > 0
    mask = remove_small_regions(mask, min_mask_region_area, mode="holes")
    mask = remove_small_regions(mask, min_mask_region_area, mode="islands")
    mask = mask.astype(np.float32)
    return mask


def vis_sam2_anything(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


def image_crop_resize(image_pil, target_shape):
    background = image_pil
    image_np = np.zeros(target_shape, dtype=np.uint8)

    # 计算目标长宽比
    target_ratio = image_np.shape[1] / image_np.shape[0]
    width, height = background.size
    img_ratio = width / height
    if img_ratio > target_ratio:
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        right = left + new_width
        top = 0
        bottom = height
        background = background.crop((left, top, right, bottom))
    else:
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        bottom = top + new_height
        left = 0
        right = width
        background = background.crop((left, top, right, bottom))
    target_sp = (image_np.shape[1], image_np.shape[0])
    background = background.resize(target_sp, Image.Resampling.LANCZOS)
    return background


def save_np_array_list_video(np_array_list, full_save_path):

    output_fps = 30  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_shape = np_array_list[0].shape

    out = cv2.VideoWriter(full_save_path, fourcc, output_fps, frame_shape)
    for np_array in np_array_list:
        out.write(np_array) 
    out.release()
    print(f"Video saved to :::{full_save_path}")

def save_video(save_fp, masks):
    video_writer = imageio.get_writer(save_fp, fps=30)
    for frame in masks:
        video_writer.append_data(np.uint8(frame) * 255)
    video_writer.close()
