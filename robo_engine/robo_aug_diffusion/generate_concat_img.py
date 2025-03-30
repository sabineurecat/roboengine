import os
from PIL import Image

def create_horizontal_strip_with_groups(folder_path, output_folder, image_groups, gap_within=10, gap_between=30, output_size=(256, 256)):
    """
    拼接每个子文件夹中的图片为 1x8 的横向长图，并设置组内和组间的间距。

    Args:
        folder_path (str): 主文件夹路径，包含子文件夹。
        output_folder (str): 保存输出图像的文件夹。
        image_groups (list of list): 图片文件分组的名称列表。
        gap_within (int): 组内图片之间的间隙宽度。
        gap_between (int): 组与组之间的间隙宽度。
        output_size (tuple): 每张图片统一调整的大小 (宽, 高)。
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历子文件夹
    subfolders = [os.path.join(folder_path, sf) for sf in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, sf))]
    
    for i, subfolder in enumerate(subfolders):
        images = []

        # 遍历每组图片
        for group in image_groups:
            for image_name in group:
                image_path = os.path.join(subfolder, image_name)
                if os.path.exists(image_path):
                    img = Image.open(image_path).convert("RGB")
                    img = img.resize(output_size)
                    images.append(img)
                else:
                    print(f"Warning: {image_name} not found in {subfolder}")
                    # 创建空白图片占位
                    placeholder = Image.new("RGB", output_size, color=(255, 255, 255))
                    images.append(placeholder)
            # 添加组间空白间隙
            if group != image_groups[-1]:  # 如果不是最后一组，添加大间隙
                gap = Image.new("RGB", (gap_between, output_size[1]), color=(255, 255, 255))
                images.append(gap)

        # 添加组内图片的间隙
        total_width = sum(img.width for img in images) + gap_within * (len(images) - len(image_groups) - 1)
        strip_height = output_size[1]
        strip = Image.new("RGB", (total_width, strip_height), color=(255, 255, 255))
        
        # 拼接图片
        x_offset = 0
        for img in images:
            strip.paste(img, (x_offset, 0))
            x_offset += img.width + (gap_within if img != images[-1] and img.width != gap_between else 0)
        
        # 保存拼接图
        output_path = os.path.join(output_folder, f"strip_{i + 1}.png")
        strip.save(output_path)
        print(f"Saved: {output_path}")

# 配置路径和参数
folder_path = "result_example"  # 主文件夹路径
output_folder = "concat_example"  # 输出文件夹路径
image_groups = [
    ["img.png", "mask.png"],  # 第一组
    ["ft_0.png", "ft_1.png", "ft_2.png"],  # 第二组
    ["ori_0.png", "ori_1.png", "ori_2.png"]  # 第三组
]
gap_within = 10  # 组内图片间隙
gap_between = 40  # 组间图片间隙
output_size = (256, 256)  # 每张图片的大小

create_horizontal_strip_with_groups(folder_path, output_folder, image_groups, gap_within, gap_between, output_size)