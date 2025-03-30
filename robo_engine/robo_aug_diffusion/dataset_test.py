from datasets import Dataset, Features, Value, Image
import os
import pandas as pd

def load_data_with_images(image_dir, conditioning_image_dir, text_dir):
    # 获取所有文件名，假设文件名相同
    file_names = sorted(os.listdir(image_dir))

    # 构建一个字典来存储数据
    data = {
        "image": [],
        "conditioning_image": [],
        "text": [],
    }

    for file_name in file_names:
        # 构造每种数据的路径
        image_path = os.path.join(image_dir, file_name)
        conditioning_image_path = os.path.join(conditioning_image_dir, file_name)
        text_path = os.path.join(text_dir, file_name.replace('.png', '.txt'))  # 假设文本文件后缀为 .txt

        # 读取文本文件内容
        with open(text_path, "r", encoding="utf-8") as f:
            text_content = f.read().strip()

        # 添加路径信息到字典中
        data["image"].append(image_path)
        data["conditioning_image"].append(conditioning_image_path)
        data["text"].append(text_content)

    # 定义数据集的特征类型
    features = Features({
        "image": Image(),
        "conditioning_image": Image(),
        "text": Value("string"),
    })

    # 加载为 Dataset 对象
    return Dataset.from_dict(data, features=features)

# 加载数据
dataset = load_data_with_images(
    image_dir="/flash2/aml/ycb24_zhust24_suraj24/data/processed_inpainting_robot_data/train/image/",
    conditioning_image_dir="/flash2/aml/ycb24_zhust24_suraj24/data/processed_inpainting_robot_data/train/conditioning_image/",
    text_dir="/flash2/aml/ycb24_zhust24_suraj24/data/processed_inpainting_robot_data/train/text/"
)

# 检查数据集
print(dataset[0])  # 查看第一条数据
dataset = dataset.train_test_split(test_size=10)

dataset.save_to_disk("/flash2/aml/ycb24_zhust24_suraj24/data/processed_inpainting_robot_data_hf")