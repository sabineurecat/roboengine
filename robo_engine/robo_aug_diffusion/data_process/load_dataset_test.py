from datasets import load_from_disk

# 加载数据集
loaded_dataset = load_from_disk("/flash2/aml/ycb24_zhust24_suraj24/data/processed_inpainting_robot_data_hf")

print(loaded_dataset["train"].column_names)