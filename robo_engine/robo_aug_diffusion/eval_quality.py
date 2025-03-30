import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.nn.functional import softmax
from PIL import Image
import os
import numpy as np

def load_images_from_folder(folder):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Inception 模型需要 299x299 输入
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            images.append(img)
    return torch.stack(images)

def compute_inception_score(images, batch_size=4):
    """
    计算 Inception Score (IS)。
    """
    inception_model = models.inception_v3(pretrained=True, transform_input=False).cuda()
    inception_model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch = batch.cuda() if torch.cuda.is_available() else batch
            logits = inception_model(batch)
            preds.append(softmax(logits, dim=1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    py = np.mean(preds, axis=0)
    kl_divergences = []
    for p in preds:
        kl_div = p * (np.log(p + 1e-16) - np.log(py + 1e-16))
        kl_divergences.append(np.sum(kl_div))
    is_score = np.exp(np.mean(kl_divergences))
    return is_score


folder_path = "results_original_img" 
images = load_images_from_folder(folder_path)
inception_score = compute_inception_score(images)
print(f"Inception Score (IS): {inception_score:.2f}")