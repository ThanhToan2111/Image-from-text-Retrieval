import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
from tqdm import tqdm
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
dataset_path = r"C:\Users\dang_\PycharmProjects\CS116\.venv\CBIR_dataset"

image_features = []
image_paths = []

for file in tqdm(os.listdir(dataset_path)):
    if file.lower().endswith((".jpg", ".png")):
        img_path = os.path.join(dataset_path, file)
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            print(f"Bỏ qua ảnh lỗi: {img_path}")
            continue
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            img_feature = clip_model.get_image_features(**inputs)
        image_features.append(img_feature)
        image_paths.append(img_path)

image_features = torch.cat(image_features, dim=0)
image_features = image_features / image_features.norm(dim=-1, keepdim=True)

os.makedirs("backend/data", exist_ok=True)
torch.save(image_features, "backend/data/image_features.pt")
np.save("backend/data/image_paths.npy", np.array(image_paths))

print(" Đã lưu xong embeddings và đường dẫn ảnh.")
