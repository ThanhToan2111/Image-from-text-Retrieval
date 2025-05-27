from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
import os

# ======== Setup model và device ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ======== Load dữ liệu đã mã hóa ==========
feature_path = "backend/data/image_features.pt"
path_path = "backend/data/image_paths.npy"

assert os.path.exists(feature_path), f"Không tìm thấy {feature_path}"
assert os.path.exists(path_path), f"Không tìm thấy {path_path}"

image_features = torch.load(feature_path)                
image_paths = np.load(path_path, allow_pickle=True)      

# Normalize image features 1 lần (chuẩn hóa theo L2)
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
image_features_np = image_features.cpu().numpy().astype("float32")

# ======== FAISS index ==========
faiss_index = faiss.IndexFlatL2(image_features_np.shape[1])
faiss_index.add(image_features_np)

# ======== FastAPI ==========
app = FastAPI()

class QueryText(BaseModel):
    text: str
    top_k: int = 5

# ----------- API 1: Dùng FAISS -------------
@app.post("/search")
def search_image_faiss(query: QueryText):
    with torch.no_grad():
        inputs = clip_processor(text=[query.text], return_tensors="pt", padding=True).to(device)
        text_features = clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_np = text_features.cpu().numpy().astype("float32")

        _, indices = faiss_index.search(text_features_np, query.top_k)

    result_paths = [image_paths[i] for i in indices[0]]
    return {"results": result_paths}

# ----------- API 2: Dùng cosine similarity bằng torch -------------
@app.post("/search_cosine")
def search_image_cosine(query: QueryText):
    with torch.no_grad():
        inputs = clip_processor(text=[query.text], return_tensors="pt", padding=True).to(device)
        text_features = clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_feats_device = image_features.to(device)
        similarities = (text_features @ image_feats_device.T).squeeze(0)

        topk = similarities.topk(query.top_k)
        indices = topk.indices.cpu().tolist()

    result_paths = [image_paths[i] for i in indices]
    return {"results": result_paths}
