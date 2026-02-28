import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import faiss
import open_clip
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

#  CONFIGURACIÓN
MODEL_DIR = Path("./model")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


print(f"Dispositivo: {DEVICE}")

with open(MODEL_DIR / "config.json") as f:
    config = json.load(f)

# CLIP
model, _, preprocess = open_clip.create_model_and_transforms(
    config["clip_model"],            # "ViT-B-32"
    pretrained=config["clip_pretrain"],  # "openai"
    device=DEVICE,
)
checkpoint = torch.load(MODEL_DIR / "best_model.pt", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f" CLIP cargado — epoch {checkpoint.get('epoch','?')} | R@5={checkpoint.get('best_recall_at_5','?'):.3f}")

# FAISS
index = faiss.read_index(str(MODEL_DIR / "product_index.faiss"))
print(f"FAISS — {index.ntotal:,} productos indexados")

# Metadatos del catálogo
meta_parquet = MODEL_DIR / "catalog_meta.parquet"
meta_csv     = MODEL_DIR / "catalog_meta.csv"
if meta_parquet.exists():
    catalog = pd.read_parquet(meta_parquet).reset_index(drop=True)
elif meta_csv.exists():
    catalog = pd.read_csv(meta_csv).reset_index(drop=True)
else:
    raise FileNotFoundError("No se encuentra catalog_meta.parquet ni catalog_meta.csv en ./model/")

print(f" Catálogo — {len(catalog):,} productos")

#  INFERENCIA

@torch.no_grad()
def predict(image_bytes: bytes, top_k: int = 5):

    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Preprocesar y calcular embedding
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    emb    = model.encode_image(tensor)
    emb    = F.normalize(emb, dim=-1).cpu().numpy().astype("float32")

    # Búsqueda FAISS
    scores, indices = index.search(emb, top_k)
    scores, indices = scores[0], indices[0]


    results = []
    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        row = catalog.iloc[idx]
        results.append({
            "rank":             rank,
            "product_asset_id": str(row["product_asset_id"]),
            "description":      str(row.get("product_description", row.get("description", ""))),
            "similarity":       round(float(score), 4),
            "image_url":        str(row["product_image_url"]) if "product_image_url" in row else None,
        })
    return results


#  API

app = FastAPI(title="Bundle Product Recognition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "products_indexed": index.ntotal, "device": DEVICE}

@app.post("/predict_url")
async def predict_url_endpoint(url: str, top_k: int = 5):
    """Recibe una URL de imagen, la descarga y ejecuta inferencia."""
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k debe estar entre 1 y 20")
    try:
        import requests as req
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.zara.com/',
        }
        r = req.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        results = predict(r.content, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error descargando o procesando imagen: {str(e)}")
    return {"results": results, "top_k": top_k}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...), top_k: int = 5):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k debe estar entre 1 y 20")

    image_bytes = await file.read()
    try:
        results = predict(image_bytes, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {str(e)}")

    return {"results": results, "top_k": top_k}

