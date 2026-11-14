import logging
import os
import threading
import time

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor
from ultralytics import YOLO
from ultralytics.utils import LOGGER

from common import BANK_FILE, DATA_DIR, HEIGHT, ROOT_DIR, TEST_PHOTO_DIR, WIDTH, YOLO_FILE, cut_photo, read_bytes

FINGERPRINT_FILE = os.path.join(DATA_DIR, "fingerprints.npz")
FINGERPRINT_SIZE = (58, 38)
SHRINK = 2
LAYERS = ["layer2", "layer3"]
NEIGHBOURHOOD = 3
CHUNK_ROWS = 4096
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 928
MIN_CONFIDENCE = 0.0
MAX_BOXES = 20
WARM_UP = 2

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
LOGGER.setLevel(logging.WARNING)


def fingerprint(crop, size):
    return np.asarray(crop.convert("L").resize(size, Image.BOX), dtype=np.float32)


@st.cache_data(show_spinner=False)
def load_fingerprints(new_photo_files):
    with np.load(FINGERPRINT_FILE) as saved:
        prints = [saved["prints"].astype(np.float32)]
        names, parts, limit = saved["names"].tolist(), saved["parts"].tolist(), float(saved["limit"])
    for file in new_photo_files:
        crop = cut_photo(read_bytes(os.path.join(TEST_PHOTO_DIR, file)))["crop"]
        prints.append(fingerprint(crop, size=FINGERPRINT_SIZE)[None])
        names.append(file.rsplit(".", 1)[0])
        parts.append("new")
    return np.concatenate(prints), names, parts, limit


def find_match(crop, new_photo_files):
    prints, names, parts, limit = load_fingerprints(new_photo_files)
    distances = np.abs(prints - fingerprint(crop, size=FINGERPRINT_SIZE)).mean(axis=(1, 2))
    best = int(distances.argmin())
    if distances[best] > limit:
        return None
    return {"name": names[best], "part": parts[best], "distance": float(distances[best])}


def models_missing():
    return [os.path.relpath(path, ROOT_DIR).replace(os.sep, "/") for path in [BANK_FILE, YOLO_FILE] if not os.path.exists(path)]


def load_picture(crop, shrink, device):
    small = crop.resize((crop.width // shrink, crop.height // shrink), Image.BOX)
    picture = torch.from_numpy(np.asarray(small, dtype=np.float32) / 255).permute(2, 0, 1)[None]
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((picture - mean) / std).to(device)


@torch.no_grad()
def patch_features(extractor, batch, layers, neighbourhood):
    maps = extractor(batch)
    grid_size = maps[layers[0]].shape[-2:]
    pooled = []
    for name in layers:
        averaged = F.avg_pool2d(maps[name], neighbourhood, stride=1, padding=neighbourhood // 2)
        pooled.append(F.interpolate(averaged, size=grid_size, mode="bilinear", align_corners=False))
    return torch.cat(pooled, dim=1).permute(0, 2, 3, 1)


@torch.no_grad()
def nearest_distances(queries, bank, chunk_rows):
    distances = []
    for start in range(0, len(queries), chunk_rows):
        distances.append(torch.cdist(queries[start:start + chunk_rows], bank).min(dim=1).values)
    return torch.cat(distances)


def patchcore_grid(models, crop):
    batch = load_picture(crop, shrink=SHRINK, device=models["device"])
    features = patch_features(models["extractor"], batch, layers=LAYERS, neighbourhood=NEIGHBOURHOOD)
    distances = nearest_distances(features.reshape(-1, features.shape[-1]), models["bank"], chunk_rows=CHUNK_ROWS)
    return distances.reshape(features.shape[1], features.shape[2]).cpu().numpy()


def yolo_boxes(models, crop):
    result = models["yolo"].predict(crop, imgsz=IMAGE_SIZE, conf=MIN_CONFIDENCE, max_det=MAX_BOXES,
                                    device=models["device"], verbose=False)[0]
    return [[*box, confidence] for box, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())]


@st.cache_resource(show_spinner=False)
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).eval().to(device)
    extractor = create_feature_extractor(backbone, return_nodes={name: name for name in LAYERS})
    with np.load(BANK_FILE) as bank_file:
        bank = torch.from_numpy(bank_file["features"].astype(np.float32)).to(device)
    models = {"device": device, "extractor": extractor, "bank": bank, "yolo": YOLO(YOLO_FILE), "lock": threading.Lock()}
    blank = Image.new("RGB", (WIDTH, HEIGHT))
    for _ in range(WARM_UP):
        patchcore_grid(models, blank)
        yolo_boxes(models, blank)
    return models


def timed(run, device):
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    output = run()
    if device == "cuda":
        torch.cuda.synchronize()
    return output, 1000 * (time.perf_counter() - start)


@st.cache_data(show_spinner=False, max_entries=32)
def check_photo(photo_bytes):
    crop = cut_photo(photo_bytes)["crop"]
    models = load_models()
    with models["lock"]:
        if models["device"] == "cuda":
            patchcore_grid(models, crop)
            yolo_boxes(models, crop)
        grid, patchcore_ms = timed(lambda: patchcore_grid(models, crop), device=models["device"])
        boxes, yolo_ms = timed(lambda: yolo_boxes(models, crop), device=models["device"])
    return {"patchcore_score": float(grid.max()), "grid": grid,
            "yolo_score": max((box[4] for box in boxes), default=0.0), "yolo_boxes": boxes,
            "times": {"PatchCore": patchcore_ms, "YOLO": yolo_ms}, "device": "GPU" if models["device"] == "cuda" else "CPU"}
