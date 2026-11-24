"""
download_data.py
Utility to download PlantVillage dataset samples directly from the official PlantVillage repository.
Downloads balanced subsets for training, validation, and testing.
"""

import os
import sys
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

SELECTED_CLASSES = [
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
]

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
GITHUB_API_BASE = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color"

def download_file(url, dest_path):
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            with open(dest_path, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return False

def download_class_images(class_name, num_images=100):
    class_dir = os.path.join(DATASET_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # Check if already downloaded
    existing = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(existing) >= num_images:
        print(f"[OK] {class_name}: Already has {len(existing)} images.")
        return len(existing)
        
    print(f"Fetching file list for {class_name}...")
    api_url = f"{GITHUB_API_BASE}/{class_name}"
    headers = {"User-Agent": "PlantDiseaseProject/1.0"}
    res = requests.get(api_url, headers=headers, timeout=25)
    if res.status_code != 200:
        print(f"[ERROR] Failed to query GitHub API for {class_name}: Status {res.status_code}")
        return 0
        
    items = res.json()
    if not isinstance(items, list):
        print(f"[ERROR] Unexpected response: {items}")
        return 0
        
    # Filter for image files
    img_items = [it for it in items if it.get("name", "").lower().endswith((".jpg", ".jpeg", ".png"))]
    selected_items = img_items[:num_images]
    
    print(f"Downloading {len(selected_items)} images for {class_name}...")
    success_count = 0
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {}
        for it in selected_items:
            dest_path = os.path.join(class_dir, it["name"])
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
                success_count += 1
                continue
            download_url = it.get("download_url")
            if download_url:
                futures[executor.submit(download_file, download_url, dest_path)] = it["name"]
                
        for fut in as_completed(futures):
            if fut.result():
                success_count += 1
                
    print(f"[DONE] {class_name}: {success_count} images downloaded.")
    return success_count

def copy_sample_images():
    """Copy a few representative images to samples/ directory for easy testing in Streamlit."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    for class_name in SELECTED_CLASSES:
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_dir):
            continue
        imgs = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if imgs:
            src = os.path.join(class_dir, imgs[0])
            dest_name = f"sample_{class_name}.jpg"
            dest = os.path.join(SAMPLES_DIR, dest_name)
            if not os.path.exists(dest):
                with open(src, "rb") as fsrc, open(dest, "wb") as fdest:
                    fdest.write(fsrc.read())
    print(f"[SAMPLES] Sample test images saved to: {SAMPLES_DIR}")

def download_dataset(images_per_class=100):
    os.makedirs(DATASET_DIR, exist_ok=True)
    print("=" * 60)
    print("  PlantVillage Dataset Downloader")
    print(f"  Target: {len(SELECTED_CLASSES)} classes, {images_per_class} images/class")
    print(f"  Destination: {DATASET_DIR}")
    print("=" * 60)
    
    total = 0
    for cls in SELECTED_CLASSES:
        cnt = download_class_images(cls, num_images=images_per_class)
        total += cnt
        
    copy_sample_images()
    print("=" * 60)
    print(f"Dataset preparation complete! Total images: {total}")
    print("=" * 60)
    return total

if __name__ == "__main__":
    count = 100
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            pass
    download_dataset(images_per_class=count)
