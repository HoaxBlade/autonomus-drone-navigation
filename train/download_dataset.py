import os
import requests
import zipfile
import io

def download_tartanair_real(target_dir="data/tartanair"):
    """
    Downloads Real TartanAir 'Abandon Village' (P001) subset.
    Approx 1.5GB total.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # TartanAir URLs
    urls = {
        "images": "https://tartanair.blob.core.windows.net/tartanair-release/abandon_village/P001/image_left.zip",
        "poses": "https://tartanair.blob.core.windows.net/tartanair-release/abandon_village/P001/pose_left.txt"
    }

    print("--- Starting Real Data Ingestion (TartanAir Subset) ---")
    
    # 1. Download Poses
    print(f"Downloading Poses from {urls['poses']}...")
    r = requests.get(urls['poses'])
    with open(os.path.join(target_dir, "pose_left.txt"), 'wb') as f:
        f.write(r.content)
    print("Poses downloaded successfully.")

    # 2. Download Images
    print(f"Downloading Images (ZIP) from {urls['images']}...")
    print("This may take several minutes depending on your connection...")
    
    r = requests.get(urls['images'], stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(target_dir)
    
    print(f"--- Data ingestion complete! Files extracted to {target_dir} ---")

if __name__ == "__main__":
    download_tartanair_real()
