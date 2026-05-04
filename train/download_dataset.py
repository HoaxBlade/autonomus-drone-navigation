import os
from huggingface_hub import hf_hub_download
import zipfile
import shutil

def download_tartanair_hf(target_dir="data/tartanair", token=None):
    """
    Downloads TartanAir 'AbandonedFactory' from Hugging Face.
    """
    if token is None:
        token = os.environ.get("HF_TOKEN")
    
    if not token:
        raise ValueError("HF_TOKEN not found. Please set the HF_TOKEN environment variable.")
    repo_id = "theairlabcmu/tartanair"
    env = "abandonedfactory"
    traj = "P001"
    
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"--- Starting Data Ingestion from Hugging Face ({env}/{traj}) ---")
    
    # We download Image and Depth zips
    modalities = ["image_left", "depth_left"]
    
    for mod in modalities:
        filename = f"{env}/Easy/{mod}.zip"
        print(f"\n[1/2] Syncing {mod} from Hugging Face...")
        
        # Download handles large files and resumes
        zip_path = hf_hub_download(
            repo_id=repo_id, 
            filename=filename, 
            repo_type="dataset", 
            token=token,
            local_dir=target_dir
        )
        
        print(f"[2/2] Extracting {traj} from {mod}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Only extract P001 to keep disk usage low
            members = [m for m in z.namelist() if f"Easy/{traj}/" in m]
            z.extractall(path=target_dir, members=members)
            
    # Normalize structure for the Trainer
    # Trainer expects: image_0, depth_0, gt_pose.txt
    base_path = os.path.join(target_dir, env, "Easy", traj)
    
    print("\n--- Finalizing Directory Structure ---")
    if os.path.exists(os.path.join(base_path, "image_left")):
        shutil.move(os.path.join(base_path, "image_left"), os.path.join(base_path, "image_0"))
        print(f"Mapped: image_left -> image_0")
        
    if os.path.exists(os.path.join(base_path, "depth_left")):
        shutil.move(os.path.join(base_path, "depth_left"), os.path.join(base_path, "depth_0"))
        print(f"Mapped: depth_left -> depth_0")
        
    if os.path.exists(os.path.join(base_path, "pose_left.txt")):
        shutil.copy(os.path.join(base_path, "pose_left.txt"), os.path.join(base_path, "gt_pose.txt"))
        print(f"Mapped: pose_left.txt -> gt_pose.txt")

    print(f"\n--- SUCCESS: Data is ready at: {base_path} ---")

if __name__ == "__main__":
    download_tartanair_hf()
