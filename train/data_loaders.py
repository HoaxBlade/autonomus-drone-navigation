import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class TartanAirDataset(Dataset):
    """
    PyTorch Dataset for TartanAir visual navigation sequences.
    """
    def __init__(self, data_dir, transform=None, seq_length=10):
        self.data_dir = data_dir
        self.transform = transform
        self.seq_length = seq_length
        
        # Load image paths for Shibuya structure
        self.img_dir = os.path.join(data_dir, "image_0")
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])
        
        # Load poses (gt_pose.txt for Shibuya)
        pose_file = os.path.join(data_dir, "gt_pose.txt")
        self.poses = np.loadtxt(pose_file)

    def __len__(self):
        return len(self.img_files) - self.seq_length

    def __getitem__(self, idx):
        # 1. Load sequence of images (the 'path')
        images = []
        for i in range(self.seq_length):
            img_path = os.path.join(self.img_dir, self.img_files[idx + i])
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        images = torch.stack(images)
        
        # 2. Target Pose (Expert action)
        # We want to predict the relative motion to the NEXT frame
        curr_pose = self.poses[idx + self.seq_length - 1]
        next_pose = self.poses[idx + self.seq_length]
        
        # Relative movement (dx, dy, dz)
        delta_pos = next_pose[:3] - curr_pose[:3]
        
        return images, torch.FloatTensor(delta_pos)
