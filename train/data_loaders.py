import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

class GaussianNoise(object):
    """Factual noise injection to simulate electronic sensor grain."""
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.sigma

    def __repr__(self):
        return self.__class__.__name__ + f'(sigma={self.sigma})'

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
        
        # 2. Load Depth for the current frame (last in sequence)
        depth_dir = os.path.join(self.data_dir, "depth_0")
        img_name = self.img_files[idx + self.seq_length - 1]
        depth_name = img_name.replace('.png', '_depth.npy') # If img is 000000_left.png, depth is 000000_left_depth.npy
        # If .npy doesn't exist, check for .png depth
        depth_path = os.path.join(depth_dir, depth_name)
        if not os.path.exists(depth_path):
            depth_path = os.path.join(depth_dir, img_name) # Shibuya might have .png depth
            
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)
        else:
            depth = np.array(Image.open(depth_path))
            
        # Normalize and transform depth
        depth = torch.from_numpy(depth).float().unsqueeze(0)
        # Resize to match RGB
        depth = torch.nn.functional.interpolate(depth.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=True).squeeze(0)

        # 3. Target Pose (Expert action)
        curr_pose = self.poses[idx + self.seq_length - 1]
        next_pose = self.poses[idx + self.seq_length]
        delta_pos = next_pose[:3] - curr_pose[:3]
        
        # 4. Siamese Goal Selection (10-20 steps into the future)
        # If we are near the end, we just use the last possible frame
        goal_idx = min(idx + self.seq_length + 15, len(self.img_files) - 1)
        goal_path = os.path.join(self.img_dir, self.img_files[goal_idx])
        goal_image = Image.open(goal_path).convert('RGB')
        if self.transform:
            goal_image = self.transform(goal_image)
            
        # 5. Success Label (Binary Classifier Ground Truth)
        # If distance to goal is < 1.0 meter, label as success (1.0)
        goal_pose = self.poses[goal_idx]
        dist_to_goal = np.linalg.norm(next_pose[:3] - goal_pose[:3])
        success_label = 1.0 if dist_to_goal < 1.0 else 0.0
            
        return images, torch.FloatTensor(delta_pos), depth, goal_image, torch.FloatTensor([success_label])

class TUMDataset(Dataset):
    """
    Loader for TUM RGB-D Benchmark sequences. 
    Synchronizes RGB and Depth frames by timestamp.
    """
    def __init__(self, data_dir, transform=None, seq_length=10):
        self.data_dir = data_dir
        self.transform = transform
        self.seq_length = seq_length
        
        # Load associated files (assumes they are pre-associated/synced)
        # Standard TUM format uses rgb.txt and depth.txt
        rgb_file = os.path.join(data_dir, "rgb.txt")
        depth_file = os.path.join(data_dir, "depth.txt")
        pose_file = os.path.join(data_dir, "groundtruth.txt")
        
        # Simple parser (skipping comment lines)
        self.rgb_data = np.genfromtxt(rgb_file, delimiter=' ', skip_header=3, dtype=None, encoding='utf-8')
        self.depth_data = np.genfromtxt(depth_file, delimiter=' ', skip_header=3, dtype=None, encoding='utf-8')
        self.poses = np.genfromtxt(pose_file, delimiter=' ', skip_header=3)
        
        # In a full implementation, we would interpolate poses to match image timestamps
        # For now, we assume fixed-rate sampling
        self.num_frames = min(len(self.rgb_data), len(self.depth_data), len(self.poses))
        
    def __len__(self):
        return self.num_frames - self.seq_length

    def __getitem__(self, idx):
        # 1. Load sequence of RGB
        images = []
        for i in range(self.seq_length):
            img_name = self.rgb_data[idx + i][1]
            img_path = os.path.join(self.data_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        images = torch.stack(images)
        
        # 2. Load Depth (last frame)
        depth_name = self.depth_data[idx + self.seq_length - 1][1]
        depth_path = os.path.join(self.data_dir, depth_name)
        depth = np.array(Image.open(depth_path)).astype(np.float32) / 5000.0 # TUM scale factor
        depth = torch.from_numpy(depth).float().unsqueeze(0)
        depth = torch.nn.functional.interpolate(depth.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=True).squeeze(0)
        
        # 3. Delta Pose
        curr_p = self.poses[idx + self.seq_length - 1][1:4]
        next_p = self.poses[idx + self.seq_length][1:4]
        delta_pos = next_p - curr_p
        
        return images, torch.FloatTensor(delta_pos), depth

class EuRoCDataset(Dataset):
    """
    Loader for EuRoC MAV Dataset. 
    Handles drone-specific motion patterns and CSV pose formats.
    """
    def __init__(self, data_dir, transform=None, seq_length=10):
        self.data_dir = data_dir
        self.transform = transform
        self.seq_length = seq_length
        
        # Standard EuRoC structure: cam0/data.csv and state_groundtruth_estimate0/data.csv
        self.img_dir = os.path.join(data_dir, "mav0/cam0/data")
        img_csv = os.path.join(data_dir, "mav0/cam0/data.csv")
        pose_csv = os.path.join(data_dir, "mav0/state_groundtruth_estimate0/data.csv")
        
        self.img_list = np.genfromtxt(img_csv, delimiter=',', skip_header=1, dtype=None, encoding='utf-8')
        self.poses = np.genfromtxt(pose_csv, delimiter=',', skip_header=1)
        
        self.num_frames = min(len(self.img_list), len(self.poses))

    def __len__(self):
        return self.num_frames - self.seq_length

    def __getitem__(self, idx):
        images = []
        for i in range(self.seq_length):
            img_name = self.img_list[idx + i][1]
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        images = torch.stack(images)
        
        # EuRoC usually doesn't have alignment depth, so we use a zero-mask or skip depth loss
        depth = torch.zeros((1, 224, 224))
        
        curr_p = self.poses[idx + self.seq_length - 1][1:4]
        next_p = self.poses[idx + self.seq_length][1:4]
        delta_pos = next_p - curr_p
        
        return images, torch.FloatTensor(delta_pos), depth

class SevenScenesDataset(Dataset):
    """
    Loader for Microsoft 7-Scenes Dataset. 
    Ideal for desk/office goal-matching.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

class CombinedNavigationDataset(Dataset):
    """
    High-level wrapper that merges multiple datasets for balanced training.
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        dataset_idx = 0
        while idx >= self.lengths[dataset_idx]:
            idx -= self.lengths[dataset_idx]
            dataset_idx += 1
        return self.datasets[dataset_idx][idx]
