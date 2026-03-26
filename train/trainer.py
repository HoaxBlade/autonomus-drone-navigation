import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from drone_nav.perception.encoders import PerceptionBackbone, VisualEncoder, GoalEncoder, DepthEncoder
from drone_nav.nav.path_follower import PathFollower
from drone_nav.nav.goal_matcher import GoalMatcher
from train.data_loaders import TartanAirDataset

class NavigationTrainer:
    def __init__(self, data_dir, lr=1e-4, batch_size=8, seq_length=10):
        # Set device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(f"Using device: {self.device}")

        # 1. Unified Backbone
        self.backbone = PerceptionBackbone(architecture='resnet18').to(self.device)
        
        # 2. Specialized Heads
        self.visual_encoder = VisualEncoder(self.backbone, use_netvlad=True).to(self.device)
        self.goal_encoder = GoalEncoder(self.backbone).to(self.device)
        self.depth_encoder = DepthEncoder(self.backbone).to(self.device)
        
        # 3. Policies
        self.path_follower = PathFollower(input_dim=self.visual_encoder.output_dim).to(self.device)
        self.goal_matcher = GoalMatcher(input_dim=self.visual_encoder.output_dim).to(self.device)
        
        # Optimizer for all parameters
        self.optimizer = optim.Adam(
            list(self.backbone.parameters()) + 
            list(self.path_follower.parameters()) + 
            list(self.goal_matcher.parameters()) +
            list(self.depth_encoder.parameters()),
            lr=lr
        )
        
        self.nav_criterion = nn.MSELoss()
        self.depth_criterion = nn.MSELoss()

        # Data Loading
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.dataset = TartanAirDataset(data_dir, transform=transform, seq_length=seq_length)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def train_epoch(self, epoch):
        self.backbone.train()
        self.depth_encoder.train()
        self.path_follower.train()
        
        total_loss = 0.0
        for i, (images_seq, target_motion, target_depth) in enumerate(self.dataloader):
            images_seq = images_seq.to(self.device)
            target_motion = target_motion.to(self.device)
            target_depth = target_depth.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 1. Shared Backbone Forward Pass
            N, T, C, H, W = images_seq.shape
            last_frame = images_seq[:, -1, :, :, :] # (N, C, H, W)
            
            # Predict Depth (Geometry Head)
            predicted_depth = self.depth_encoder(last_frame)
            depth_loss = self.depth_criterion(predicted_depth, target_depth)
            
            # 2. Navigation Forward Pass (VPR + Policy)
            images_flat = images_seq.view(N * T, C, H, W)
            features_seq = self.visual_encoder(images_flat).view(N, T, -1)
            current_obs = features_seq[:, -1, :]
            
            predicted_action = self.path_follower(current_obs, features_seq)
            nav_loss = self.nav_criterion(predicted_action, target_motion)
            
            # 3. Combined Multi-task Loss
            loss = nav_loss + 10.0 * depth_loss # Weight depth loss more for faster geometry learning
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch}] Batch [{i}/{len(self.dataloader)}] Nav Loss: {nav_loss.item():.4f} | Depth Loss: {depth_loss.item():.4f}")
                
        return total_loss / len(self.dataloader)

if __name__ == "__main__":
    trainer = NavigationTrainer(data_dir="data/tartanair_shibuya/TartanAir_shibuya/RoadCrossing03")
    for epoch in range(1, 11):
        avg_loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch} Average Multi-Task Loss: {avg_loss:.4f}")
