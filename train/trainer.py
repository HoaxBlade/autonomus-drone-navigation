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
    def __init__(self, datasets_config, lr=1e-4, batch_size=2, seq_length=10, freeze_backbone=True):
        """
        datasets_config: List of dicts, e.g., 
        [{'type': 'tartanair', 'path': '...'}, {'type': 'tum', 'path': '...'}]
        """
        from drone_nav.utils.device import get_device
        self.device = get_device()
        print(f"Trainer Initialized on: {self.device}")

        # 1. Unified Backbone
        self.backbone = PerceptionBackbone(architecture='resnet18').to(self.device)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen to save memory.")
        
        # 2. Specialized Heads
        self.visual_encoder = VisualEncoder(self.backbone, use_netvlad=True).to(self.device)
        self.goal_encoder = GoalEncoder(self.backbone).to(self.device)
        self.depth_encoder = DepthEncoder(self.backbone).to(self.device)
        
        # 3. Policies
        self.path_follower = PathFollower(input_dim=self.visual_encoder.output_dim).to(self.device)
        self.goal_matcher = GoalMatcher(input_dim=self.backbone.out_channels).to(self.device)
        
        # Optimizer (excluding frozen params)
        params = list(self.path_follower.parameters()) + \
                 list(self.goal_matcher.parameters()) + \
                 list(self.depth_encoder.parameters())
        if not freeze_backbone:
            params += list(self.backbone.parameters())
        self.optimizer = optim.Adam(params, lr=lr)
        
        self.nav_criterion = nn.MSELoss()
        self.depth_criterion = nn.MSELoss()

        # 4. Multi-Dataset Loading
        from train.data_loaders import TartanAirDataset, TUMDataset, EuRoCDataset, CombinedNavigationDataset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        all_datasets = []
        for config in datasets_config:
            d_type = config['type']
            d_path = config['path']
            if d_type == 'tartanair':
                all_datasets.append(TartanAirDataset(d_path, transform=transform, seq_length=seq_length))
            elif d_type == 'tum':
                all_datasets.append(TUMDataset(d_path, transform=transform, seq_length=seq_length))
            elif d_type == 'euroc':
                all_datasets.append(EuRoCDataset(d_path, transform=transform, seq_length=seq_length))

        self.dataset = CombinedNavigationDataset(all_datasets)
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
            features_vpr_seq = self.visual_encoder(images_flat).view(N, T, -1)
            current_vpr_obs = features_vpr_seq[:, -1, :]
            
            # Predict action from path follower
            predicted_action = self.path_follower(current_vpr_obs, features_vpr_seq)
            nav_loss = self.nav_criterion(predicted_action, target_motion)
            
            # 3. Goal Similarity Forward Pass
            obs_siamese = self.goal_encoder(last_frame)
            # For training, compute similarity between current and a "near-future" frame as goal
            # This teaches the model what "arriving" looks like.
            goal_loss = self.nav_criterion(obs_siamese, obs_siamese) # Placeholder for Siamese loss logic
            
            # 4. Multi-task Loss Balancing
            # w1: Action (Navigation), w2: Depth (Geometry), w3: Goal (Alignment)
            w_nav, w_depth, w_goal = 1.0, 5.0, 2.0
            
            loss = (w_nav * nav_loss) + (w_depth * depth_loss) + (w_goal * goal_loss)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch}] Batch [{i}/{len(self.dataloader)}]")
                print(f" -> [Nav]: {nav_loss.item():.4f} | [Depth]: {depth_loss.item():.4f} | [Goal]: {goal_loss.item():.4f}")
                print(f" -> Total Balanced Loss: {loss.item():.4f}")

if __name__ == "__main__":
    trainer = NavigationTrainer(data_dir="data/tartanair_shibuya/TartanAir_shibuya/RoadCrossing03")
    for epoch in range(1, 11):
        avg_loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch} Average Multi-Task Loss: {avg_loss:.4f}")
    
    # Save final model state
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/nav_stack_v2_2.pth"
    torch.save({
        'backbone': trainer.backbone.state_dict(),
        'visual_encoder': trainer.visual_encoder.state_dict(),
        'goal_encoder': trainer.goal_encoder.state_dict(),
        'depth_encoder': trainer.depth_encoder.state_dict(),
        'path_follower': trainer.path_follower.state_dict(),
        'goal_matcher': trainer.goal_matcher.state_dict(),
    }, save_path)
    print(f"Final model saved to: {save_path}")
