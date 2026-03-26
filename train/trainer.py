import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from drone_nav.perception.encoders import VisualEncoder, GoalEncoder
from drone_nav.nav.path_follower import PathFollower
from drone_nav.nav.goal_matcher import GoalMatcher
from .data_loaders import TartanAirDataset

class NavigationTrainer:
    def __init__(self, data_dir, lr=1e-4, batch_size=8, seq_length=10):
        # Set device (MPS for Mac, CUDA for Linux, CPU otherwise)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(f"Using device: {self.device}")

        # Initialize Encoders
        self.visual_encoder = VisualEncoder(architecture='resnet18', use_netvlad=True).to(self.device)
        self.goal_encoder = GoalEncoder(self.visual_encoder).to(self.device)
        
        # Initialize Policies
        self.path_follower = PathFollower(input_dim=self.visual_encoder.output_dim).to(self.device)
        self.goal_matcher = GoalMatcher(input_dim=self.visual_encoder.output_dim).to(self.device)
        
        # Optimizers
        self.optimizer = optim.Adam(
            list(self.visual_encoder.parameters()) + 
            list(self.path_follower.parameters()) + 
            list(self.goal_matcher.parameters()), 
            lr=lr
        )
        self.criterion = nn.MSELoss()

        # Data Loading
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.dataset = TartanAirDataset(data_dir, transform=transform, seq_length=seq_length)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def train_epoch(self, epoch):
        self.visual_encoder.train()
        self.path_follower.train()
        
        running_loss = 0.0
        for i, (images_seq, target_motion) in enumerate(self.dataloader):
            images_seq = images_seq.to(self.device) # (N, T, C, H, W)
            target_motion = target_motion.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Encoder features for each frame in the sequence
            N, T, C, H, W = images_seq.shape
            images_flat = images_seq.view(N * T, C, H, W)
            features_flat = self.visual_encoder(images_flat)
            features_seq = features_flat.view(N, T, -1)
            
            current_obs = features_seq[:, -1, :] # Last frame in sequence
            
            # Predict action from path follower
            predicted_action = self.path_follower(current_obs, features_seq)
            
            # Loss Calculation (Imitation)
            loss = self.criterion(predicted_action, target_motion)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch}] Batch [{i}/{len(self.dataloader)}] Loss: {loss.item():.4f}")
                
        return running_loss / len(self.dataloader)

if __name__ == "__main__":
    # Placeholder for actual data dir
    trainer = NavigationTrainer(data_dir="data/tartanair/abandon_village/P001")
    for epoch in range(1, 11):
        avg_loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
