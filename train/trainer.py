import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class NavigationTrainer:
    """
    Trainer for Imitation Learning (IL) on navigation tasks.
    """
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss() # For velocity regression

    def train_step(self, current_obs, path_seq, goal_img, expert_action):
        self.optimizer.zero_grad()
        
        # Predict action
        predicted_action = self.model(current_obs, path_seq, goal_img)
        
        # Loss (Imitation)
        loss = self.criterion(predicted_action, expert_action)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save_checkpoint(self, path="checkpoint.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
