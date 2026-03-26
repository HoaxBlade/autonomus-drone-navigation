import torch
import torch.nn as nn
import torch.nn.functional as F

class GoalMatcher(nn.Module):
    """
    Goal Matcher for Image-Goal Navigation (IGN).
    Compares current view with the target goal image.
    """
    def __init__(self, input_dim):
        super(GoalMatcher, self).__init__()
        
        # Metric learning head
        self.similarity_fc = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid() # Probability of being at the goal
        )

    def forward(self, current_embedding, goal_embedding):
        """
        Calculates similarity between current view and goal.
        """
        # Concatenate embeddings
        combined = torch.cat([current_embedding, goal_embedding], dim=-1)
        similarity = self.similarity_fc(combined)
        
        return similarity

    def get_distance(self, current_embedding, goal_embedding):
        """
        Simple Euclidean or Cosine distance for heuristic-based planning.
        """
        return F.cosine_similarity(current_embedding, goal_embedding)
