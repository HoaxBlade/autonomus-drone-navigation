import torch
import torch.nn as nn

class PathFollower(nn.Module):
    """
    Modular path follower that processes a sequence of visual embeddings.
    Used for Visual Route Following (VRF).
    """
    def __init__(self, input_dim, hidden_dim=512, action_dim=3):
        super(PathFollower, self).__init__()
        
        # LSTM to process the sequence of keyframes/path observations
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Policy head: Maps hidden state to action (e.g., velocity vx, vy, yaw_rate)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh() # Normalized output (-1 to 1) for velocity commands
        )

    def forward(self, current_embedding, path_sequence_embeddings):
        """
        Args:
            current_embedding: (N, D) - current drone view embedding
            path_sequence_embeddings: (N, T, D) - sequence of keyframes for the route
        """
        # Concatenate or process the sequence with the current observation
        # For simplicity, let's treat the path as a memory context
        _, (h_n, _) = self.lstm(path_sequence_embeddings)
        
        # h_n shape: (num_layers, N, hidden_dim). Take the last layer.
        context = h_n[-1]
        
        # In a more advanced version, we would use an attention mechanism 
        # between current_embedding and path_sequence_embeddings.
        # Here we just use the final LSTM hidden state as the "path intent".
        
        action = self.policy(context)
        return action
