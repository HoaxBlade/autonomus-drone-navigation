import torch
from drone_nav.perception.encoders import VisualEncoder, GoalEncoder
from drone_nav.nav.path_follower import PathFollower
from drone_nav.nav.goal_matcher import GoalMatcher
from drone_nav.control.planner import IntegratedPlanner

def main():
    print("Initializing Advanced Drone Navigation System (ML)...")
    
    # 1. Initialize Encoders
    visual_encoder = VisualEncoder(architecture='resnet18', use_netvlad=True)
    goal_encoder = GoalEncoder(visual_encoder)
    
    # 2. Initialize Navigation Policies
    path_follower = PathFollower(input_dim=visual_encoder.output_dim)
    goal_matcher = GoalMatcher(input_dim=visual_encoder.output_dim)
    
    # 3. Initialize Fusion Planner
    planner = IntegratedPlanner(path_follower, goal_matcher)
    
    # 4. Mock Inputs (Testing structure)
    batch_size = 1
    dummy_obs = torch.randn(batch_size, 3, 224, 224)
    dummy_goal = torch.randn(batch_size, 3, 224, 224)
    dummy_path = torch.randn(batch_size, 10, 3, 224, 224) # Sequence of 10 keyframes
    
    print("Encoding inputs...")
    with torch.no_grad():
        obs_emb = visual_encoder(dummy_obs)
        goal_emb = goal_encoder(dummy_goal)
        
        # Process path sequence (flattened for the LSTM in this version)
        path_emb = torch.stack([visual_encoder(dummy_path[:, i]) for i in range(10)], dim=1)
        
        print(f"Observation Embedding Shape: {obs_emb.shape}")
        print(f"Goal Embedding Shape: {goal_emb.shape}")
        print(f"Path Embedding Shape: {path_emb.shape}")
        
        # 5. Plan action
        result = planner.plan(obs_emb, path_emb, goal_emb)
        print(f"Planner Result: {result}")

if __name__ == "__main__":
    main()
