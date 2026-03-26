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
    
    print("Encoding environmental and goal data...")
    with torch.no_grad():
        obs_emb = visual_encoder(dummy_obs)
        goal_emb = goal_encoder(dummy_goal)
        
        # Process path sequence (flattened for the LSTM in this version)
        path_emb = torch.stack([visual_encoder(dummy_path[:, i]) for i in range(10)], dim=1)
        
        print(f"Perception Status: Success")
        print(f" - Live View: Extracted {obs_emb.shape[1]} unique visual features using NetVLAD.")
        print(f" - Path Memory: Loaded {path_emb.shape[1]} keyframes for the current route.")
        print(f" - Goal Target: Destination features successfully encoded.")
        
        # 5. Plan action
        result = planner.plan(obs_emb, path_emb, goal_emb)
        
        print("\nNavigation Decision:")
        action = result.get("action")
        velocity = result.get("velocity")[0]
        
        if action == "MOVE":
            print(f" - Action: Following the visual path.")
            print(f" - Predicted Velocity: Forward={velocity[0]:.2f} m/s, Right={velocity[1]:.2f} m/s, Down={velocity[2]:.2f} m/s")
        elif action == "LAND":
            print(" - Action: Goal reached. Initiating landing sequence.")

if __name__ == "__main__":
    main()
