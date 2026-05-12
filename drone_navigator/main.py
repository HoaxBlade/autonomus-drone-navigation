import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from drone_nav.perception.encoders import (
    PerceptionBackbone, VisualEncoder, GoalEncoder, DepthEncoder, MemoryModule
)
from drone_nav.nav.path_follower import PathFollower
from drone_nav.nav.goal_matcher import GoalMatcher
from drone_nav.control.planner import IntegratedPlanner

def main():
    from drone_nav.utils.device import get_device
    device = get_device()
    print(f"Initializing Drone Navigator v2.2 on {device}...")

    # 1. Initialize Unified Backbone
    backbone = PerceptionBackbone(architecture='resnet18').to(device)

    # 2. Initialize Specialized Heads
    visual_encoder = VisualEncoder(backbone, use_netvlad=True).to(device)
    goal_encoder   = GoalEncoder(backbone).to(device)
    depth_encoder  = DepthEncoder(backbone).to(device)

    # 3. Memory + Navigation Policies
    memory        = MemoryModule(input_dim=visual_encoder.output_dim).to(device)
    path_follower = PathFollower(input_dim=memory.hidden_dim).to(device)
    goal_matcher  = GoalMatcher(input_dim=backbone.out_channels).to(device)

    # 4. Initialize Fusion Planner
    planner = IntegratedPlanner(path_follower, goal_matcher, memory=memory)
    
    # 5. Mock Inputs (Testing v2.0 structure)
    batch_size = 1
    dummy_obs = torch.randn(batch_size, 3, 224, 224)
    dummy_goal = torch.randn(batch_size, 3, 224, 224)
    dummy_path = torch.randn(batch_size, 10, 3, 224, 224)
    
    print("Encoding environmental and goal data with Unified Backbone...")
    with torch.no_grad():
        # High-level goal/visual alignment (Pooled 512)
        obs_siamese = goal_encoder(dummy_obs)
        goal_siamese = goal_encoder(dummy_goal)
        
        # Place Recognition / Path Following (NetVLAD 32768)
        obs_vpr = visual_encoder(dummy_obs)
        path_vpr = torch.stack([visual_encoder(dummy_path[:, i]) for i in range(10)], dim=1)
        
        # Geometry
        depth_map = depth_encoder(dummy_obs)
        
        print(f"Perception Status: Success")
        print(f" - Geometry: Estimated {depth_map.shape[2]}x{depth_map.shape[3]} depth map.")
        print(f" - VPR Features: {obs_vpr.shape[1]} descriptors for path memory.")
        print(f" - Goal Signal: Siamese alignment prepared.")
        
        # 6. Plan action with geometric awareness and dual embeddings
        result = planner.plan(obs_siamese, path_vpr, goal_siamese, depth_map=depth_map, vpr_obs=obs_vpr)
        
        print("\nNavigation Decision:")
        action   = result.get("action")
        velocity = result.get("velocity")   # already [vx, vy, vz]

        if action == "MOVE":
            print(f" - Action: Following the visual path.")
            print(f" - Predicted Velocity: Forward={velocity[0]:.2f} m/s, Right={velocity[1]:.2f} m/s, Down={velocity[2]:.2f} m/s")
        elif action == "LAND":
            print(" - Action: Goal reached. Initiating landing sequence.")

if __name__ == "__main__":
    main()
