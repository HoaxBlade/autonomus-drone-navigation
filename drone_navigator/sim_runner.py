import torch
import numpy as np
import os
from drone_nav.sim_interface.habitat_bridge import HabitatBridge
from drone_nav.sim_interface.dynamics import DroneDynamics
from drone_nav.control.planner import IntegratedPlanner
from drone_nav.perception.encoders import PerceptionBackbone, VisualEncoder, GoalEncoder, DepthEncoder
from drone_nav.nav.path_follower import PathFollower
from drone_nav.nav.goal_matcher import GoalMatcher

def run_simulation(scene_path, goal_pos, weights_path="checkpoints/nav_stack_v2_2.pth", max_steps=500):
    print(f"--- STARTING CLOSED-LOOP SIMULATION ---")
    
    # 1. Initialize Simulation Components
    bridge = HabitatBridge(scene_path)
    dynamics = DroneDynamics(dt=0.1)
    
    # 2. Initialize Navigation Stack
    from drone_nav.utils.device import get_device
    device = get_device()
    
    backbone = PerceptionBackbone(architecture='resnet18').to(device)
    visual_encoder = VisualEncoder(backbone).to(device)
    goal_encoder = GoalEncoder(backbone).to(device)
    depth_encoder = DepthEncoder(backbone).to(device)
    
    path_follower = PathFollower(input_dim=visual_encoder.output_dim).to(device)
    goal_matcher = GoalMatcher(input_dim=backbone.out_channels).to(device)
    planner = IntegratedPlanner(path_follower, goal_matcher)
    
    # Load Trained Weights
    if os.path.exists(weights_path):
        print(f"Loading weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        backbone.load_state_dict(checkpoint['backbone'])
        visual_encoder.load_state_dict(checkpoint['visual_encoder'])
        goal_encoder.load_state_dict(checkpoint['goal_encoder'])
        depth_encoder.load_state_dict(checkpoint['depth_encoder'])
        path_follower.load_state_dict(checkpoint['path_follower'])
        goal_matcher.load_state_dict(checkpoint['goal_matcher'])
    else:
        print(f"[WARNING] Weights not found at {weights_path}. Running with random initialization.")

    backbone.eval()
    visual_encoder.eval()
    goal_encoder.eval()
    depth_encoder.eval()
    
    # 3. Simulation Loop
    dynamics.reset(position=[0, 0, 0])
    
    # Memory for VPR (10 frames)
    vpr_memory = []
    
    for step in range(max_steps):
        state = dynamics.get_state()
        pos = state['position']
        
        # A. Perception
        obs = bridge.get_observation(pos, rotation=[0,0,0,1])
        rgb_tensor = torch.from_numpy(obs['rgb']).permute(2,0,1).float().unsqueeze(0).to(device)
        rgb_tensor /= 255.0 # Normalize to 0-1
        
        with torch.no_grad():
            # 1. Visual Encoder (VPR)
            current_vpr = visual_encoder(rgb_tensor)
            vpr_memory.append(current_vpr)
            if len(vpr_memory) > 10:
                vpr_memory.pop(0)
            
            # Form the memory tensor (N, T, Dim)
            vpr_memory_tensor = torch.stack(vpr_memory, dim=1) # (1, T, Dim)
            # If memory is not full yet, pad it
            if vpr_memory_tensor.shape[1] < 10:
                padding = torch.zeros((1, 10 - vpr_memory_tensor.shape[1], vpr_memory_tensor.shape[2])).to(device)
                vpr_memory_tensor = torch.cat([padding, vpr_memory_tensor], dim=1)
            
            # 2. Goal Matcher (Siamese)
            obs_siamese = goal_encoder(rgb_tensor)
            # Mock goal (In real eval, this would be the destination image features)
            goal_siamese = obs_siamese # Placeholder
            
            # 3. Depth (Geometry)
            # depth = depth_encoder(rgb_tensor)
            
            # B. Planning
            result = planner.plan(obs_siamese, vpr_memory_tensor, goal_siamese, vpr_obs=current_vpr)
            target_v = np.array(result['velocity'])
            
        # C. Dynamics Update
        new_pos, new_vel = dynamics.step(target_v)
        
        # D. Success/Failure Checks
        dist_to_goal = np.linalg.norm(np.array(new_pos) - np.array(goal_pos))
        
        if dist_to_goal < 1.0:
            print(f"Goal Reached at step {step}! SUCCESS.")
            break
            
        if bridge.check_collision(new_pos):
            print(f"Collision detected at step {step}! CRASH.")
            break
            
        if step % 50 == 0:
            print(f"Step {step}: Dist to Goal = {dist_to_goal:.2f}m | Speed = {state['speed']:.2f} m/s")

    print(f"--- SIMULATION FINISHED ---")

if __name__ == "__main__":
    run_simulation(scene_path="scenes/shibuya.glb", goal_pos=[10, 0, 10])
