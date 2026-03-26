import torch
import numpy as np
from drone_nav.sim_interface.habitat_bridge import HabitatBridge
from drone_nav.sim_interface.dynamics import DroneDynamics
from drone_nav.control.planner import IntegratedPlanner
from drone_nav.perception.encoders import PerceptionBackbone, VisualEncoder, GoalEncoder, DepthEncoder
from drone_nav.nav.path_follower import PathFollower
from drone_nav.nav.goal_matcher import GoalMatcher

def run_simulation(scene_path, goal_pos, max_steps=500):
    print(f"--- STARTING CLOSED-LOOP SIMULATION ---")
    
    # 1. Initialize Simulation Components
    bridge = HabitatBridge(scene_path)
    dynamics = DroneDynamics(dt=0.1) # 10Hz perception/control for sim
    
    # 2. Initialize Navigation Stack
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    backbone = PerceptionBackbone(architecture='resnet18').to(device)
    visual_encoder = VisualEncoder(backbone).to(device)
    goal_encoder = GoalEncoder(backbone).to(device)
    
    path_follower = PathFollower(input_dim=visual_encoder.output_dim).to(device)
    goal_matcher = GoalMatcher(input_dim=backbone.out_channels).to(device)
    planner = IntegratedPlanner(path_follower, goal_matcher)
    
    # 3. Simulation Loop
    dynamics.reset(position=[0, 0, 0])
    
    for step in range(max_steps):
        state = dynamics.get_state()
        pos = state['position']
        
        # A. Perception
        obs = bridge.get_observation(pos, rotation=[0,0,0,1])
        rgb_tensor = torch.from_numpy(obs['rgb']).permute(2,0,1).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            obs_vpr = visual_encoder(rgb_tensor)
            obs_siamese = goal_encoder(rgb_tensor)
            # Mock goal embedding (usually pre-encoded)
            goal_siamese = obs_siamese 
            
            # B. Planning
            # Placeholder for path memory (requires 10 previous frames)
            dummy_path = torch.randn(1, 10, obs_vpr.shape[1]).to(device)
            result = planner.plan(obs_siamese, dummy_path, goal_siamese, vpr_obs=obs_vpr)
            
            target_v = np.array(result['velocity'])
            
        # C. Dynamics Update
        new_pos, new_vel = dynamics.step(target_v)
        
        # D. Success/Failure Checks
        dist_to_goal = np.linalg.norm(np.array(new_pos) - np.array(goal_pos))
        
        if dist_to_goal < 1.0:
            print(f"Goal Reached at step {step}!")
            break
            
        if bridge.check_collision(new_pos):
            print(f"Collision detected at step {step}!")
            break
            
        if step % 50 == 0:
            print(f"Step {step}: Distance to Goal = {dist_to_goal:.2f}m | Speed = {state['speed']:.2f} m/s")

    print(f"--- SIMULATION FINISHED ---")

if __name__ == "__main__":
    run_simulation(scene_path="scenes/shibuya.glb", goal_pos=[10, 0, 10])
