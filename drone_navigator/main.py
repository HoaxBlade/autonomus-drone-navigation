from drone_nav.perception.encoders import PerceptionBackbone, VisualEncoder, GoalEncoder, DepthEncoder

def main():
    print("Initializing Industry-Grade Drone Navigation System (v2.0)...")
    
    # 1. Initialize Unified Backbone
    backbone = PerceptionBackbone(architecture='resnet18')
    
    # 2. Initialize Specialized Heads
    visual_encoder = VisualEncoder(backbone, use_netvlad=True)
    goal_encoder = GoalEncoder(backbone)
    depth_encoder = DepthEncoder(backbone)
    
    # 3. Initialize Navigation Policies
    path_follower = PathFollower(input_dim=visual_encoder.output_dim)
    goal_matcher = GoalMatcher(input_dim=visual_encoder.output_dim)
    
    # 4. Initialize Fusion Planner
    planner = IntegratedPlanner(path_follower, goal_matcher)
    
    # 5. Mock Inputs (Testing v2.0 structure)
    batch_size = 1
    dummy_obs = torch.randn(batch_size, 3, 224, 224)
    dummy_goal = torch.randn(batch_size, 3, 224, 224)
    dummy_path = torch.randn(batch_size, 10, 3, 224, 224)
    
    print("Encoding environmental and goal data with Unified Backbone...")
    with torch.no_grad():
        # Using the same backbone for multiple heads
        obs_emb = visual_encoder(dummy_obs)
        goal_emb = goal_encoder(dummy_goal)
        depth_map = depth_encoder(dummy_obs)
        
        # Process path sequence
        path_emb = torch.stack([visual_encoder(dummy_path[:, i]) for i in range(10)], dim=1)
        
        print(f"Perception Status: Success")
        print(f" - Geometry: Estimated {depth_map.shape[2]}x{depth_map.shape[3]} depth map for obstacle avoidance.")
        print(f" - Live View: Extracted {obs_emb.shape[1]} unique visual features using NetVLAD.")
        print(f" - Path Memory: Loaded {path_emb.shape[1]} keyframes for the current route.")
        print(f" - Goal Target: Destination features successfully encoded.")
        
        # 6. Plan action with geometric awareness
        result = planner.plan(obs_emb, path_emb, goal_emb, depth_map=depth_map)
        
        print("\nNavigation Decision (v2.1):")
        action = result.get("action")
        velocity = result.get("velocity")[0] if isinstance(result.get("velocity"), list) else result.get("velocity")
        
        if action == "MOVE":
            print(f" - Action: Following the visual path.")
            print(f" - Predicted Velocity: Forward={velocity[0]:.2f} m/s, Right={velocity[1]:.2f} m/s, Down={velocity[2]:.2f} m/s")
        elif action == "LAND":
            print(" - Action: Goal reached. Initiating landing sequence.")

if __name__ == "__main__":
    main()
