from drone_nav.perception.encoders import PerceptionBackbone, VisualEncoder, GoalEncoder, DepthEncoder

def main():
    from drone_nav.utils.device import get_device
    device = get_device()
    print(f"Initializing Drone Navigator v2.1 on {device}...")
    
    # 1. Initialize Unified Backbone
    backbone = PerceptionBackbone(architecture='resnet18').to(device)
    
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
        action = result.get("action")
        velocity = result.get("velocity")[0] if isinstance(result.get("velocity"), list) else result.get("velocity")
        
        if action == "MOVE":
            print(f" - Action: Following the visual path.")
            print(f" - Predicted Velocity: Forward={velocity[0]:.2f} m/s, Right={velocity[1]:.2f} m/s, Down={velocity[2]:.2f} m/s")
        elif action == "LAND":
            print(" - Action: Goal reached. Initiating landing sequence.")

if __name__ == "__main__":
    main()
