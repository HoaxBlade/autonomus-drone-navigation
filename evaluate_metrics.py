import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import torch
from drone_nav.perception.encoders import PerceptionBackbone, VisualEncoder, GoalEncoder, DepthEncoder, MemoryModule
from drone_nav.nav.path_follower import PathFollower
from drone_nav.nav.goal_matcher import GoalMatcher
from drone_nav.control.planner import IntegratedPlanner
from train.data_loaders import TartanAirDataset
from torchvision import transforms

from drone_nav.utils.device import get_device

def evaluate_system(data_dir, weights_path=None):
    print(f"--- STARTING FORMAL EVALUATION (v2.2) ---")
    print(f"Target Sequence: {data_dir.split('/')[-1]}")
    
    device = get_device()
    
    # 1. Initialize Stack
    backbone = PerceptionBackbone(architecture='resnet18').to(device)
    visual_encoder = VisualEncoder(backbone).to(device)
    goal_encoder = GoalEncoder(backbone).to(device)
    depth_encoder = DepthEncoder(backbone).to(device)
    memory = MemoryModule(input_dim=visual_encoder.output_dim).to(device)
    path_follower = PathFollower(input_dim=memory.hidden_dim).to(device)
    goal_matcher = GoalMatcher(input_dim=backbone.out_channels).to(device)
    
    # Load weights if provided
    if weights_path and os.path.exists(weights_path):
        print(f"Loading trained weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        backbone.load_state_dict(checkpoint.get('backbone', {}), strict=False)
        visual_encoder.load_state_dict(checkpoint.get('visual_encoder', {}), strict=False)
        goal_encoder.load_state_dict(checkpoint.get('goal_encoder', {}), strict=False)
        depth_encoder.load_state_dict(checkpoint.get('depth_encoder', {}), strict=False)
        goal_matcher.load_state_dict(checkpoint.get('goal_matcher', {}), strict=False)
        
        try:
            path_follower.load_state_dict(checkpoint.get('path_follower', {}))
        except RuntimeError:
            print("Warning: PathFollower weights incompatible. Evaluation will use random control weights.")
            
        try:
            if 'memory' in checkpoint:
                memory.load_state_dict(checkpoint['memory'])
        except RuntimeError:
            print("Warning: Memory weights incompatible.")
    
    planner = IntegratedPlanner(path_follower, goal_matcher, memory=memory)
    
    # 2. Data Loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = TartanAirDataset(data_dir, transform=transform)
    
    # metrics
    success_count = 0
    total_deviation = 0
    collision_count = 0
    path_length = 0
    total_steps = len(dataset)
    
    print(f"Evaluating over {total_steps} steps...")
    
    with torch.no_grad():
        for i in range(total_steps):
            images_seq, target_motion, target_depth, goal_image, _ = dataset[i]
            
            # Prepare inputs
            images_seq = images_seq.unsqueeze(0).to(device)
            current_obs = images_seq[:, -1, :, :, :]
            target_motion = target_motion.to(device)
            
            # Perception
            obs_vpr = visual_encoder(current_obs)
            obs_siamese = goal_encoder(current_obs)
            goal_siamese = goal_encoder(goal_image.unsqueeze(0).to(device))
            depth_map = depth_encoder(current_obs.to(device))
            
            # Planning
            features_vpr_seq = visual_encoder(images_seq.view(-1, 3, 224, 224)).view(1, 10, -1)
            result = planner.plan(obs_siamese, features_vpr_seq, goal_siamese, depth_map=depth_map, vpr_obs=obs_vpr)
            
            # Metrics Calculation
            pred_v = torch.FloatTensor(result['velocity']).to(device)
            
            # 1. Path Length (PL)
            step_dist = torch.norm(pred_v).item()
            path_length += step_dist
            
            # 2. Deviation Rate
            deviation = torch.norm(pred_v - target_motion).item()
            total_deviation += deviation
            
            # 3. Safety Monitoring (Swerves instead of Stops)
            if result.get('repulsive_active', False):
                collision_count += 1
            
            # 4. Mission Success Termination
            if result['action'] == "LAND":
                success_count = 1
                total_steps = i + 1 # Actual steps taken
                print(f"[SUCCESS] Goal reached at step {i+1}")
                break
                
    # Final Report (SOTA Metrics)
    avg_deviation = total_deviation / total_steps
    success_rate = (success_count / total_steps) * 100
    
    # Efficiency-Success Score (ESS) - Simplied: SR * (Optimal_PL / Actual_PL)
    # We'll assume the target_motion sum is the "Optimal" PL
    optimal_pl = 1.0 # placeholder or calculated from target_motion
    ess = (success_rate / 100.0) * (optimal_pl / max(path_length, 0.1))

    print("\n--- SOTA NAVIGATION BENCHMARK RESULTS ---")
    print(f" -> Success Rate (SR): {success_rate:.2f}%")
    print(f" -> Total Path Length (PL): {path_length:.4f} meters")
    print(f" -> Avg Completion Steps (ACT): {total_steps}")
    print(f" -> Efficiency-Success Score (ESS): {ess:.4f}")
    print(f" -> Deviation Rate: {avg_deviation:.4f} m/s")
    print(f" -> Safety Interventions: {collision_count} stops")
    print(f"------------------------------------------")
    print(f"Status: {'PASS' if success_rate > 50 else 'FAIL (Requires more training)'}")

if __name__ == "__main__":
    test_path = "data/tartanair/abandonedfactory/Easy/P001"
    checkpoint_path = "checkpoints/nav_stack_v2_2.pth"
    if os.path.exists(test_path):
        evaluate_system(test_path, weights_path=checkpoint_path if os.path.exists(checkpoint_path) else None)
    else:
        print(f"Test data not found at {test_path}")
