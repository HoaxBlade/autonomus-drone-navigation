import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import torch
from drone_nav.perception.encoders import PerceptionBackbone, VisualEncoder, GoalEncoder, DepthEncoder
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
    path_follower = PathFollower(input_dim=visual_encoder.output_dim).to(device)
    goal_matcher = GoalMatcher(input_dim=backbone.out_channels).to(device)
    
    # Load weights if provided
    if weights_path and os.path.exists(weights_path):
        print(f"Loading trained weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        backbone.load_state_dict(checkpoint['backbone'])
        visual_encoder.load_state_dict(checkpoint['visual_encoder'])
        goal_encoder.load_state_dict(checkpoint['goal_encoder'])
        depth_encoder.load_state_dict(checkpoint['depth_encoder'])
        path_follower.load_state_dict(checkpoint['path_follower'])
        goal_matcher.load_state_dict(checkpoint['goal_matcher'])
    
    planner = IntegratedPlanner(path_follower, goal_matcher)
    
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
    steps = len(dataset)
    
    print(f"Evaluating over {steps} steps...")
    
    with torch.no_grad():
        for i in range(steps):
            images_seq, target_motion, target_depth, goal_image = dataset[i]
            
            # Prepare inputs
            images_seq = images_seq.unsqueeze(0).to(device)
            current_obs = images_seq[:, -1, :, :, :]
            target_motion = target_motion.to(device)
            
            # Perception
            # 1. NetVLAD embedding (32768) for the path follower
            obs_vpr = visual_encoder(current_obs)
            # 2. Pooled embedding (512) for the goal matcher
            obs_siamese = goal_encoder(current_obs)
            # 3. Target goal embedding (512)
            goal_siamese = goal_encoder(goal_image.unsqueeze(0).to(device))
            
            depth_map = depth_encoder(current_obs.to(device))
            
            # Planning
            features_vpr_seq = visual_encoder(images_seq.view(-1, 3, 224, 224)).view(1, 10, -1)
            
            # Pass Siamese/Pooled features for goal-matching, and NetVLAD for path-following
            result = planner.plan(obs_siamese, features_vpr_seq, goal_siamese, depth_map=depth_map, vpr_obs=obs_vpr)
            
            # Metrics Calculation
            pred_v = torch.FloatTensor(result['velocity']).to(device)
            deviation = torch.norm(pred_v - target_motion).item()
            total_deviation += deviation
            
            if result['action'] == "EMERGENCY_STOP":
                collision_count += 1
            
            if result['action'] == "LAND":
                success_count += 1
                
    # Final Report
    avg_deviation = total_deviation / steps
    success_rate = (success_count / steps) * 100
    
    print("\nFINAL BENCHMARK RESULTS:")
    print(f" -> Success Rate (SR): {success_rate:.2f}%")
    print(f" -> Avg Path Deviation: {avg_deviation:.4f} m/s")
    print(f" -> Safety Interventions: {collision_count} stops")
    print(f"Status: {'PASS' if avg_deviation < 0.2 else 'FAIL'}")

if __name__ == "__main__":
    test_path = "data/tartanair/abandonedfactory/Easy/P001"
    checkpoint_path = "checkpoints/nav_stack_v2_2.pth"
    if os.path.exists(test_path):
        evaluate_system(test_path, weights_path=checkpoint_path if os.path.exists(checkpoint_path) else None)
    else:
        print(f"Test data not found at {test_path}")
