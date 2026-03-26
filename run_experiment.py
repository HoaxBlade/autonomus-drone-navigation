import os
import sys
from train.trainer import NavigationTrainer

if __name__ == "__main__":
    print("STARTING HETEROGENEOUS EXPERIMENT RUNNER")
    print("--------------------------------------------------")

    # 1. Dataset Configuration with Automatic Fallback
    datasets_config = []
    
    # Check for lightweight real-world data
    if os.path.exists('data/tum_indoor'):
        datasets_config.append({'type': 'tum', 'path': 'data/tum_indoor'})
    if os.path.exists('data/euroc_mav'):
        datasets_config.append({'type': 'euroc', 'path': 'data/euroc_mav'})
        
    # Mandatory Fallback to local TartanAir if others missing
    if not datasets_config:
        print("[INFO] Lightweight datasets missing. Falling back to local TartanAir.")
        shibuya_path = "data/tartanair_shibuya/TartanAir_shibuya/RoadCrossing03"
        if os.path.exists(shibuya_path):
            datasets_config.append({'type': 'tartanair', 'path': shibuya_path})
        else:
            print("[ERROR] No datasets found in 'data/' directory. Please add data.")
            sys.exit(1)

    print(f"[STEP 1/2] Launching Multi-Task Training on {len(datasets_config)} dataset(s)...")
    try:
        trainer = NavigationTrainer(datasets_config=datasets_config, lr=1e-4, batch_size=2)
        
        for epoch in range(1, 11):
            avg_loss = trainer.train_epoch(epoch)
            print(f"Epoch {epoch} Complete. Average Multi-Task Loss: {avg_loss:.4f}")
            
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        sys.exit(1)

    print("\n[STEP 2/2] Launching Simulation Framework...")
    # NOTE: In Phase 7, the sim_runner.py is used for closed-loop testing.
    # To start simulation manually, run: python3 drone_navigator/sim_runner.py
    print("Pre-training steps finished. System ready for closed-loop evaluation.")
    print("--------------------------------------------------")
    print("Check 'checkpoints/' for saved weights and 'plots/' for visualizations.")
