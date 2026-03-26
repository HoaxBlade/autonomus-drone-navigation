import os
import sys
from train.trainer import NavigationTrainer

def run_experiment():
    print("STARTING HETEROGENEOUS EXPERIMENT RUNNER")
    print("--------------------------------------------------")

    # 1. Configuration for Lightweight Training (8GB RAM Optimization)
    # To re-enable TartanAir, simply add: {'type': 'tartanair', 'path': 'data/tartanair_shibuya/...'}
    datasets_config = [
        {'type': 'tum', 'path': 'data/tum_indoor'},
        {'type': 'euroc', 'path': 'data/euroc_mav'}
    ]

    print("[STEP 1/2] Launching Multi-Task Training...")
    try:
        trainer = NavigationTrainer(datasets_config=datasets_config, lr=1e-4, batch_size=2)
        
        for epoch in range(1, 11):
            avg_loss = trainer.train_epoch(epoch)
            print(f"Epoch {epoch} Complete. Average Multi-Task Loss: {avg_loss:.4f}")
            
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return

    print("\n[STEP 2/2] Launching Simulation Framework...")
    # NOTE: In Phase 7, the sim_runner.py is used for closed-loop testing.
    # To start simulation manually, run: python3 drone_navigator/sim_runner.py
    print("Pre-training steps finished. System ready for closed-loop evaluation.")
    print("--------------------------------------------------")
    print("Check 'checkpoints/' for saved weights and 'plots/' for visualizations.")

if __name__ == "__main__":
    run_experiment()
