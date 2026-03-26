import os
import subprocess
import sys

def run_pipeline():
    print("STARTING INDUSTRY-GRADE EXPERIMENT RUNNER (v2.4)")
    print("--------------------------------------------------")
    
    # 1. Training Phase
    print("\n[STEP 1/2] Launching Multi-Task Training (10 Epochs)...")
    train_cmd = "export PYTHONPATH=$PYTHONPATH:. && .venv/bin/python3 train/trainer.py"
    
    try:
        subprocess.run(train_cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print("[ERROR] Training failed. Aborting pipeline.")
        return

    # 2. Evaluation Phase
    print("\n[STEP 2/2] Launching Formal Evaluation with Trained Weights...")
    eval_cmd = "export PYTHONPATH=$PYTHONPATH:. && .venv/bin/python3 evaluate_metrics.py"
    # Note: evaluate_metrics.py now loads weights automatically if checkpoints exist.
    
    subprocess.run(eval_cmd, shell=True)
    
    print("\n--------------------------------------------------")
    print("EXPERIMENT COMPLETE")
    print("Check 'plots/' for trajectory and loss visualizations.")

if __name__ == "__main__":
    run_pipeline()
