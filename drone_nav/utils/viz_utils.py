import matplotlib.pyplot as plt
import numpy as np
import os

def plot_trajectory(pred_path, gt_path, save_path="plots/latest_eval.png"):
    """
    Plots the 2D (top-down) trajectory of the drone.
    pred_path: List of (x, y) coordinates from the model.
    gt_path: List of (x, y) coordinates from the dataset.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    pred_path = np.array(pred_path)
    gt_path = np.array(gt_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(gt_path[:, 0], gt_path[:, 1], 'g--', label='Ground Truth (Expert)')
    plt.plot(pred_path[:, 0], pred_path[:, 1], 'b-', label='Predicted (AI)')
    
    # Start/End markers
    plt.scatter(gt_path[0, 0], gt_path[0, 1], c='green', marker='o', label='Start')
    plt.scatter(gt_path[-1, 0], gt_path[-1, 1], c='red', marker='x', label='Goal')
    
    plt.title("Drone Navigation Trajectory: Predicted vs. Ground Truth")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    print(f"Trajectory plot saved to: {save_path}")
    plt.close()

def plot_loss_curves(loss_history, save_path="plots/loss_history.png"):
    """
    Plots the multi-task loss curves.
    loss_history: Dict with keys 'nav', 'depth', 'goal', 'total'.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    for key, values in loss_history.items():
        plt.plot(values, label=key.capitalize())
    
    plt.title("Multi-Task Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
