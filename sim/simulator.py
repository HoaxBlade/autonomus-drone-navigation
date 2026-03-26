import matplotlib.pyplot as plt
import numpy as np
import time

class SimpleDroneSim:
    def __init__(self, goal_pos=(10, 10), obstacles=None):
        self.drone_pos = np.array([0.0, 0.0])
        self.goal_pos = np.array(goal_pos)
        self.obstacles = obstacles if obstacles else [] # List of (x, y, radius)
        self.path = [self.drone_pos.copy()]
        
    def step(self, velocity, dt=0.5):
        # Update position
        self.drone_pos += np.array(velocity) * dt
        self.path.append(self.drone_pos.copy())
        
        # Check collision
        for ox, oy, r in self.obstacles:
            dist = np.linalg.norm(self.drone_pos - np.array([ox, oy]))
            if dist < r:
                return "COLLISION"
                
        # Check goal
        if np.linalg.norm(self.drone_pos - self.goal_pos) < 0.5:
            return "GOAL_REACHED"
            
        return "FLYING"

    def visualize(self):
        plt.figure(figsize=(8, 8))
        path_arr = np.array(self.path)
        plt.plot(path_arr[:, 0], path_arr[:, 1], 'b-', label='Drone Path')
        plt.plot(self.goal_pos[0], self.goal_pos[1], 'gx', markersize=10, label='Goal')
        
        for ox, oy, r in self.obstacles:
            circle = plt.Circle((ox, oy), r, color='r', alpha=0.5)
            plt.gca().add_patch(circle)
            
        plt.xlim(-2, 12)
        plt.ylim(-2, 12)
        plt.grid(True)
        plt.legend()
        plt.title("Drone Navigation Simulation")
        plt.savefig("sim_result.png")
        print("Simulation result saved to sim_result.png")

if __name__ == "__main__":
    # Test simulation with a simple potential field-like behavior
    sim = SimpleDroneSim(obstacles=[(5, 5, 1), (3, 7, 1)])
    
    for _ in range(50):
        # Very basic logic: move towards goal
        to_goal = sim.goal_pos - sim.drone_pos
        to_goal = to_goal / np.linalg.norm(to_goal) * 0.5
        
        # Avoid obstacles (very simple)
        avoidance = np.zeros(2)
        for ox, oy, r in sim.obstacles:
            dist_vec = sim.drone_pos - np.array([ox, oy])
            dist = np.linalg.norm(dist_vec)
            if dist < r + 2:
                avoidance += (dist_vec / dist**2) * 2.0
                
        vel = to_goal + avoidance
        status = sim.step(vel)
        if status == "GOAL_REACHED":
            print("Reached Goal!")
            break
        elif status == "COLLISION":
            print("Crashed!")
            break
            
    sim.visualize()
