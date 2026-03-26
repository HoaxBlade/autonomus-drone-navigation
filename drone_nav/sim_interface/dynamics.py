import numpy as np

class DroneDynamics:
    """
    Simulates 2nd-order drone dynamics: x_t = x_t-1 + v_t * dt
    Where v_t = v_t-1 + (target_v - damping * v_t-1) * dt
    """
    def __init__(self, dt=0.033, damping=0.1, max_speed=5.0):
        self.dt = dt
        self.damping = damping
        self.max_speed = max_speed
        
        # State: [x, y, z] and [vx, vy, vz]
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        
    def reset(self, position=None, velocity=None):
        """Resets the drone state."""
        self.position = np.array(position) if position is not None else np.zeros(3)
        self.velocity = np.array(velocity) if velocity is not None else np.zeros(3)
        
    def step(self, target_velocity):
        """
        Updates the state based on target velocity command.
        target_velocity: [vx, vy, vz]
        """
        # 1. Calculate Acceleration (simple proportional control toward target)
        # In a real drone, this would be the motor response
        accel = (target_velocity - self.damping * self.velocity)
        
        # 2. Update Velocity: v = v + a * dt
        self.velocity += accel * self.dt
        
        # 3. Clip to Max Speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
            
        # 4. Update Position: x = x + v * dt
        self.position += self.velocity * self.dt
        
        return self.position, self.velocity

    def get_state(self):
        return {
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "speed": float(np.linalg.norm(self.velocity))
        }
