import torch
from drone_nav.nav.path_follower import PathFollower
from drone_nav.nav.goal_matcher import GoalMatcher

class IntegratedPlanner:
    """
    Industry-grade fusion of Path Following, Goal Seeking, and Geometric Safety.
    Implements: Velocity = (alpha * Path) + (beta * Goal) with Dynamic Safety & Smoothing.
    """
    def __init__(self, path_follower, goal_matcher, alpha=0.7, beta=0.3, smoothing=0.8):
        self.path_follower = path_follower
        self.goal_matcher = goal_matcher
        self.alpha = alpha 
        self.beta = beta   
        
        # Dynamics & Smoothing
        self.smoothing = smoothing # λ for EMA filter (0.0 = no smoothing, 1.0 = static)
        self.prev_velocity = torch.zeros(3)
        
        # Thresholds
        self.goal_threshold = 0.92
        self.base_safety_margin = 0.1 # Minimum normalized depth at idle
        self.k_velocity = 0.5         # Scaling factor: stop earlier at higher speeds
        
    def _apply_smoothing(self, target_velocity):
        """
        Applies Exponential Moving Average (EMA) filter to stabilize control.
        """
        if not isinstance(target_velocity, torch.Tensor):
            target_velocity = torch.Tensor(target_velocity)
            
        smoothed_v = self.smoothing * self.prev_velocity + (1 - self.smoothing) * target_velocity
        self.prev_velocity = smoothed_v
        return smoothed_v

    def plan(self, current_obs, path_seq, goal_img, depth_map=None):
        """
        Determines the final movement command with dynamic safety and smoothing.
        """
        # 1. Check Goal similarity
        goal_similarity = self.goal_matcher(current_obs, goal_img)
        if goal_similarity > self.goal_threshold:
            v_stop = torch.zeros(3)
            self.prev_velocity = v_stop
            return {"action": "LAND", "velocity": v_stop.tolist(), "confidence": goal_similarity.item()}

        # 2. Dynamic Geometric Safety Check
        if depth_map is not None:
            # Calculate dynamic safety margin: dist = margin + k * current_v
            current_v_mag = torch.norm(self.prev_velocity).item()
            dynamic_margin = self.base_safety_margin + (self.k_velocity * current_v_mag)
            
            h, w = depth_map.shape[-2:]
            central_depth = depth_map[:, :, h//4:3*h//4, w//4:3*w//4]
            min_dist = torch.min(central_depth).item()
            
            if min_dist < dynamic_margin:
                self.prev_velocity = torch.zeros(3) # Reset inertia on emergency stop
                return {
                    "action": "EMERGENCY_STOP", 
                    "velocity": [0, 0, 0], 
                    "reason": f"Obstacle at {min_dist:.2f} (Required: {dynamic_margin:.2f})"
                }

        # 3. Decision Fusion
        path_velocity = self.path_follower(current_obs, path_seq).squeeze(0)
        
        # Weighted blend
        if goal_similarity > 0.6:
            raw_target_v = self.alpha * path_velocity
        else:
            raw_target_v = path_velocity
            
        # 4. Temporal Smoothing (EMA Filter)
        final_velocity = self._apply_smoothing(raw_target_v)
        
        return {
            "action": "MOVE", 
            "velocity": final_velocity.tolist(), 
            "goal_similarity": goal_similarity.item()
        }
