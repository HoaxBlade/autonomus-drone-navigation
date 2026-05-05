import torch
from drone_nav.nav.path_follower import PathFollower
from drone_nav.nav.goal_matcher import GoalMatcher

class IntegratedPlanner:
    """
    Industry-grade fusion of Path Following, Goal Seeking, and Geometric Safety.
    Implements: Velocity = (alpha * Path) + (beta * Goal) with Dynamic Safety & Smoothing.
    """
    def __init__(self, path_follower, goal_matcher, memory=None, alpha=0.7, beta=0.3, smoothing=0.8):
        self.path_follower = path_follower
        self.goal_matcher = goal_matcher
        self.memory = memory
        self.alpha = alpha 
        self.beta = beta   
        self.h = None # Hidden state for memory
        
        # Dynamics & Smoothing
        self.smoothing = smoothing # λ for EMA filter (0.0 = no smoothing, 1.0 = static)
        self.prev_velocity = torch.zeros(3)
        
        # Thresholds
        self.goal_threshold = 0.85
        self.base_safety_margin = 0.1 # Minimum normalized depth at idle
        self.k_velocity = 0.5         # Scaling factor: stop earlier at higher speeds
        
    def _apply_smoothing(self, target_velocity):
        """
        Applies Exponential Moving Average (EMA) filter to stabilize control.
        """
        if not isinstance(target_velocity, torch.Tensor):
            target_velocity = torch.Tensor(target_velocity)
            
        # Ensure prev_velocity is on the same device as target_velocity
        self.prev_velocity = self.prev_velocity.to(target_velocity.device)
            
        smoothed_v = self.smoothing * self.prev_velocity + (1 - self.smoothing) * target_velocity
        self.prev_velocity = smoothed_v
        return smoothed_v

    def plan(self, current_obs, path_seq, goal_img, depth_map=None, vpr_obs=None):
        """
        Determines the final movement command using learned termination and local potential fields (VFH).
        """
        # 0. Set observations for different heads
        goal_matching_obs = current_obs
        path_following_obs = vpr_obs if vpr_obs is not None else current_obs

        # 1. Learned Mission Termination (Differentiable Classifier)
        goal_confidence = self.goal_matcher(goal_matching_obs, goal_img).item()
        
        if goal_confidence > self.goal_threshold:
            v_stop = torch.zeros(3)
            self.prev_velocity = v_stop
            return {"action": "LAND", "velocity": v_stop.tolist(), "confidence": goal_confidence}

        # 2. Local Potential Fields (VFH - Repulsive Vectors)
        repulsive_v = torch.zeros(3)
        if depth_map is not None:
            # We divide the depth map into sectors (Left, Center, Right)
            h, w = depth_map.shape[-2:]
            
            # Divide into 3 vertical sectors
            left_sector = depth_map[:, :, :, :w//3]
            center_sector = depth_map[:, :, :, w//3:2*w//3]
            right_sector = depth_map[:, :, :, 2*w//3:]
            
            l_mean = torch.mean(left_sector).item()
            c_mean = torch.mean(center_sector).item()
            r_mean = torch.mean(right_sector).item()
            
            # Repulsive logic: if a sector is "tight" (< 0.25), generate force in opposite direction
            safety_limit = 0.25
            force_magnitude = 0.4
            
            if l_mean < safety_limit:
                repulsive_v[1] -= force_magnitude # Push Right
            if r_mean < safety_limit:
                repulsive_v[1] += force_magnitude # Push Left
            if c_mean < safety_limit:
                repulsive_v[0] -= force_magnitude # Push Back
                if l_mean > r_mean:
                    repulsive_v[1] += force_magnitude 
                else:
                    repulsive_v[1] -= force_magnitude

        # 3. Decision Fusion (Attractive Path + Repulsive Obstacles)
        if self.memory is not None:
            mem_in = path_seq
            memory_out, self.h = self.memory(mem_in, self.h)
            path_velocity = self.path_follower(memory_out[:, -1, :], memory_out).squeeze(0)
        else:
            path_velocity = self.path_follower(path_following_obs, path_seq).squeeze(0)

        # Final control vector
        combined_v = path_velocity + repulsive_v
        
        # Scale down if approaching goal
        if goal_confidence > 0.6:
            combined_v *= (1.0 - (goal_confidence - 0.6) * 2) 

        # 4. Temporal Smoothing (EMA Filter)
        final_velocity = self._apply_smoothing(combined_v)
        
        return {
            "action": "MOVE", 
            "velocity": final_velocity.tolist(), 
            "goal_confidence": goal_confidence,
            "repulsive_active": torch.norm(repulsive_v).item() > 0
        }
