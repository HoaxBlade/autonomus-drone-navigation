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
        Determines the final movement command with dynamic safety and smoothing.
        Args:
            current_obs: (N, D_pooled) - Siamese/Pooled embedding for goal matching
            path_seq: (N, T, D_vpr) - Sequence of NetVLAD embeddings for path
            goal_img: (1, D_pooled) - Targeted goal image embedding
            depth_map: (N, 1, H, w) - Estimated depth
            vpr_obs: (N, D_vpr) - NetVLAD embedding for path following (if different from obs)
        """
        # 0. Set observations for different heads
        goal_matching_obs = current_obs
        path_following_obs = vpr_obs if vpr_obs is not None else current_obs

        # 1. Check Goal similarity
        goal_similarity = self.goal_matcher(goal_matching_obs, goal_img)
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
            min_dist = torch.mean(central_depth).item()
            
            if min_dist < dynamic_margin:
                self.prev_velocity = torch.zeros(3) # Reset inertia on emergency stop
                return {
                    "action": "EMERGENCY_STOP", 
                    "velocity": [0, 0, 0], 
                    "reason": f"Obstacle at {min_dist:.2f} (Required: {dynamic_margin:.2f})"
                }

        # 3. Decision Fusion
        if self.memory is not None:
            # Memory expects (Batch, Seq, Dim)
            mem_in = path_seq
            memory_out, self.h = self.memory(mem_in, self.h)
            path_velocity = self.path_follower(memory_out[:, -1, :], memory_out).squeeze(0)
        else:
            path_velocity = self.path_follower(path_following_obs, path_seq).squeeze(0)
        
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
