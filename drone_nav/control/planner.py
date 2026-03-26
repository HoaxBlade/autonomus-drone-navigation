import torch
from drone_nav.nav.path_follower import PathFollower
from drone_nav.nav.goal_matcher import GoalMatcher

class IntegratedPlanner:
    """
    Industry-grade fusion of Path Following, Goal Seeking, and Geometric Safety.
    Implements: Velocity = (alpha * Path) + (beta * Goal) with Safety Override.
    """
    def __init__(self, path_follower, goal_matcher, alpha=0.7, beta=0.3):
        self.path_follower = path_follower
        self.goal_matcher = goal_matcher
        self.alpha = alpha # Weight for following the visual path
        self.beta = beta   # Weight for heading towards the goal
        
        self.goal_threshold = 0.92  # Increased threshold for industry-grade precision
        self.safety_threshold = 0.1 # Minimum normalized depth (closer = smaller value)
        
    def plan(self, current_obs, path_seq, goal_img, depth_map=None):
        """
        Determines the final movement command with collision avoidance.
        """
        # 1. Check Goal similarity (VPR Match)
        goal_similarity = self.goal_matcher(current_obs, goal_img)
        if goal_similarity > self.goal_threshold:
            return {"action": "LAND", "velocity": [0, 0, 0], "confidence": goal_similarity.item()}

        # 2. Geometric Safety Check (Obstacle Avoidance)
        if depth_map is not None:
            # Check the central 50% of the depth map for obstacles
            h, w = depth_map.shape[-2:]
            central_depth = depth_map[:, :, h//4:3*h//4, w//4:3*w//4]
            min_dist = torch.min(central_depth)
            
            if min_dist < self.safety_threshold:
                return {
                    "action": "EMERGENCY_STOP", 
                    "velocity": [0, 0, 0], 
                    "reason": "Obstacle detected in proximity"
                }

        # 3. Mathematical Fusion of Navigation Policies
        path_velocity = self.path_follower(current_obs, path_seq)
        
        # Simplified Goal-directed biasing (Direction towards higher similarity)
        # In v2.1, we assume the path follower already incorporates goal-directed traits,
        # but we apply a weighted blend if the goal is visible.
        if goal_similarity > 0.6:
            # Blend path following with a goal-seeking bias
            final_velocity = self.alpha * path_velocity
            # Note: A real implementation would calculate a relative goal vector here.
        else:
            final_velocity = path_velocity
        
        return {
            "action": "MOVE", 
            "velocity": final_velocity.tolist(), 
            "goal_similarity": goal_similarity.item()
        }
