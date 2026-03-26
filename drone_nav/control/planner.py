import torch
from drone_nav.nav.path_follower import PathFollower
from drone_nav.nav.goal_matcher import GoalMatcher

class IntegratedPlanner:
    """
    Fuses Path Following and Goal Seeking into a single navigation strategy.
    """
    def __init__(self, path_follower, goal_matcher):
        self.path_follower = path_follower
        self.goal_matcher = goal_matcher
        self.goal_threshold = 0.85 # Similarity threshold to stop
        
    def plan(self, current_obs, path_seq, goal_img):
        """
        Determines the final movement command.
        """
        # 1. Get Path Following action
        path_action = self.path_follower(current_obs, path_seq)
        
        # 2. Check Goal similarity
        goal_similarity = self.goal_matcher(current_obs, goal_img)
        
        # 3. Decision Logic:
        # If very close to goal, prioritize landing/stopping.
        # Otherwise, follow the path but bias towards the goal if visible.
        
        if goal_similarity > self.goal_threshold:
            return {"action": "LAND", "velocity": [0, 0, 0]}
            
        # Weighted fusion (simplistic version)
        # In a real version, we'd use a gating network or RL to fuse these.
        final_velocity = path_action 
        
        return {"action": "MOVE", "velocity": final_velocity}
