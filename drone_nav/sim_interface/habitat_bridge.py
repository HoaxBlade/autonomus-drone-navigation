import habitat
from habitat.core.simulator import Observations
import numpy as np
import torch

class HabitatBridge:
    """
    Bridge to interface our navigation modular system with Habitat-Sim.
    """
    def __init__(self, config_path="configs/tasks/imagenav_gibson.yaml"):
        self.env = habitat.Env(config=habitat.get_config(config_path))
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        observations = self.env.reset()
        return self._process_obs(observations)

    def step(self, action):
        """
        Executes an action in the simulator and returns processed observations.
        """
        # Convert our policy output (-1, 1) to simulator actions
        # Habitat usually uses discrete actions (MOVE_FORWARD, TURN_LEFT, etc.)
        # unless configured for continuous control.
        observations = self.env.step(action)
        return self._process_obs(observations)

    def _process_obs(self, obs: Observations):
        """
        Extract RGB and Depth as tensors.
        """
        rgb = torch.from_numpy(obs["rgb"]).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(obs["depth"]).unsqueeze(0).float()
        return rgb, depth

    def close(self):
        self.env.close()
