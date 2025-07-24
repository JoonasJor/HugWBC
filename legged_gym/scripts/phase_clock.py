import math
import numpy as np
import torch

class PhaseClock:
    """
    Stateful generator for HugWBC’s 2-dimensional contact-phase clock.
    On each update(), advances gait_idx by freq*dt, warps into stance/swing,
    and returns [sin(2π·foot0), sin(2π·foot1)].
    """

    def __init__(self, dt: float, device: str):
        self.gait_idx = 0.0
        self.dt = dt
        self.device = device

    def _warp(self, idx: float, duration: float) -> float:
        # map idx∈[0,1) into [0,1] with 0→0.5 stance and 0.5→1 swing
        if idx < duration:
            return idx * (0.5 / duration)
        else:
            return 0.5 + (idx - duration) * (0.5 / (1.0 - duration))

    def update(self, commands: np.ndarray) -> np.ndarray:
        """
        commands: 1D array of length ≥6, where
          commands[3] = gait frequency (Hz)
          commands[4] = phase offset for foot0 (fraction)
          commands[5] = stance duration   (fraction)
        Returns:
          2-element array [clock0, clock1] with values ≈[-1,1].
        """
        frequency = commands[3]
        phase_offset = commands[4]
        duration = commands[5]

        # advance global gait index
        self.gait_idx = (self.gait_idx + frequency * self.dt) % 1.0

        # raw foot phases
        foot0 = (self.gait_idx + phase_offset) % 1.0
        foot1 = self.gait_idx

        # warp into stance/swing envelope
        foot0 = self._warp(foot0, duration)
        foot1 = self._warp(foot1, duration)

        # sine of 2π·phase
        clock0 = math.sin(2 * math.pi * foot0)
        clock1 = math.sin(2 * math.pi * foot1)

        return torch.tensor([clock0, clock1], dtype=torch.float, device=self.device)
