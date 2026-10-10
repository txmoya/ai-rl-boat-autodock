"""Toy 2D boat-docking environment (Gymnasium API).

The dock slip is at the origin. The boat should arrive at (0, 0)
pointing along +x (heading 0), moving slowly.
World frame: x forward toward the dock, y to the left, heading in radians.
"""
#imports 
import numpy as np 
import gymnasium as gym
from gymnasium import spaces


def wrap(a): ## math here forces all angles to be between -180 & 180 degrees
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class BoatDockEnv(gym.Env):
    def __init__(self, max_steps=500, dt=0.1, current=0.0, randomize=False):
        super().__init__()
        self.dt, self.max_steps = dt, max_steps
        self.current, self.randomize = current, randomize
        # Actions: throttle and rudder, both in [-1, 1]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # Observations: dock position in the boat's frame (ex, ey),
        # heading error as (cos, sin), surge speed, yaw rate
        high = np.array([60, 60, 1, 1, 5, 3], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    # helpers (6 values, from 3 important data points). IRL "sensors"but in sim'd neural net form 
    def _obs(self):
        dx, dy = -self.x, -self.y
        c, s = np.cos(self.psi), np.sin(self.psi)
        ex = c * dx + s * dy          # world -> boat frame rotation
        ey = -s * dx + c * dy
        he = wrap(0.0 - self.psi)
        obs = np.array([ex, ey, np.cos(he), np.sin(he), self.u, self.r], dtype=np.float32) # pack those 6 val's into 1 array
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # Gymnasium API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.x = rng.uniform(-25.0, -12.0)
        self.y = rng.uniform(-8.0, 8.0)
        self.psi = rng.uniform(-np.pi / 3, np.pi / 3)
        self.u, self.r, self.t = 0.0, 0.0, 0
        # Domain randomization (sim-to-real): vary the "physics"
        if self.randomize:
            self.drag = rng.uniform(0.2, 0.45)
            self.cur = rng.uniform(-0.15, 0.15)
        else:
            self.drag, self.cur = 0.3, self.current
        self.prev_dist = np.hypot(self.x, self.y)
        return self._obs(), {}

    def step(self, action):
        throttle, rudder = np.clip(action, -1.0, 1.0)
        dt = self.dt
        # Simple dynamics: thrust minus linear and quadratic drag.
        u_dot = 1.0 * throttle - self.drag * self.u - 0.05 * self.u * abs(self.u)
        # Rudder only turns boat when water flows past it (when there is speed )
        r_dot = 0.6 * rudder * self.u - 1.2 * self.r
        self.u += u_dot * dt
        self.r += r_dot * dt
        self.psi = wrap(self.psi + self.r * dt)
        self.x += self.u * np.cos(self.psi) * dt
        self.y += (self.u * np.sin(self.psi) + self.cur) * dt   # current pushes sideways
        self.t += 1

        dist = np.hypot(self.x, self.y)
        head_err = abs(wrap(self.psi))

        # reward system (& respective weights)
        reward = 2.0 * (self.prev_dist - dist)          # progress toward the dock
        reward -= 0.01                                  # time penalty; discourage stalling action
        reward -= 0.002 * float(throttle**2 + rudder**2)  # smooth, efficient control (discourage jerky movement.)
        if dist < 5.0:                                  # nearing the dock radius, heading matters
            reward -= 0.02 * head_err
        self.prev_dist = dist

        terminated, info = False, {"success": False}
        if dist < 1.5:
            terminated = True
            if abs(self.u) < 0.8 and head_err < 0.5:
                reward += 50.0                          # docked & aligned
                info["success"] = True
            else:
                reward -= 20.0                          # hit the dock too fast / crooked
        elif dist > 45.0 or self.x > 3.0:
            terminated = True                           # got lost / overshot slip
            reward -= 20.0
        truncated = self.t >= self.max_steps
        info["pose"] = (float(self.x), float(self.y), float(self.psi))
        return self._obs(), float(reward), terminated, truncated, info
