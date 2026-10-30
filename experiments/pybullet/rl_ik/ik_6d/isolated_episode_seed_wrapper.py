"""Episode-indexed scene seeding isolated from algorithm-side RNG streams."""

from __future__ import annotations

import hashlib
import json
import random

import gymnasium as gym
import numpy as np


class IsolatedEpisodeSeedWrapper(gym.Wrapper):
    """Seed only environment reset generation, then restore global RNG states.

    Torch is intentionally not imported or seeded here.  Python and NumPy
    global RNG states are restored in ``finally`` even if reset raises.
    """

    def __init__(self, env: gym.Env, *, base_seed: int):
        super().__init__(env)
        self.base_seed = int(base_seed)
        if self.base_seed < 0:
            raise ValueError("base_seed must be non-negative")
        self.episode_index = 0
        self.last_episode_seed: int | None = None
        self.reset_records: list[dict[str, object]] = []

    @staticmethod
    def _scene_fingerprint(env: gym.Env, observation: dict) -> str:
        root = env.unwrapped
        obstacle = getattr(root, "obstacle_info", None)
        if obstacle is None:
            raise RuntimeError("obstacle_info is missing after reset")
        digest = hashlib.sha256()
        # robot_keypoints is intentionally excluded: it is the treatment.
        for key in ("state", "point_cloud", "base_action"):
            array = np.asarray(observation[key])
            digest.update(key.encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(np.ascontiguousarray(array).tobytes())
        for name in ("target_pos", "target_quat", "current_joints"):
            array = np.asarray(getattr(root, name))
            digest.update(name.encode("ascii"))
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(
            json.dumps(obstacle, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        return digest.hexdigest()

    def reset(self, *, seed=None, options=None):
        del seed  # The episode-indexed experiment schedule is authoritative.
        episode_index = self.episode_index
        episode_seed = self.base_seed + episode_index
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            observation, info = self.env.reset(seed=episode_seed, options=options)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

        self.last_episode_seed = episode_seed
        self.reset_records.append(
            {
                "episode_index": episode_index,
                "seed": episode_seed,
                "scene_fingerprint": self._scene_fingerprint(self.env, observation),
            }
        )
        self.episode_index += 1
        return observation, info


__all__ = ["IsolatedEpisodeSeedWrapper"]
