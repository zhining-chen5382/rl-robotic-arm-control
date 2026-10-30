"""Episode-indexed deterministic reset seed scheduling for one Gymnasium env."""

from __future__ import annotations

import hashlib
import json
import random

import gymnasium as gym
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EpisodeSeedScheduleWrapper(gym.Wrapper):
    """Apply ``base_seed + episode_index`` at every reset.

    SB3's single-environment vector wrapper automatically calls ``reset()``
    after a terminal transition, usually without a seed.  This wrapper makes
    that call deterministic without changing observations, actions, rewards,
    termination flags, or the wrapped environment's step behavior.
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
    def _scene_fingerprint(env: gym.Env, observation) -> str:
        root = env.unwrapped
        obstacle = getattr(root, "obstacle_info", None)
        if obstacle is None:
            raise RuntimeError("obstacle_info is missing after reset")
        digest = hashlib.sha256()
        for key in ("state", "point_cloud"):
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
        del seed  # The experiment schedule is authoritative for every episode.
        episode_index = self.episode_index
        episode_seed = self.base_seed + episode_index
        seed_everything(episode_seed)
        observation, info = self.env.reset(seed=episode_seed, options=options)
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


__all__ = ["EpisodeSeedScheduleWrapper", "seed_everything"]
