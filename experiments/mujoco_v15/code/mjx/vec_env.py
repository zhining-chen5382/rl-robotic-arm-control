"""SB3 兼容的 MJX 向量化环境封装

将 RobotIKAvoidEnvMJX.step_batch 包装为 stable_baselines3 VecEnv 接口:
    - GPU 上全量世界一次 step
    - terminated/truncated 世界自动批量 reset (reset_batch, 分桶定形)
    - 支持 env_method('set_obstacle_stage', ...) 课程广播
    - infos 携带 terminal_observation, 兼容 SB3 PPO/VecMonitor/VecNormalize

用法:
    env = MJXAvoidVecEnv(cfg, n_worlds=2048, stage=0)
    env = VecMonitor(env); env = VecNormalize(env, ...)
"""
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mjx.env_mjx import RobotIKAvoidEnvMJX


class MJXAvoidVecEnv(VecEnv):
    """单进程 GPU 向量化 VecEnv (num_envs = n_worlds)"""

    def __init__(self, cfg, n_worlds=2048, stage=0, seed=0, max_points=768):
        self.mjx = RobotIKAvoidEnvMJX(cfg=cfg, n_worlds=n_worlds,
                                      stage=stage, max_points=max_points)
        self.mjx.rng = np.random.default_rng(seed)
        self.cfg = cfg
        obs_dim = self.mjx.obs_dim
        observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(self.mjx.n_ctrl,), dtype=np.float32)
        super().__init__(num_envs=n_worlds,
                         observation_space=observation_space,
                         action_space=action_space)
        self._actions = None
        self._obs = self.mjx.reset_batch(np.arange(n_worlds))

    # ---------- SB3 VecEnv 必需接口 ----------

    def reset(self):
        self._obs = self.mjx.reset_batch(np.arange(self.num_envs))
        return self._obs

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        obs, rewards, terms, truncs = self.mjx.step_batch(self._actions)
        dones = terms | truncs
        infos = [{} for _ in range(self.num_envs)]
        if dones.any():
            done_idx = np.where(dones)[0]
            for i in done_idx:
                infos[i]['terminal_observation'] = obs[i]
            new_obs = self.mjx.reset_batch(done_idx)
            obs = obs.copy()
            obs[done_idx] = new_obs
        self._obs = obs
        return obs, rewards, dones, infos

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is not None:
            self.mjx.rng = np.random.default_rng(seed)
        return [seed]

    # ---------- 属性/方法透传 (课程回调依赖 env_method) ----------

    def env_method(self, method_name, *method_args, indices=None,
                   **method_kwargs):
        method = getattr(self.mjx, method_name)
        return [method(*method_args, **method_kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.mjx, attr_name)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.mjx, attr_name, value)

    def getattr_depth_check(self, name, already_found):
        return None

    def get_images(self):
        raise NotImplementedError

    def render(self, mode=None):
        pass
