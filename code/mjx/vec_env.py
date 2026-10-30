"""MJX 向量化环境封装 (独立实现, 无第三方学习器框架依赖)

将 RobotIKAvoidEnvMJX.step_batch 包装为批量训练接口:
    - GPU 上全量世界一次 step
    - terminated/truncated 世界自动批量 reset (reset_batch, 分桶定形)
    - 支持 env_method('set_obstacle_stage', ...) 课程广播

历史上本类继承 stable_baselines3 VecEnv 以兼容 SB3 PPO.learn;
SB3 训练路径移除后改为独立实现, 接口面保持不变 (train_gpu.py /
GpuPPO.collect 依赖 reset/step/env_method/observation_space/
action_space)。

用法:
    env = MJXAvoidVecEnv(cfg, n_worlds=2048, stage=0)
"""
import numpy as np
import gymnasium as gym

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mjx.env_mjx import RobotIKAvoidEnvMJX


class MJXAvoidVecEnv:
    """单进程 GPU 向量化环境 (num_envs = n_worlds)"""

    def __init__(self, cfg, n_worlds=2048, stage=0, seed=0, max_points=768):
        self.mjx = RobotIKAvoidEnvMJX(cfg=cfg, n_worlds=n_worlds,
                                      stage=stage, max_points=max_points)
        self.mjx.rng = np.random.default_rng(seed)
        self.cfg = cfg
        obs_dim = self.mjx.obs_dim
        self.num_envs = n_worlds
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        action_dim = self.mjx.n_ctrl + int(cfg.env.learned_step_scale)
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        self._actions = None
        self.last_collision_cost = np.zeros(n_worlds, dtype=np.float32)
        self.last_safety_cost = np.zeros(n_worlds, dtype=np.float32)
        self.last_has_obstacle = np.zeros(n_worlds, dtype=bool)
        self._obs = self.mjx.reset_batch(np.arange(n_worlds))

    # ---------- 批量交互接口 ----------

    def reset(self):
        self._obs = self.mjx.reset_batch(np.arange(self.num_envs))
        return self._obs

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        backtrace_collision = (
            self.cfg.train.collision_backtrace_steps > 0
            and self.cfg.train.collision_backtrace_penalty > 0)
        if self.cfg.train.constrained_ppo:
            obs, rewards, terms, truncs, min_dist, safety_cost = \
                self.mjx.step_batch(
                    self._actions, return_min_dist=True,
                    return_safety_cost=True)
            # Capture transition outcomes before auto-reset replaces worlds.
            # Successful episodes also terminate, but keep positive clearance.
            self.last_has_obstacle = np.asarray(self.mjx.n_obs > 0, dtype=bool)
            self.last_collision_cost = np.asarray(
                terms & (min_dist <= 0.0), dtype=np.float32)
            self.last_safety_cost = np.asarray(safety_cost, dtype=np.float32)
        elif backtrace_collision:
            obs, rewards, terms, truncs, min_dist = self.mjx.step_batch(
                self._actions, return_min_dist=True)
            self.last_has_obstacle = np.asarray(self.mjx.n_obs > 0, dtype=bool)
            self.last_collision_cost = np.asarray(
                terms & (min_dist <= 0.0), dtype=np.float32)
            self.last_safety_cost.fill(0.0)
        else:
            obs, rewards, terms, truncs = self.mjx.step_batch(self._actions)
            self.last_has_obstacle.fill(False)
            self.last_collision_cost.fill(0.0)
            self.last_safety_cost.fill(0.0)
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

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

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

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.mjx, attr_name)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.mjx, attr_name, value)
