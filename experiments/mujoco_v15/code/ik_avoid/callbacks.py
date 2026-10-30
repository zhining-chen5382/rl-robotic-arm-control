"""训练回调 (无 PyBullet 依赖)

linear_schedule / ProgressCallback / ObstacleCurriculumCallback 为
train_avoid.py (PyBullet 版) 与 train_mjx.py 共用;
MJXBatchEvalCallback 为 MJX 训练专用的批量评测回调,
替代原 PyBullet DummyVecEnv + EvalCallback 组合。
"""
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ik_avoid.eval_stats import inject_scenes, rollout_eval, summarize


def linear_schedule(initial_value, final_value):
    def f(progress_remaining):
        return final_value + (initial_value - final_value) * progress_remaining
    return f


class ProgressCallback(BaseCallback):
    """训练进度打印"""
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.initial_timesteps = 0
        self.last_print = 0

    def _on_training_start(self):
        self.initial_timesteps = self.model.num_timesteps

    def _on_step(self):
        new_steps = self.model.num_timesteps - self.initial_timesteps
        progress = (new_steps / self.total_timesteps) * 100
        if progress - self.last_print >= 1.0:
            self.last_print = progress
            print(f"\n  [进度] {progress:.1f}% ({new_steps:,}/{self.total_timesteps:,})")
        return True


class ObstacleCurriculumCallback(BaseCallback):
    """障碍物课程学习: 按训练进度推进阶段 (env_method 广播)"""

    def __init__(self, total_timesteps, stages, verbose=0, start_stage=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.stages = stages
        self.initial_stage = int(np.clip(start_stage, 0, len(stages) - 1))
        self.last_stage = self.initial_stage - 1

    def _on_step(self):
        progress = self.model.num_timesteps / self.total_timesteps
        current = self.initial_stage
        for i, s in enumerate(self.stages):
            if progress >= s.threshold:
                current = i
        current = max(current, self.initial_stage)
        if current != self.last_stage:
            self.last_stage = current
            s = self.stages[current]
            try:
                self.training_env.env_method('set_obstacle_stage', current)
                print(f"\n  [障碍物课程] 阶段 {current+1}/{len(self.stages)}: "
                      f"{s.name} (障碍数 {s.n_min}~{s.n_max}, "
                      f"侧向距离 {s.dist_min:.2f}~{s.dist_max:.2f}m, "
                      f"间隙 {s.clearance:.2f}m)")
            except Exception as e:
                print(f"\n  [障碍物课程] 更新失败: {e}")
        return True


class MJXBatchEvalCallback(BaseCallback):
    """MJX 批量评测回调 (无 PyBullet 依赖)

    固定评测场景集 (训练开始时生成一次, 跨评测可比), 按训练步间隔
    做确定性批量 rollout, 记录障碍/无障碍两组严格成功率,
    并以两组严格成功率均值保存 best 模型。
    """

    def __init__(self, cfg, eval_freq, n_eval_episodes=20,
                 best_model_save_path=None, seed=9999,
                 max_points=768, verbose=0):
        super().__init__(verbose)
        self.cfg = cfg
        self.eval_freq = max(int(eval_freq), 1)
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.seed = seed
        self.max_points = max_points
        self.eval_env = None
        self.best_metric = -np.inf

    def _on_training_start(self):
        from mjx.env_mjx import RobotIKAvoidEnvMJX
        n_half = max(self.n_eval_episodes // 2, 1)
        self.eval_env = RobotIKAvoidEnvMJX(
            cfg=self.cfg, n_worlds=n_half, max_points=self.max_points)
        # 固定场景集: 障碍组与无障碍组各 n_half 个
        self.scenes_obs = self.eval_env.generate_scenes(
            n_half, self.seed, with_obstacles=True)
        self.scenes_free = self.eval_env.generate_scenes(
            n_half, self.seed + 1000, with_obstacles=False)

    def _on_step(self):
        # n_calls = 回调调用次数, 每次对应 n_worlds 个环境步
        # (与 SB3 EvalCallback 的 eval_freq 语义一致)
        if self.n_calls % self.eval_freq != 0:
            return True

        inject_scenes(self.eval_env, self.scenes_obs)
        r_obs = rollout_eval(self.eval_env, self.model, self.scenes_obs,
                             self.eval_env.max_steps)
        inject_scenes(self.eval_env, self.scenes_free)
        r_free = rollout_eval(self.eval_env, self.model, self.scenes_free,
                              self.eval_env.max_steps)

        s_obs = summarize(r_obs, 'eval_obstacles')['strict_success_rate']
        s_free = summarize(r_free, 'eval_free')['strict_success_rate']
        metric = 0.5 * (s_obs + s_free)

        self.logger.record('eval/strict_obstacles', s_obs)
        self.logger.record('eval/strict_free', s_free)
        print(f"\n  [MJX评测] @{self.num_timesteps:,} 环境步: "
              f"障碍严格 {s_obs*100:.1f}% | 无障碍严格 {s_free*100:.1f}%")

        if self.best_model_save_path and metric > self.best_metric:
            self.best_metric = metric
            os.makedirs(self.best_model_save_path, exist_ok=True)
            self.model.save(os.path.join(self.best_model_save_path,
                                         'best_model'))
            if self.verbose > 0:
                print(f"  [MJX评测] 新 best 模型 (均值 {metric*100:.1f}%) "
                      f"-> {self.best_model_save_path}")
        return True

    def _on_training_end(self):
        if self.eval_env is not None:
            self.eval_env = None


class MJXSpotEvalCallback(BaseCallback):
    """训练中大样本确定性抽查 (不参与 best 模型选择)

    训练期常规评测 (n=20) 噪声 ±5%, 不足以做决策; 本回调按较低频率
    (如每 15M 步) 用固定大场景集 (如 200) 跑完整口径抽查:
    严格成功率/终态碰撞率/轨迹碰撞率/中位误差, 供 go/no-go 判断。
    """

    def __init__(self, cfg, spot_n, eval_freq, seed=7777,
                 max_points=768, verbose=0):
        super().__init__(verbose)
        self.cfg = cfg
        self.spot_n = int(spot_n)
        self.eval_freq = max(int(eval_freq), 1)
        self.seed = seed
        self.max_points = max_points
        self.eval_env = None

    def _on_training_start(self):
        from mjx.env_mjx import RobotIKAvoidEnvMJX
        n_half = max(self.spot_n // 2, 1)
        self.eval_env = RobotIKAvoidEnvMJX(
            cfg=self.cfg, n_worlds=n_half, max_points=self.max_points)
        self.scenes_obs = self.eval_env.generate_scenes(
            n_half, self.seed, with_obstacles=True)
        self.scenes_free = self.eval_env.generate_scenes(
            n_half, self.seed + 2000, with_obstacles=False)

    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True
        inject_scenes(self.eval_env, self.scenes_obs)
        r_obs = rollout_eval(self.eval_env, self.model, self.scenes_obs,
                             self.eval_env.max_steps)
        inject_scenes(self.eval_env, self.scenes_free)
        r_free = rollout_eval(self.eval_env, self.model, self.scenes_free,
                              self.eval_env.max_steps)
        s_obs = summarize(r_obs, 'spot_obstacles')
        s_free = summarize(r_free, 'spot_free')
        self.logger.record('spot/strict_obstacles',
                           s_obs['strict_success_rate'])
        self.logger.record('spot/strict_free', s_free['strict_success_rate'])
        self.logger.record('spot/traj_collision_obstacles',
                           s_obs['trajectory_collision_rate'])
        print(f"\n  [抽查 n={self.spot_n}] @{self.num_timesteps:,} 步: "
              f"有障碍严格 {s_obs['strict_success_rate']*100:.1f}% "
              f"(轨迹碰撞 {s_obs['trajectory_collision_rate']*100:.1f}%) | "
              f"无障碍严格 {s_free['strict_success_rate']*100:.1f}%")
        return True

    def _on_training_end(self):
        self.eval_env = None
