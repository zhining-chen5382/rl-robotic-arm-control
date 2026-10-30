"""避障 IK 模块 (collision-aware IK)

包含:
  - config:       参数配置 (YAML 驱动)
  - eval_stats:   批量 rollout 评测与统计口径
  - callbacks:    课程学习 / 评测回调

注: 环境与求解器已迁移至 MJX 实现, 见 mjx/ (env_mjx, avoid_solvers)
"""
from .config import AvoidConfig, load_config

__all__ = ['AvoidConfig', 'load_config']
