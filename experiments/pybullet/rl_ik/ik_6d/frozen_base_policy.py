"""只读、不可训练的原始 Panda 6D PPO IK 策略适配器。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "panda_ppo_ik_6d_map.zip"
)
STATE_DIM = 17
ACTION_DIM = 7


class FrozenBasePolicy:
    """仅开放确定性 ``predict`` 的冻结 PPO 包装。

    PPO 模型不连接训练环境，policy 被切换到 evaluation mode，全部参数均禁止
    计算梯度。调用 ``learn`` 会明确失败。
    """

    def __init__(self, model_path: str | Path = DEFAULT_BASE_MODEL_PATH):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"原始 PPO 模型不存在: {self.model_path}")

        self._model = PPO.load(
            str(self.model_path),
            device="cpu",
            custom_objects={"learning_rate": 1e-5},
        )
        self._validate_model_spaces()
        self._freeze()

    def _validate_model_spaces(self) -> None:
        observation_shape = getattr(self._model.observation_space, "shape", None)
        action_shape = getattr(self._model.action_space, "shape", None)
        if observation_shape != (STATE_DIM,):
            raise ValueError(
                f"原始 PPO observation 应为 ({STATE_DIM},)，实际为 {observation_shape}"
            )
        if action_shape != (ACTION_DIM,):
            raise ValueError(
                f"原始 PPO action 应为 ({ACTION_DIM},)，实际为 {action_shape}"
            )

    def _freeze(self) -> None:
        self._model.policy.set_training_mode(False)
        for parameter in self._model.policy.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.assert_frozen()

    def assert_frozen(self) -> None:
        if self._model.policy.training:
            raise RuntimeError("原始 PPO policy 意外处于 training mode")
        if any(
            parameter.requires_grad
            for parameter in self._model.policy.parameters()
        ):
            raise RuntimeError("原始 PPO 存在未冻结参数")

    @staticmethod
    def _validate_state(state: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (STATE_DIM,):
            raise ValueError(
                f"base policy state 应为 ({STATE_DIM},)，实际为 {state_array.shape}"
            )
        if not np.all(np.isfinite(state_array)):
            raise ValueError("base policy state 包含 NaN 或 Inf")
        return state_array

    def predict(self, state: np.ndarray) -> np.ndarray:
        """根据单个 17 维 state 返回确定性的 7 维 base action。"""
        self.assert_frozen()
        state_array = self._validate_state(state)
        with torch.inference_mode():
            action, _ = self._model.predict(state_array, deterministic=True)
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (ACTION_DIM,):
            raise RuntimeError(
                f"base policy 输出应为 ({ACTION_DIM},)，实际为 {action_array.shape}"
            )
        if not np.all(np.isfinite(action_array)):
            raise RuntimeError("base policy 输出包含 NaN 或 Inf")
        return np.clip(action_array, -1.0, 1.0).astype(np.float32)

    def learn(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("FrozenBasePolicy 是只读推理策略，禁止调用 learn()")


__all__ = [
    "ACTION_DIM",
    "DEFAULT_BASE_MODEL_PATH",
    "FrozenBasePolicy",
    "STATE_DIM",
]
