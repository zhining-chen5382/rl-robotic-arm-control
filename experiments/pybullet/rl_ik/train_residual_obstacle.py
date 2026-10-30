"""第一轮短时间 Residual PPO 实验（硬上限 20,000 timesteps）。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from ik_6d.residual_callbacks import ResidualMetricsCallback
from ik_6d.residual_pointnet_features import ResidualPointNetFeatureExtractor
from ik_6d.rl_env_6d_residual_obstacle import ResidualObstacleEnv


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "residual_obstacle"
LOG_DIR = ROOT / "logs" / "residual_obstacle"
INIT_MODEL = MODEL_DIR / "residual_init.zip"
FINAL_MODEL = MODEL_DIR / "residual_final.zip"
TOTAL_TIMESTEPS = 20_000
SEED = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(seed: int):
    env = ResidualObstacleEnv(
        residual_scale=0.05,
        residual_penalty_weight=1e-4,
    )
    env.action_space.seed(seed)
    return Monitor(env)


def policy_kwargs() -> dict:
    return {
        "features_extractor_class": ResidualPointNetFeatureExtractor,
        "features_extractor_kwargs": {"point_feature_dim": 128},
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "activation_fn": torch.nn.Tanh,
        "log_std_init": -2.0,
    }


def create_untrained_model(env) -> PPO:
    return PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs(),
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="cpu",
        seed=SEED,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
    )


def prepare_init() -> None:
    if INIT_MODEL.exists():
        raise FileExistsError(f"拒绝覆盖已有 init 模型: {INIT_MODEL}")
    seed_everything(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(SEED)
    try:
        model = create_untrained_model(env)
        model.save(str(INIT_MODEL.with_suffix("")))
        print(f"未训练 residual PPO 已保存: {INIT_MODEL}")
        print("未调用 model.learn()")
    finally:
        env.close()


def train_from_init() -> None:
    if not INIT_MODEL.is_file():
        raise FileNotFoundError(f"请先生成并评估 init 模型: {INIT_MODEL}")
    forbidden_existing = [
        MODEL_DIR / "residual_5000_steps.zip",
        MODEL_DIR / "residual_10000_steps.zip",
        MODEL_DIR / "residual_15000_steps.zip",
        MODEL_DIR / "residual_20000_steps.zip",
        FINAL_MODEL,
    ]
    existing = [path for path in forbidden_existing if path.exists()]
    if existing:
        raise FileExistsError(f"拒绝覆盖已有 residual 输出: {existing}")

    seed_everything(SEED)
    env = make_env(SEED)
    try:
        model = PPO.load(
            str(INIT_MODEL),
            env=env,
            device="cpu",
            tensorboard_log=str(LOG_DIR),
        )
        callback = ResidualMetricsCallback(
            checkpoint_dir=MODEL_DIR,
            metrics_csv_path=LOG_DIR / "training_metrics.csv",
            checkpoint_steps=(5_000, 10_000, 15_000, 20_000),
            window_size=100,
            max_timesteps=TOTAL_TIMESTEPS,
            log_interval_steps=1_000,
            verbose=1,
        )
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            reset_num_timesteps=True,
            progress_bar=False,
        )
        if model.num_timesteps != TOTAL_TIMESTEPS:
            raise RuntimeError(
                f"训练必须恰好停止在 {TOTAL_TIMESTEPS}，实际为 {model.num_timesteps}"
            )
        model.save(str(FINAL_MODEL.with_suffix("")))
        print(f"20k 短训练完成并停止: {FINAL_MODEL}")
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-init", action="store_true")
    mode.add_argument("--train-from-init", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_init:
        prepare_init()
    else:
        train_from_init()


if __name__ == "__main__":
    main()
