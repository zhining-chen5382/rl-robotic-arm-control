from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from ik_6d.rl_env_6d_obstacle import RobotIK6DEnv
from ik_6d.pointnet_features import PointNetFeatureExtractor
from ik_6d.obstacle_callback import ObstacleMetricsCallback


# =========================
# Environment
# =========================
env = RobotIK6DEnv()


# =========================
# PointNet + PPO
# =========================
policy_kwargs = dict(
    features_extractor_class=PointNetFeatureExtractor,
    features_extractor_kwargs=dict(
        point_feature_dim=512
    )
)


model = PPO(
    "MultiInputPolicy",
    env,

    policy_kwargs=policy_kwargs,

    learning_rate=3e-4,

    n_steps=1024,
    batch_size=256,

    gamma=0.99,
    gae_lambda=0.95,

    clip_range=0.2,

    verbose=1,

    tensorboard_log="./logs/pointnet_obstacle/"
)


# =========================
# Callbacks
# =========================

metrics_callback = ObstacleMetricsCallback()

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./models/pointnet_obstacle/",
    name_prefix="ppo_pointnet"
)

callback = CallbackList([
    metrics_callback,
    checkpoint_callback
])


# =========================
# Training
# =========================

print("================================")
print("开始 PointNet + PPO 避障训练")
print("================================")


model.learn(
    total_timesteps=100000,
    callback=callback
)


# =========================
# Save final model
# =========================

model.save(
    "./models/pointnet_obstacle/ppo_pointnet_final"
)


env.close()

print("训练完成！")