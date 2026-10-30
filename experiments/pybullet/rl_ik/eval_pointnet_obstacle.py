import numpy as np
from stable_baselines3 import PPO

from ik_6d.rl_env_6d_obstacle import RobotIK6DEnv
from ik_6d.pointnet_features import PointNetFeatureExtractor


# =========================
# Environment
# =========================

env = RobotIK6DEnv()


# =========================
# Load trained model
# =========================

model = PPO.load(
    "./models/pointnet_obstacle/ppo_pointnet_final",
    env=env
)


# =========================
# Evaluation
# =========================

n_episodes = 200

success_count = 0
collision_episode_count = 0
collision_free_success_count = 0

final_pos_errors = []
final_orient_errors = []


for episode in range(n_episodes):

    obs, info = env.reset()

    done = False
    last_info = None

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        last_info = info

    # Episode 最终结果
    success = last_info.get("success", False)
    had_collision = last_info.get("had_collision", False)

    if success:
        success_count += 1

    if had_collision:
        collision_episode_count += 1

    if success and not had_collision:
        collision_free_success_count += 1

    final_pos_errors.append(
        last_info["pos_distance"]
    )

    final_orient_errors.append(
        last_info["orient_distance_deg"]
    )


# =========================
# Results
# =========================

print("\n==============================")
print("PointNet + PPO Evaluation")
print("==============================")

print(
    f"Success rate: "
    f"{success_count / n_episodes * 100:.1f}%"
)

print(
    f"Collision rate: "
    f"{collision_episode_count / n_episodes * 100:.1f}%"
)

print(
    f"Collision-free success rate: "
    f"{collision_free_success_count / n_episodes * 100:.1f}%"
)

print(
    f"Mean final position error: "
    f"{np.mean(final_pos_errors) * 1000:.2f} mm"
)

print(
    f"Mean final orientation error: "
    f"{np.mean(final_orient_errors):.2f} deg"
)

env.close()