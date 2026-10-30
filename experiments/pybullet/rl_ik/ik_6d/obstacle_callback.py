import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class ObstacleMetricsCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

        self.total_episodes = 0
        self.collision_episodes = 0
        self.success_episodes = 0
        self.collision_free_successes = 0

        self.min_distances = []

    def _on_step(self):

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):

            # 每一步记录机械臂与障碍物最近距离
            if "min_obstacle_distance" in info:
                self.min_distances.append(
                    float(info["min_obstacle_distance"])
                )

            # episode 结束时统计
            if done:
                self.total_episodes += 1

                had_collision = bool(
                    info.get("had_collision", False)
                )

                success = bool(
                    info.get("success", False)
                )

                if had_collision:
                    self.collision_episodes += 1

                if success:
                    self.success_episodes += 1

                if success and not had_collision:
                    self.collision_free_successes += 1

        # 写入 SB3 logger
        if self.total_episodes > 0:

            collision_rate = (
                self.collision_episodes
                / self.total_episodes
            )

            success_rate = (
                self.success_episodes
                / self.total_episodes
            )

            collision_free_success_rate = (
                self.collision_free_successes
                / self.total_episodes
            )

            self.logger.record(
                "obstacle/collision_rate",
                collision_rate
            )

            self.logger.record(
                "obstacle/success_rate",
                success_rate
            )

            self.logger.record(
                "obstacle/collision_free_success_rate",
                collision_free_success_rate
            )

        if len(self.min_distances) > 0:

            mean_min_distance = np.mean(
                self.min_distances[-1000:]
            )

            self.logger.record(
                "obstacle/mean_min_distance",
                mean_min_distance
            )

        return True