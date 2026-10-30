"""
机械臂 6D 位姿逆运动学强化学习环境 (Gymnasium)

支持: Panda (7轴) — 7 DOF 对 6D 任务有 1 DOF 冗余

核心设计 (MAPPO-IK 论文方案):
  1. 观测空间 (n+10): joints(n) + pos_dir(3) + pos_dist(1) + orient_aa(3)
                       + cosine(1) + final(1) + step(1)
  2. 奖励: 乘法耦合 distance = pos_dist × cosine_dist + b
  3. 目标采样: 随机关节角 -> FK -> 6D 位姿，保证可达
  4. 姿态误差: 等效轴角 (axis * angle)，连续且无奇点
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
from robot_configs import get_robot_config


def quat_multiply(q1, q2):
    """四元数乘法: q1 * q2 (xyzw 格式)"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float32)


def quat_conjugate(q):
    """四元数共轭 (xyzw 格式)"""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def quat_normalize(q):
    """四元数归一化"""
    n = np.linalg.norm(q)
    if n < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return q / n


def quat_to_axis_angle(q):
    """四元数转等效轴角 (xyzw 格式) -> axis * angle (3D 向量)
    
    返回值:
        3D 向量，方向=旋转轴，模长=旋转角度 (弧度)
        angle 范围 [0, pi]
    """
    q = quat_normalize(q)
    # 确保 w >= 0 (取短路径旋转)
    if q[3] < 0:
        q = -q
    
    # angle = 2 * acos(w), axis = xyz / sin(angle/2)
    w = np.clip(q[3], -1.0, 1.0)
    angle = 2.0 * np.arccos(abs(w))
    
    sin_half = np.sqrt(1.0 - w * w)
    if sin_half < 1e-8:
        # 角度接近 0，无旋转
        return np.zeros(3, dtype=np.float32)
    
    axis = q[:3] / sin_half
    return (axis * angle).astype(np.float32)


def quat_angle(q1, q2):
    """两个四元数之间的角度差 (弧度, [0, pi])"""
    q_err = quat_multiply(q2, quat_conjugate(q1))
    aa = quat_to_axis_angle(q_err)
    return float(np.linalg.norm(aa))


class RobotIK6DEnv(gym.Env):
    """6D 位姿机械臂逆运动学 Gymnasium 环境 (MAPPO-IK)"""

    metadata = {"render_modes": []}

    def __init__(self, robot_id='panda', max_steps=100, pos_tolerance=0.005,
                 orient_tolerance=0.175, action_scale=None):
        """
        Args:
            robot_id: 机器人标识 (目前仅 panda)
            max_steps: 每轮最大步数 (默认 100)
            pos_tolerance: 位置容差 (m), 默认 5mm
            orient_tolerance: 姿态容差 (rad), 默认 0.175 rad ≈ 10 deg
            action_scale: 固定动作尺度 (None=自适应)
        """
        super().__init__()

        self.robot_id = robot_id
        self.config = get_robot_config(robot_id)
        self.max_steps = max_steps
        self.pos_tolerance = pos_tolerance
        self.orient_tolerance = orient_tolerance
        self._fixed_action_scale = action_scale

        # pybullet
        self.physics_client = None
        self.robot = None

        # obstacle
        self.obstacle_id = None
        self.obstacle_info = None

        self.num_obstacle_points = 512
        self.point_cloud_debug_id = None
        self.obstacle_point_cloud = np.zeros(
            (self.num_obstacle_points, 3),
            dtype=np.float32
        )

        # 关节信息
        self.controlled_joints = []
        self.joint_limits = []
        self.n_ctrl = self.config.get('num_axes', 7)
        self.ee_link_index = None

        # Home position
        self.home_position = np.array(
            self.config.get('home_position', [0.0] * self.n_ctrl), dtype=np.float32
        )

        # MAPPO-IK 观测空间: joints(n) + pos_dir(3) + pos_dist(1) + orient_aa(3)
        #                     + cosine(1) + final(1) + step(1) = n + 10
        obs_dim = self.n_ctrl + 10

        self.observation_space = spaces.Dict({
            "state": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            ),

            "point_cloud": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_obstacle_points, 3),
                dtype=np.float32
            )
        })

        # Action: normalized joint deltas ∈ [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_ctrl,), dtype=np.float32
        )

        # Episode state
        self.current_joints = np.zeros(self.n_ctrl, dtype=np.float32)
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.target_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.step_count = 0
        self.prev_pos_dist = 0.0
        self.prev_orient_dist = 0.0
        self.had_collision = False

        # 课程学习
        self._curriculum_margin = 0.15
    def _create_obstacle(self):
        # 1. 随机选择障碍物类型
        obstacle_type = np.random.choice(["sphere", "box"])

        # 2. 随机生成障碍物中心位置
        position = [
            np.random.uniform(0.25, 0.60),
            np.random.uniform(-0.30, 0.30),
            np.random.uniform(0.15, 0.60)
        ]

        # 3. 球体
        if obstacle_type == "sphere":
            radius = np.random.uniform(0.06, 0.12)

            collision_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=radius
            )

            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=radius,
                rgbaColor=[1, 0, 0, 1]
            )

            self.obstacle_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position
            )

            self.obstacle_info = {
                "type": "sphere",
                "position": position,
                "radius": radius
            }

        # 4. 长方体
        else:
            half_extents = [
                np.random.uniform(0.05, 0.12),
                np.random.uniform(0.05, 0.12),
                np.random.uniform(0.05, 0.15)
            ]

            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents
            )

            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=[0, 0, 1, 1]
            )

            self.obstacle_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position
            )

            self.obstacle_info = {
                "type": "box",
                "position": position,
                "half_extents": half_extents
            }
        
        self._sample_obstacle_point_cloud()

    def _remove_obstacle(self):
        if self.point_cloud_debug_id is not None:
            p.removeUserDebugItem(self.point_cloud_debug_id)
            self.point_cloud_debug_id = None
        if self.obstacle_id is not None:
            p.removeBody(self.obstacle_id)
            self.obstacle_id = None
            self.obstacle_info = None
            self.obstacle_point_cloud.fill(0.0)

    def _sample_sphere_point_cloud(self, center, radius, num_points):
        directions = np.random.normal(size=(num_points, 3))

        directions = directions / np.linalg.norm(
            directions,
            axis=1,
            keepdims=True
        )

        points = center + radius * directions

        return points.astype(np.float32)


    def _sample_box_point_cloud(self, center, half_extents, num_points):
        center = np.array(center, dtype=np.float32)
        half_extents = np.array(half_extents, dtype=np.float32)

        hx, hy, hz = half_extents

        points = np.zeros((num_points, 3), dtype=np.float32)

        # 6个面的面积，用于决定点落在哪个面
        face_areas = np.array([
            hy * hz,  # +x
            hy * hz,  # -x
            hx * hz,  # +y
            hx * hz,  # -y
            hx * hy,  # +z
            hx * hy   # -z
        ])

        face_probs = face_areas / np.sum(face_areas)

        faces = np.random.choice(
            6,
            size=num_points,
            p=face_probs
        )

        for i, face in enumerate(faces):

            x = np.random.uniform(-hx, hx)
            y = np.random.uniform(-hy, hy)
            z = np.random.uniform(-hz, hz)

            if face == 0:
                x = hx
            elif face == 1:
                x = -hx
            elif face == 2:
                y = hy
            elif face == 3:
                y = -hy
            elif face == 4:
                z = hz
            else:
                z = -hz

            points[i] = center + np.array([x, y, z])

        return points.astype(np.float32)
    
    def _sample_obstacle_point_cloud(self):
        if self.obstacle_info is None:
            return

        if self.obstacle_info["type"] == "sphere":

            center = np.array(
                self.obstacle_info["position"],
                dtype=np.float32
            )

            radius = self.obstacle_info["radius"]

            self.obstacle_point_cloud = self._sample_sphere_point_cloud(
                center,
                radius,
                self.num_obstacle_points
            )

        elif self.obstacle_info["type"] == "box":

            center = np.array(
                self.obstacle_info["position"],
                dtype=np.float32
            )

            half_extents = np.array(
                self.obstacle_info["half_extents"],
                dtype=np.float32
            )

            self.obstacle_point_cloud = self._sample_box_point_cloud(
                center,
                half_extents,
                self.num_obstacle_points
            )

    def _draw_obstacle_point_cloud(self):
        # 删除上一批显示的点
        if self.point_cloud_debug_id is not None:
            p.removeUserDebugItem(self.point_cloud_debug_id)
            self.point_cloud_debug_id = None

        # 如果没有点云就不画
        if self.obstacle_point_cloud is None:
            return

        points = self.obstacle_point_cloud.tolist()

        # 每一个点使用相同颜色
        colors = [[1.0, 1.0, 0.0]] * len(points)

        self.point_cloud_debug_id = p.addUserDebugPoints(
            pointPositions=points,
            pointColorsRGB=colors,
            pointSize=4
        )
    
    def _init_pybullet(self):
        if self.physics_client is not None:
            return

        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        urdf_path = self.config['urdf_path']
        self.robot = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )

        self.num_joints = p.getNumJoints(self.robot)
        self.controlled_joints = []
        self.joint_limits = []

        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot, i)
            if info[2] == p.JOINT_REVOLUTE:
                self.controlled_joints.append(i)
                self.joint_limits.append((info[8], info[9]))

        self.n_ctrl = len(self.controlled_joints)

        # EE link: 最后一个可控关节之后的固定关节
        last_ctrl = self.controlled_joints[-1]
        ee_idx = last_ctrl
        for i in range(last_ctrl + 1, self.num_joints):
            info = p.getJointInfo(self.robot, i)
            if info[2] == p.JOINT_FIXED:
                ee_idx = i
            else:
                break
        self.ee_link_index = ee_idx

    def _apply_joints(self, q):
        for i, jidx in enumerate(self.controlled_joints):
            p.resetJointState(self.robot, jidx, float(q[i]))
        p.performCollisionDetection()

    def _check_obstacle_collision(self):
        if self.obstacle_id is None:
            return False

        p.performCollisionDetection()

        contacts = p.getContactPoints(
            bodyA=self.robot,
            bodyB=self.obstacle_id
        )

        return len(contacts) > 0
    def _get_min_obstacle_distance(self, max_distance=0.20):
        if self.obstacle_id is None:
            return max_distance

        p.performCollisionDetection()

        closest_points = p.getClosestPoints(
            bodyA=self.robot,
            bodyB=self.obstacle_id,
            distance=max_distance
        )

        # 20 cm 范围内都没有任何接近点
        if len(closest_points) == 0:
            return max_distance

        # 每个 closest point 中第 8 项是 contact distance
        min_distance = min(
            point[8] for point in closest_points
        )

        return float(min_distance)

    def _compute_proximity_penalty(
        self,
        min_distance,
        safe_distance=0.10,
        weight=500.0
    ):
        # 10 cm 以外认为安全
        if min_distance >= safe_distance:
            return 0.0

        # 碰撞时 distance 可能 <= 0
        d = max(min_distance, 0.0)

        ratio = (safe_distance - d) / safe_distance

        penalty = -weight * (ratio ** 2)

        return float(penalty)

    def _is_target_blocked(self, margin=0.05):
        if self.obstacle_info is None:
            return False

        target = np.array(self.target_pos, dtype=np.float32)
        center = np.array(
            self.obstacle_info["position"],
            dtype=np.float32
        )

        if self.obstacle_info["type"] == "sphere":
            radius = self.obstacle_info["radius"]

            distance = np.linalg.norm(target - center)

            return distance <= radius + margin

        elif self.obstacle_info["type"] == "box":
            half_extents = np.array(
                self.obstacle_info["half_extents"],
                dtype=np.float32
            )

            relative = np.abs(target - center)

            return np.all(
                relative <= half_extents + margin
            )

        return False

    def _create_valid_obstacle(self, max_attempts=100):

        for _ in range(max_attempts):

            self._remove_obstacle()
            self._create_obstacle()

            collision = self._check_obstacle_collision()
            target_blocked = self._is_target_blocked()

            if not collision and not target_blocked:
                return

        raise RuntimeError(
            "Failed to generate a valid obstacle."
        )
        
    def _get_ee_pos(self):
        state = p.getLinkState(self.robot, self.ee_link_index)
        return np.array(state[4], dtype=np.float32)

    def _get_ee_quat(self):
        state = p.getLinkState(self.robot, self.ee_link_index)
        return np.array(state[5], dtype=np.float32)

    def _get_ee_pose(self):
        """返回 (position, quaternion)"""
        state = p.getLinkState(self.robot, self.ee_link_index)
        return np.array(state[4], dtype=np.float32), np.array(state[5], dtype=np.float32)

    def _clip_joints(self, q):
        for i in range(self.n_ctrl):
            lower, upper = self.joint_limits[i]
            q[i] = np.clip(q[i], lower, upper)
        return q

    def _sample_target(self):
        """随机关节角 + FK → 可达 6D 目标位姿"""
        random_q = np.zeros(self.n_ctrl, dtype=np.float32)
        margin = getattr(self, '_curriculum_margin', 0.15)
        for i in range(self.n_ctrl):
            lower, upper = self.joint_limits[i]
            m = (upper - lower) * margin
            random_q[i] = np.random.uniform(lower + m, upper - m)
        self._apply_joints(random_q)
        ee_pos, ee_quat = self._get_ee_pose()

        # 可选: 加小扰动增加鲁棒性
        pos_noise = np.random.uniform(-0.005, 0.005, size=3).astype(np.float32)  # ±5mm
        target_pos = ee_pos + pos_noise

        return target_pos, ee_quat.copy(), random_q

    def set_curriculum_margin(self, margin):
        """设置课程学习的采样边距"""
        self._curriculum_margin = margin

    def set_action_scale(self, scale):
        """设置固定动作尺度 (None=恢复自适应)"""
        self._fixed_action_scale = scale

    def _get_action_scale(self, distance):
        """自适应动作尺度: 距离越近, scale 越小
        近端 min_scale=0.003 保证 5mm 精度不 overshoot
        """
        if self._fixed_action_scale is not None:
            return self._fixed_action_scale
        scale = 0.003 + 0.49 * min(distance / 0.5, 1.0)
        return scale

    def _compute_orient_error(self):
        """计算当前姿态与目标姿态之间的等效轴角误差"""
        current_quat = self._get_ee_quat()
        # q_error = q_target * q_current^(-1)
        q_err = quat_multiply(self.target_quat, quat_conjugate(current_quat))
        return quat_to_axis_angle(q_err)

    def _get_obs(self):
        """MAPPO-IK 观测: joints(n) + pos_dir(3) + pos_dist(1) + orient_aa(3)
                        + cosine(1) + final(1) + step(1)"""
        ee_pos = self._get_ee_pos()
        pos_diff = self.target_pos - ee_pos
        pos_dist = float(np.linalg.norm(pos_diff))

        if pos_dist > 1e-8:
            pos_unit_dir = pos_diff / pos_dist
        else:
            pos_unit_dir = np.zeros(3, dtype=np.float32)

        # 姿态误差
        orient_error_aa = self._compute_orient_error()

        # MAPPO-IK 观测: joints + 相对位置方向 + 距离 + 姿态误差 + 综合距离指标 + 步数
        orient_dist = float(np.linalg.norm(orient_error_aa))
        cosine_dist = 2.0 - np.cos(orient_dist)      # ∈ [1, 3]
        final_dist = pos_dist * cosine_dist + (cosine_dist - 1.0)
        step_norm = self.step_count / self.max_steps
        state = np.concatenate([
            self.current_joints,
            pos_unit_dir,
            [pos_dist],
            orient_error_aa,
            [cosine_dist],
            [final_dist],
            [step_norm],
        ]).astype(np.float32)

        obs = {
            "state": state,
            "point_cloud": self.obstacle_point_cloud.copy()
        }

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_pybullet()

        self._remove_obstacle()

        self.target_pos, self.target_quat, target_q = self._sample_target()

        # 起始位置策略: home + 噪声 或 全随机
        if np.random.random() < 0.5:
            noise = np.random.uniform(-0.8, 0.8, size=self.n_ctrl).astype(np.float32)
            self.current_joints = self.home_position.copy() + noise
        else:
            self.current_joints = np.zeros(self.n_ctrl, dtype=np.float32)
            for i in range(self.n_ctrl):
                lower, upper = self.joint_limits[i]
                self.current_joints[i] = np.random.uniform(lower, upper)

        self.current_joints = self._clip_joints(self.current_joints)
        self._apply_joints(self.current_joints)
        self.step_count = 0
        self.had_collision = False
        self._create_valid_obstacle()

        ee_pos = self._get_ee_pos()
        self.prev_pos_dist = float(np.linalg.norm(ee_pos - self.target_pos))
        self.prev_orient_dist = quat_angle(self._get_ee_quat(), self.target_quat)
        self._prev_ee_pos = ee_pos.copy()

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # 归一化动作 [-1,1] × 自适应尺度
        action_clipped = np.clip(action, -1.0, 1.0).astype(np.float32)
        scale = self._get_action_scale(self.prev_pos_dist)
        delta_q = action_clipped * scale

        self.current_joints = self.current_joints + delta_q
        self.current_joints = self._clip_joints(self.current_joints)

        self._apply_joints(self.current_joints)
        collision = self._check_obstacle_collision()

        min_obstacle_distance = self._get_min_obstacle_distance()

        proximity_penalty = self._compute_proximity_penalty(
            min_obstacle_distance
        )

        ee_pos, ee_quat = self._get_ee_pose()

        pos_dist = float(np.linalg.norm(ee_pos - self.target_pos))
        orient_dist = quat_angle(ee_quat, self.target_quat)

        # ============ MAPPO-IK 论文奖励 (乘法耦合) ============
        # cosine_distance ∈ [1,3]: 姿态完全一致=1, 相差180°=3
        cosine_dist = 2.0 - np.cos(orient_dist)
        b = cosine_dist - 1.0
        # distance = 欧氏距离 × 余弦距离 + 偏移项
        distance = pos_dist * cosine_dist + b
        # 论文奖励公式: reward = 2 + 0.006 - distance × 1000
        ik_reward = 2.0 + 0.006 - distance * 1000.0
        reward = ik_reward
        # 每步小惩罚 (加速收敛)
        reward -= 0.0005
        # 障碍物安全距离惩罚
        reward += proximity_penalty
        collision_penalty = 0.0
        # 关节超限惩罚
        for i in range(self.n_ctrl):
            lower, upper = self.joint_limits[i]
            if self.current_joints[i] <= lower + 1e-6 or self.current_joints[i] >= upper - 1e-6:
                reward -= 1.0

        if collision:
            collision_penalty = -3000.0
            reward += collision_penalty
            self.had_collision = True
        info = {
            'pos_distance': pos_dist,
            'orient_distance': orient_dist,
            'orient_distance_deg': np.degrees(orient_dist),
            'ee_position': ee_pos.tolist(),
            'ee_orientation': ee_quat.tolist(),
            'target_position': self.target_pos.tolist(),
            'target_orientation': self.target_quat.tolist(),
            'scale': scale,

            # 避障相关
            'collision': collision,
            'had_collision': self.had_collision,
            'min_obstacle_distance': min_obstacle_distance,

            # reward 分解
            'ik_reward': ik_reward,
            'proximity_penalty': proximity_penalty,
            'collision_penalty': collision_penalty,

            'success': False,
        }

        terminated = False
        truncated = False

        # 成功判定: 位置+姿态同时达标
        if (
            not self.had_collision
            and pos_dist < self.pos_tolerance
            and orient_dist < self.orient_tolerance
        ):
            reward += 20.0  # 到达奖励
            terminated = True
            info['success'] = True
            info['pos_success'] = True
            info['orient_success'] = True

        if self.step_count >= self.max_steps and not terminated:
            truncated = True

        self.prev_pos_dist = pos_dist
        self.prev_orient_dist = orient_dist
        self._prev_ee_pos = ee_pos.copy()
        # Reward scaling: 保持奖励相对关系不变，只缩小数值尺度
        raw_reward = reward
        reward = reward * 0.001

        info['raw_reward'] = raw_reward
        info['scaled_reward'] = reward
        
        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except Exception:
                pass
            self.physics_client = None


# 注册环境
gym.register(
    id='PandaIK6DObstacle-v0',
    entry_point='ik_6d.rl_env_6d_obstacle:RobotIK6DEnv',
    max_episode_steps=100,
    kwargs={'robot_id': 'panda'},
)