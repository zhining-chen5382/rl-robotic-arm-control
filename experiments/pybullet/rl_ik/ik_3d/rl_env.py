"""
机械臂逆运动学强化学习环境 (Gymnasium) — 渐进式迭代版

支持: UR5 (6轴), Panda (7轴)

核心设计:
  1. 归一化方向向量: obs 中使用 unit_direction 而非原始差值, 使策略尺度不变
  2. 动作缩放: action ∈ [-1,1]^n, 实际 delta_q = action * scale, scale 由距离自适应
  3. 随机起始距离: 训练时从各种距离开始, Agent 学会在任何尺度下都有效
  4. 推理时多轮渐进: 粗→细, 每轮从上一轮最优解继续, 逐步缩小 scale
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
from robot_configs import get_robot_config


class RobotIKEnv(gym.Env):
    """通用机械臂逆运动学 Gymnasium 环境 (渐进式迭代)"""

    metadata = {"render_modes": []}

    def __init__(self, robot_id='ur5', max_steps=40, tolerance=0.005):
        super().__init__()

        self.robot_id = robot_id
        self.config = get_robot_config(robot_id)
        self.max_steps = max_steps
        self.tolerance = tolerance

        # pybullet
        self.physics_client = None
        self.robot = None

        # 关节信息
        self.controlled_joints = []
        self.joint_limits = []
        self.n_ctrl = self.config.get('num_axes', 6)
        self.ee_link_index = None

        # Home position
        self.home_position = np.array(
            self.config.get('home_position', [0.0] * self.n_ctrl), dtype=np.float32
        )

        # Observation: joint_angles(n) + unit_direction(3) + distance(1) + target_pos(3) = n + 7
        obs_dim = self.n_ctrl + 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: normalized joint deltas ∈ [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_ctrl,), dtype=np.float32
        )

        # Episode state
        self.current_joints = np.zeros(self.n_ctrl, dtype=np.float32)
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.prev_distance = 0.0

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

    def _get_ee_pos(self):
        state = p.getLinkState(self.robot, self.ee_link_index)
        return np.array(state[4], dtype=np.float32)

    def _clip_joints(self, q):
        for i in range(self.n_ctrl):
            lower, upper = self.joint_limits[i]
            q[i] = np.clip(q[i], lower, upper)
        return q

    def _sample_target(self):
        """随机关节角度 + FK → 可达目标"""
        random_q = np.zeros(self.n_ctrl, dtype=np.float32)
        # 课程学习: margin 随训练进度变化
        margin = getattr(self, '_curriculum_margin', 0.15)
        for i in range(self.n_ctrl):
            lower, upper = self.joint_limits[i]
            m = (upper - lower) * margin
            random_q[i] = np.random.uniform(lower + m, upper - m)
        self._apply_joints(random_q)
        return self._get_ee_pos().copy(), random_q

    def set_curriculum_margin(self, margin):
        """设置课程学习的采样边距"""
        self._curriculum_margin = margin

    def _get_action_scale(self, distance):
        """自适应动作尺度: 距离越近, scale 越小"""
        # scale 从 0.5 (远距离) 线性衰减到 0.01 (近距离)
        scale = 0.01 + 0.49 * min(distance / 0.5, 1.0)
        return scale

    def _get_obs(self):
        """观测: [joint_angles(6), unit_direction(3), distance(1), target_pos(3)]"""
        ee_pos = self._get_ee_pos()
        diff = self.target_pos - ee_pos
        distance = float(np.linalg.norm(diff))

        if distance > 1e-8:
            unit_dir = diff / distance
        else:
            unit_dir = np.zeros(3, dtype=np.float32)

        obs = np.concatenate([
            self.current_joints,
            unit_dir,
            [distance],
            self.target_pos,
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_pybullet()

        self.target_pos, target_q = self._sample_target()

        # 随机起始: 有时从 home 附近开始, 有时从随机位置开始
        if np.random.random() < 0.5:
            noise = np.random.uniform(-0.8, 0.8, size=self.n_ctrl).astype(np.float32)
            self.current_joints = self.home_position.copy() + noise
        else:
            # 从完全随机位置开始 (模拟不同距离)
            self.current_joints = np.zeros(self.n_ctrl, dtype=np.float32)
            for i in range(self.n_ctrl):
                lower, upper = self.joint_limits[i]
                self.current_joints[i] = np.random.uniform(lower, upper)

        self.current_joints = self._clip_joints(self.current_joints)
        self._apply_joints(self.current_joints)
        self.step_count = 0

        ee_pos = self._get_ee_pos()
        self.prev_distance = float(np.linalg.norm(ee_pos - self.target_pos))
        self._prev_ee_pos = ee_pos.copy()  # 初始化上一步位置

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # 归一化动作 [-1,1] × 自适应尺度
        action_clipped = np.clip(action, -1.0, 1.0).astype(np.float32)
        scale = self._get_action_scale(self.prev_distance)
        delta_q = action_clipped * scale

        self.current_joints = self.current_joints + delta_q
        self.current_joints = self._clip_joints(self.current_joints)

        self._apply_joints(self.current_joints)
        ee_pos = self._get_ee_pos()

        distance = float(np.linalg.norm(ee_pos - self.target_pos))
        dist_improvement = self.prev_distance - distance

        # Reward 设计 (V3 - 精度优化版)
        reward = 0.0
        
        # 1. 距离缩小奖励 (核心驱动力)
        reward += 50.0 * dist_improvement
        
        # 2. 相对改善率 (尺度无关)
        if self.prev_distance > 1e-6:
            reward += 10.0 * (dist_improvement / self.prev_distance)
        
        # 3. 对数距离惩罚 (远距离时信号更强)
        reward -= np.log(1 + 10 * distance) * 5
        
        # 4. 方向一致性奖励
        if distance > 1e-6 and self.prev_distance > 1e-6:
            prev_dir = (self.target_pos - self._get_prev_ee_pos())
            prev_dir = prev_dir / (np.linalg.norm(prev_dir) + 1e-8)
            curr_dir = (self.target_pos - ee_pos)
            curr_dir = curr_dir / (distance + 1e-8)
            direction_alignment = float(np.dot(prev_dir, curr_dir))
            reward += 3.0 * max(direction_alignment, 0)
        
        # 5. 指数近距离奖励 (更强)
        reward += 20.0 * np.exp(-10.0 * distance)
        
        # 6. 多阈值精细调节奖励 (大幅提升近距离奖励)
        if distance < 0.10:  # 10cm
            reward += 80.0 * (0.10 - distance)
        if distance < 0.05:  # 5cm
            reward += 150.0 * (0.05 - distance)
        if distance < 0.02:  # 2cm
            reward += 300.0 * (0.02 - distance)
        if distance < 0.01:  # 1cm
            reward += 500.0 * (0.01 - distance)
        
        # 7. 10-20mm 惩罚区 (推动模型进入 10mm 以内)
        if 0.01 < distance < 0.02:  # 10-20mm 区间
            reward -= 20.0 * (distance - 0.01)  # 越接近 20mm 惩罚越大

        terminated = False
        truncated = False
        info = {
            'distance': distance,
            'ee_position': ee_pos.tolist(),
            'target_position': self.target_pos.tolist(),
            'scale': scale,
        }

        # 8. 到达奖励 (渐进式奖励)
        if distance < self.tolerance:
            # 距离越近奖励越大
            if distance < 0.005:  # 5mm 以内
                reward += 1500.0
            elif distance < 0.010:  # 10mm 以内
                reward += 1000.0
            else:  # tolerance 以内
                reward += 500.0
            terminated = True
            info['success'] = True

        if self.step_count >= self.max_steps and not terminated:
            # 超时惩罚
            reward -= 10.0
            truncated = True

        self.prev_distance = distance
        self._prev_ee_pos = ee_pos.copy()  # 保存用于下一步方向计算
        return self._get_obs(), reward, terminated, truncated, info

    def _get_prev_ee_pos(self):
        """获取上一步的末端位置"""
        if hasattr(self, '_prev_ee_pos'):
            return self._prev_ee_pos
        return self._get_ee_pos()

    def close(self):
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except Exception:
                pass
            self.physics_client = None


# 向后兼容别名
UR5IKEnv = lambda **kwargs: RobotIKEnv(robot_id='ur5', **kwargs)
PandaIKEnv = lambda **kwargs: RobotIKEnv(robot_id='panda', **kwargs)


# 注册环境
gym.register(
    id='UR5IK-v0',
    entry_point='rl_env:RobotIKEnv',
    max_episode_steps=40,
    kwargs={'robot_id': 'ur5'},
)

gym.register(
    id='PandaIK-v0',
    entry_point='rl_env:RobotIKEnv',
    max_episode_steps=40,
    kwargs={'robot_id': 'panda'},
)
