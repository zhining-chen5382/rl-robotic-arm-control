"""避障 IK 参数配置 (dataclass + YAML)

所有超参集中在 configs/avoid.yaml, 本文件定义默认值与加载逻辑。
"""
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(_BASE_DIR, 'configs', 'avoid.yaml')


@dataclass
class RewardConfig:
    """奖励权重"""
    task_base: float = 2.006          # MAPPO-IK 耦合奖励常数
    task_scale: float = 1000.0        # distance -> reward 缩放
    task_positive_cap: float = 0.0    # 单步任务正奖励上限；<=0 表示不限制
    zero_positive_task_on_collision: bool = False  # 碰撞转移不保留正任务奖励
    w_progress: float = 10.0          # 耦合距离进步奖励
    w_safe: float = 0.6               # 安全距离塑形 (min_dist < safe_margin 时)
    w_collision: float = 50.0         # 碰撞惩罚
    w_joint_limit: float = 1.0        # 关节限位惩罚 (每步每关节)
    w_smooth: float = 0.05            # 动作平滑惩罚 ||a||^2
    step_penalty: float = 0.0005      # 每步时间惩罚
    success_reward: float = 100.0     # 成功奖励
    w_orient_progress: float = 0.0    # 姿态线性进步塑形 (0=关闭; 末端姿态收敛增强)
    w_rep: float = 0.0                # 方向性斥力塑形 (0=关闭): 逼近障碍时按接近速度惩罚
    w_risk_step: float = 0.0          # v22: 风险区内“逼近量×动态步长²”惩罚权重
    risk_step_margin: float = 0.10    # v22: 风险感知步长奖励生效距离 (m)
    risk_step_closing_scale: float = 0.10  # 逼近速度达到该值时惩罚强度饱和 (m/step)
    w_risk_recovery: float = 0.0      # v31: 远离障碍且接近目标时惩罚过小步长
    risk_recovery_opening_scale: float = 0.01
    risk_recovery_progress_scale: float = 0.01
    w_align: float = 0.0              # v7: 安全方向对齐奖励 (0=关闭): 危险区内 action·safe_dir>0 发奖
    gate_progress: bool = False       # v7: 危险区进度豁免: 进度项只奖不罚,  移除绕行税
    task_diff: bool = False           # v8: 任务项改势能差 task_scale*(prev_coupled-coupled), 回报沿轨迹望远镜相消, 消除早撞套利


@dataclass
class EnvConfig:
    """环境参数"""
    robot_id: str = 'panda'
    max_steps: int = 100
    pos_tolerance: float = 0.010      # 成功位置容差 10mm
    orient_tolerance_deg: float = 3.0 # 成功姿态容差 3°
    safe_margin: float = 0.03         # 安全距离塑形阈值 (3cm)
    success_margin: float = 0.015     # 成功要求的最小安全距离
    min_action_scale: float = 0.001   # 自适应动作最小尺度
    max_action_span: float = 0.49     # 自适应动作最大附加尺度
    learned_step_scale: bool = False  # v21: 第8维动作学习整步缩放系数
    learned_step_min: float = 0.10    # 第8维=-1时的最小步长倍率；+1时为1
    normalize_joint_direction: bool = False  # 动态步长时将前7维max-abs归一化
    orient_scale_weight: float = 0.2  # 步长输入中姿态误差权重 (位置收敛后保障姿态精调步长)
    brake_margin: float = 0.0         # 步长刹车: min_dist<此值时线性压步长 (0=关闭)
    brake_floor: float = 0.25         # 刹车下限系数 (最小步长 = scale*floor)
    brake_mode: str = 'global'        # v13: global=全关节同一系数(历史行为) /
                                      # per_joint=逐关节按各自远端连杆组最近距离刹车
    obs_obstacle_hint: bool = False   # 观测附加最近障碍方向+距离 (+4 维, 旧模型不兼容)
    obs_safe_joint_dir: bool = False  # 观测附加关节空间安全方向 (+7 维, 旧模型不兼容)
    obs_lookahead: bool = False       # v11: 观测附加前瞻 2 维 (上一步接近速度 + 下一步预测余量, 治单帧穿透)
    obs_link_delta: bool = False      # v24: 每个监控连杆的安全距离变化 (+L 维)
    obs_count_hint: bool = True       # v16: 观测含障碍数归一维 n_obs/3 (+1 维; 关闭=删维, 旧模型不兼容)
    train_inflate_margin: float = 0.0 # v12: 障碍物膨胀量 (m, 0=关闭): SDF/碰撞/观测统一按膨胀后几何计算,
                                      # 训练用膨胀值, 评测必须用 0 配置 (真实几何)
    rep_margin: float = 0.0           # 斥力塑形生效半径 (prev_min_dist < 此 值时, 0=关闭)
    align_margin: float = 0.15        # v7: 安全方向对齐奖励生效半径 (prev_min_dist < 此值)
    progress_mode: str = 'coupled'    # 进步奖励语义: coupled(耦合距离) / projection(目标轴投影, 绕行不惩罚)
    dist_clip: float = 0.5            # 观测中距离截断 (m)
    target_margin: float = 0.15       # 目标采样关节边距
    start_random_prob: float = 0.5    # 起始构型全随机概率 (否则 home+噪声)
    # 监控 link (pybullet link 索引): panda_link1~7 (0-6) + panda_hand (8)
    # + 左右手指 (9, 10) —— 手指是末端最前端部件, 必须纳入碰撞监控
    monitored_links: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 8, 9, 10])
    gui: bool = False


@dataclass
class ObstacleStage:
    """课程学习单个阶段"""
    name: str = 'stage'
    threshold: float = 0.0            # 训练进度阈值
    n_min: int = 0                    # 障碍物数量下限
    n_max: int = 0                    # 障碍物数量上限
    dist_min: float = 0.30            # 障碍物中心到基座-目标连线的最近侧向距离
    dist_max: float = 0.45            # 障碍物中心到基座-目标连线的最远侧向距离
    clearance: float = 0.10           # 与种子构型 link 的最小间隙
    # 可选的离散采样权重, 顺序对应 n_min..n_max; None 表示均匀采样。
    # 仅训练 reset 使用, 固定评测场景仍按 n_min..n_max 均匀生成。
    n_weights: Optional[List[float]] = None


@dataclass
class ObstacleConfig:
    """障碍物生成参数"""
    shapes: List[str] = field(default_factory=lambda: ['box', 'sphere', 'cylinder'])
    box_half_extents: List[float] = field(default_factory=lambda: [0.02, 0.10])   # 半边长范围 (m)
    sphere_radius: List[float] = field(default_factory=lambda: [0.03, 0.10])
    cylinder_radius: List[float] = field(default_factory=lambda: [0.02, 0.07])
    cylinder_height: List[float] = field(default_factory=lambda: [0.10, 0.40])
    target_clearance: float = 0.12    # 障碍物与目标点的最小距离
    start_clearance: float = 0.05     # 障碍物与起始构型的最小间隙
    query_range: float = 1.0          # getClosestPoints 查询范围 (m)
    max_attempts: int = 80            # 单障碍物采样最大尝试次数
    color: List[float] = field(default_factory=lambda: [0.85, 0.30, 0.25, 0.85])
    # 默认采样区域 (世界坐标), 保证落在机械臂工作空间附近
    region_x: List[float] = field(default_factory=lambda: [-0.2, 0.75])
    region_y: List[float] = field(default_factory=lambda: [-0.65, 0.65])
    region_z: List[float] = field(default_factory=lambda: [0.08, 0.95])
    # 课程阶段 (threshold 递增)
    stages: List[ObstacleStage] = field(default_factory=lambda: [
        ObstacleStage('无障碍 IK',      0.00, 0, 0, 0.0,  0.0,  0.0),
        ObstacleStage('单障碍-远',      0.20, 1, 1, 0.30, 0.45, 0.10),
        ObstacleStage('单障碍-近',      0.45, 1, 1, 0.12, 0.30, 0.05),
        ObstacleStage('多障碍随机环境', 0.70, 1, 3, 0.10, 0.35, 0.05),
    ])


@dataclass
class TrainConfig:
    """训练超参"""
    model_name: str = 'panda_ppo_ik_avoid'
    timesteps: int = 10_000_000
    n_envs: int = 16
    learning_rate: float = 2e-4
    lr_final: float = 1e-5
    n_steps: int = 2048
    batch_size: int = 1024
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    extra_action_init: float = 1.0     # 7->8维迁移时新增动作均值初值
    extra_action_log_std: float = -2.0 # 新增动作探索标准差的对数初值
    net_arch: List[int] = field(default_factory=lambda: [512, 512, 256, 128])
    eval_freq_steps: int = 250_000
    n_eval_episodes: int = 20
    checkpoint_freq_steps: int = 1_000_000
    clip_reward: float = 20.0
    spot_eval_n: int = 0               # 训练中大样本抽查场景数 (0=关闭; 建议 200)
    spot_eval_freq_steps: int = 15_000_000  # 抽查间隔环境步
    constrained_ppo: bool = False      # v25: 独立碰撞成本 + PPO-Lagrangian
    cost_gamma: float = 0.99
    cost_gae_lambda: float = 0.95
    cost_vf_coef: float = 0.5
    cost_limit_start: float = 0.08     # 障碍回合轨迹碰撞率约束（训练初期）
    cost_limit_final: float = 0.03     # 障碍回合轨迹碰撞率约束（训练末期）
    lagrange_init: float = 0.0
    lagrange_lr: float = 5.0           # 每个 rollout 按约束违反量更新
    lagrange_max: float = 5.0
    dense_safety_cost_weight: float = 0.0   # v25c: 危险逼近大步动作成本
    dense_safety_cost_margin: float = 0.05  # 生效安全距离 (m)
    dense_safety_closing_scale: float = 0.01  # 接近速度归一化尺度 (m/step)
    # v26: 仅在正常采样得到“有障碍”回合时，以此概率替换为独立失败库状态。
    # 例如正常场景 80% 有障碍，设为 0.25 后，总回合约 20% 来自失败库，
    # 同时原有 20% 无障碍回合完全保留。
    failure_state_bank_path: str = ''
    failure_state_obstacle_replace_prob: float = 0.0
    failure_state_min_clearance: float = 0.0
    # v27: 对当前 on-policy 回合碰撞前若干转移追加递减回溯惩罚。
    collision_backtrace_steps: int = 0
    collision_backtrace_penalty: float = 0.0
    collision_backtrace_decay: float = 0.8
    # v28: protect the migrated motion direction and train only one action
    # output (the learned global step scale) plus the complete value network.
    # "full" keeps the historical PPO behavior.
    actor_update_mode: str = 'full'
    step_action_index: int = -1
    step_risk_head: bool = False       # v30: independent risk-trend step residual
    step_risk_hidden: int = 16
    direction_risk_head: bool = False  # v33: gated residual for first 7 actions
    direction_risk_hidden: int = 32
    direction_risk_max_residual: float = 0.15
    direction_risk_margin_norm: float = 0.20  # 10cm / dist_clip(50cm)
    # Net2Wider 迁移到更宽网络时，给新增神经元施加极小扰动以打破完全
    # 对称；0 保持历史加载行为和严格函数等价。
    widen_noise_std: float = 0.0


@dataclass
class AvoidConfig:
    reward: RewardConfig = field(default_factory=RewardConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    obstacle: ObstacleConfig = field(default_factory=ObstacleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def orient_tolerance_rad(self):
        return float(self.env.orient_tolerance_deg * 3.141592653589793 / 180.0)


def _update_dataclass(dc, data: dict):
    """递归用 dict 更新 dataclass 字段 (忽略未知键)"""
    if not isinstance(data, dict):
        return
    for k, v in data.items():
        if not hasattr(dc, k):
            continue
        cur = getattr(dc, k)
        if isinstance(v, dict) and hasattr(cur, '__dataclass_fields__'):
            _update_dataclass(cur, v)
        else:
            setattr(dc, k, v)


def load_config(path: str = None) -> AvoidConfig:
    """加载配置; 文件不存在时使用默认值"""
    cfg = AvoidConfig()
    path = path or DEFAULT_CONFIG_PATH
    if os.path.exists(path):
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        _update_dataclass(cfg, data)
        # obstacle.stages: list of dict -> ObstacleStage
        raw_stages = (data.get('obstacle') or {}).get('stages')
        if raw_stages:
            cfg.obstacle.stages = [ObstacleStage(**s) for s in raw_stages]
    return cfg


def save_config(cfg: AvoidConfig, path: str):
    """保存配置快照 (训练时留档)"""
    import yaml
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(asdict(cfg), f, allow_unicode=True, sort_keys=False)
