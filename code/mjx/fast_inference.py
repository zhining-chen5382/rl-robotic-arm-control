"""纯 numpy 快速推理链路 (v16+): 与 avoid_solvers.RLAvoidSolverMJX 语义等价

动机: jax jit 单世界推理每步 ~10ms, 其中 XLA 逐次分发开销占大头 (kernel
本身是纯运动学: FK + 点云 SDF, 无物理仿真)。本模块把观测/step/碰撞复检
全部移植为 numpy 向量化实现, 策略权重直接从 SB3 格式 zip 的 policy.pth
读取 (torch state_dict, 不经 stable_baselines3), 单步 ~1ms 量级。

等价性保障:
- 几何常量 (FK 链/手/手指位姿、点云、关节限位、刹车掩码) 直接从
  RobotIKAvoidEnvMJX 实例拷贝, 不重复解析
- 观测/step 语义逐项对照 env_mjx._obs_batch / _step_batch (含 hint/
  safe_dir/lookahead 分支, 兼容 v15 系 68 维与 v16 系 56 维配置)
- 障碍物打包复刻 _specs_to_arrays (含 train_inflate_margin)
- 提速另靠 step 闭环点云抽稀 (infer_max_points): 策略输入用抽稀
  点云, 碰撞判定类路径 (轨迹复检/终态/DLS/早停复核) 始终全量点云,
  抽稀距离上偏只可判碰撞不可判安全 (见 README §6.11)
- 验收: 与 jax 链路同场景 A/B (统计等效 + 自洽复检零漏判, §6.11)
"""
import time

import numpy as np

from ik_6d.quat_utils import quat_normalize
from ik_avoid.policy_io import load_policy


# ==================== 四元数 (xyzw, 对齐 env_mjx 同名 jax 版) ====================

def _quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], axis=-1).astype(np.float32)


def _quat_conjugate(q):
    return q * np.array([-1.0, -1.0, -1.0, 1.0], dtype=np.float32)


def _quat_to_axis_angle(q):
    """xyzw -> 轴角向量 (短路径), 批量; 与 env_mjx.quat_to_axis_angle 逐式对齐"""
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(n < 1e-10, np.array([0, 0, 0, 1.0], dtype=np.float32),
                 q / np.maximum(n, 1e-30))
    q = np.where(q[..., 3:4] < 0, -q, q)
    w = np.clip(q[..., 3], -1.0, 1.0)
    angle = 2.0 * np.arccos(np.abs(w))
    sin_half = np.sqrt(np.maximum(1.0 - w * w, 1e-16))
    scale = np.where(sin_half < 1e-8, 2.0, angle / sin_half)
    return (q[..., :3] * scale[..., None]).astype(np.float32)


def _q_mul_w(q1, q2):
    """wxyz 四元数乘法 (批量)"""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def _q_rot_w(q, v):
    """wxyz 四元数旋转向量 (q: (...,4), v: (...,3))"""
    w = q[..., 0]
    t = 2.0 * np.cross(q[..., 1:4], v)
    return v + w[..., None] * t + np.cross(q[..., 1:4], t)


def _q_rot_w_manual(q, v):
    """与 _q_rot_w 同式的末轴 3D 专用实现，避免 np.cross 的通用轴开销。"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.stack([
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ], axis=-1)


def _q_mul_w_components(w1, x1, y1, z1, w2, x2, y2, z2):
    """_q_mul_w 的分量形式，避免循环内反复 stack 四元数。"""
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )


def _q_rot_w_components(w, x, y, z, vx, vy, vz):
    """_q_rot_w_manual 的分量形式，返回旋转后的 x/y/z 数组。"""
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _to_xyzw(qw):
    return np.stack([qw[..., 1], qw[..., 2], qw[..., 3], qw[..., 0]], axis=-1)


def _to_wxyz(qx):
    return np.stack([qx[..., 3], qx[..., 0], qx[..., 1], qx[..., 2]], axis=-1)


def _coupled_distance(pos_dist, orient_dist):
    cosine_dist = 2.0 - np.cos(orient_dist)
    return pos_dist * cosine_dist + (cosine_dist - 1.0)


# ==================== 快速环境状态机 ====================

class FastEnvState:
    """RobotIKAvoidEnvMJX 的 numpy 单世界推理端口 (仅观测/step, 无奖励)

    几何常量从参考 env 实例拷贝; 状态用普通 numpy 数组就地更新。
    """

    def __init__(self, ref_env, infer_max_points=None,
                 adaptive_infer_points=False, infer_min_points=128,
                 infer_near_distance=0.12, infer_full_distance=0.04,
                 skip_inactive_obstacle_slots=True,
                 defer_nearest_direction=True,
                 reuse_sdf_buffers=False,
                 fast_quat_rotate=True,
                 fused_fk=True,
                 batch_point_rotation=True,
                 stop_pos_tolerance=None,
                 stop_orient_tolerance_deg=None,
                 observation_max_steps=None):
        e = ref_env
        self.cfg = e.cfg
        ec = e.cfg.env
        self.max_steps = e.max_steps
        self.observation_max_steps = (
            self.max_steps if observation_max_steps is None
            else int(observation_max_steps))
        if self.observation_max_steps <= 0:
            raise ValueError('observation_max_steps must be positive')
        self.pos_tolerance = (e.pos_tolerance
                              if stop_pos_tolerance is None
                              else float(stop_pos_tolerance))
        self.orient_tolerance = (
            e.orient_tolerance
            if stop_orient_tolerance_deg is None
            else float(np.radians(stop_orient_tolerance_deg)))
        self.success_margin = ec.success_margin
        self.n_ctrl = e.n_ctrl
        self.n_link_obs = e.n_link_obs
        self.home_position = np.asarray(e.home_position, np.float32).copy()
        self.joint_lo = np.asarray(e.joint_lo, np.float32).copy()
        self.joint_hi = np.asarray(e.joint_hi, np.float32).copy()

        # FK 链常量 (wxyz)
        self._chain = [(np.asarray(p, np.float32).copy(),
                        np.asarray(q, np.float32).copy()) for p, q in e._chain]
        self._hand_off = (np.asarray(e._hand_off[0], np.float32).copy(),
                          np.asarray(e._hand_off[1], np.float32).copy())
        self._finger_off = np.asarray(e._finger_off, np.float32).copy()
        self._qz180 = np.asarray(e._qz180, np.float32).copy()

        # 监控组点云 (局部系) 与逐关节刹车掩码; 拼接为大点云做
        # 单次旋转 + 每插槽单次向量化 SDF (避免 10 组 Python 循环开销)
        # 双点云: 全量 (_full, 轨迹复检/终态碰撞判定等安全口径) +
        # 抽稀 (step 闭环提速; 组内均匀抽稀保空间覆盖)。抽稀使距离
        # 上偏 (可能漏报), 故只允许用在策略输入, 不得用于碰撞判定
        self.mon_groups = list(e.mon_groups)
        self._pts_full = [np.asarray(e._pts[g], np.float32)
                          for g in self.mon_groups]
        self._P_full = [len(p) for p in self._pts_full]

        def make_cloud(max_points):
            if max_points is None:
                pts = self._pts_full
            else:
                if max_points <= 0:
                    raise ValueError('点云预算必须为正整数或 None')
                pts = []
                for p in self._pts_full:
                    if len(p) <= max_points:
                        pts.append(p)
                    else:
                        idx = np.round(np.linspace(
                            0, len(p) - 1, max_points)).astype(int)
                        pts.append(np.ascontiguousarray(p[idx]))
            sizes = [len(p) for p in pts]
            return pts, np.concatenate(pts, axis=0), np.cumsum([0] + sizes)

        self.adaptive_infer_points = bool(adaptive_infer_points)
        self.infer_max_points = infer_max_points
        self.infer_min_points = int(infer_min_points)
        self.infer_near_distance = float(infer_near_distance)
        self.infer_full_distance = float(infer_full_distance)
        self.skip_inactive_obstacle_slots = bool(skip_inactive_obstacle_slots)
        self.defer_nearest_direction = bool(defer_nearest_direction)
        self.reuse_sdf_buffers = bool(reuse_sdf_buffers)
        self.fast_quat_rotate = bool(fast_quat_rotate)
        self.fused_fk = bool(fused_fk)
        self.batch_point_rotation = bool(batch_point_rotation)
        self._point_group_indices = {}
        self._rotate = _q_rot_w_manual if self.fast_quat_rotate else _q_rot_w
        # key=(用途, 除批次外的 shape, dtype); 每类只保留最大批次，避免轨迹
        # 复核的不同长度不断累积缓存。FastEnvState 本身是有状态单求解器。
        self._sdf_buffers = {}
        if self.infer_full_distance < 0:
            raise ValueError('infer_full_distance 必须 >= 0')
        if self.infer_near_distance <= self.infer_full_distance:
            raise ValueError('infer_near_distance 必须大于 infer_full_distance')

        fixed = make_cloud(infer_max_points)
        self._pts, self._pts_cat, self._group_slice = fixed
        self._P = [len(p) for p in self._pts]
        self._infer_clouds = {'fixed': fixed}
        self._infer_tier_labels = {'fixed': (
            'full' if infer_max_points is None else f'p{int(infer_max_points)}')}
        if self.adaptive_infer_points:
            high_budget = infer_max_points
            low_budget = self.infer_min_points
            if high_budget is not None:
                low_budget = min(low_budget, int(high_budget))
            self._infer_clouds.update({
                'low': make_cloud(low_budget),
                'high': fixed,
                'full': make_cloud(None),
            })
            self._infer_tier_labels.update({
                'low': f'p{low_budget}',
                'high': ('full' if high_budget is None else f'p{int(high_budget)}'),
                'full': 'full',
            })
        self.infer_tier_counts = {}
        self._pts_cat_full = np.concatenate(self._pts_full, axis=0)
        self._group_slice_full = np.cumsum([0] + self._P_full)
        # 各组包络球 (局部系中心 + 半径, 轨迹复检粗筛用; 基于全量点云)
        self._grp_local_c = np.stack(
            [p.mean(axis=0) for p in self._pts_full])
        self._grp_local_r = np.array([
            np.linalg.norm(p - c, axis=-1).max()
            for p, c in zip(self._pts_full, self._grp_local_c)], np.float32)
        self.brake_mode = ec.brake_mode
        self.brake_margin = ec.brake_margin
        self.brake_floor = ec.brake_floor
        self._distal_mask = (np.asarray(e._distal_mask, bool)
                             if ec.brake_mode == 'per_joint' else None)

        # 观测/动作缩放参数
        self.min_action_scale = ec.min_action_scale
        self.max_action_span = ec.max_action_span
        self.learned_step_scale = bool(ec.learned_step_scale)
        self.learned_step_min = float(ec.learned_step_min)
        self.normalize_joint_direction = bool(ec.normalize_joint_direction)
        self.orient_scale_weight = ec.orient_scale_weight
        self.dist_clip = ec.dist_clip
        self.obs_count_hint = ec.obs_count_hint
        self.obs_obstacle_hint = ec.obs_obstacle_hint
        self.obs_safe_joint_dir = ec.obs_safe_joint_dir
        self.obs_lookahead = ec.obs_lookahead
        self.obs_link_delta = ec.obs_link_delta
        self.train_inflate_margin = ec.train_inflate_margin
        self.inflate = ec.train_inflate_margin

        # 场景状态 (单世界)
        self.qpos = np.zeros(7, np.float32)
        self.target_pos = np.zeros(3, np.float32)
        self.target_quat = np.array([0, 0, 0, 1.0], np.float32)
        self.obs_pos = np.zeros((3, 3), np.float32)
        self.obs_pos[:, 2] = -1000.0
        self.obs_shape = np.zeros(3, np.int32)
        self.obs_he = np.zeros((3, 3), np.float32)
        self.obs_r = np.zeros(3, np.float32)
        self.obs_cyl = np.zeros((3, 2), np.float32)
        self.n_obs = 0
        self.step_count = 0
        self.prev_pos_dist = 0.0
        self.prev_orient_dist = 0.0
        self.prev_min_dist = 0.5
        self.prev_mon_dist = np.full(self.n_link_obs, 0.5, np.float32)

    def _infer_tier(self, prev_md):
        """按上一帧距离选择闭环点云；当前帧过近时会在 _obs 内立即升级。"""
        if not self.adaptive_infer_points:
            return 'fixed'
        if self.n_obs == 0:
            return 'low'
        if self.step_count == 0:
            return 'high'
        if prev_md <= self.infer_full_distance:
            return 'full'
        if prev_md <= self.infer_near_distance:
            return 'high'
        return 'low'

    def _count_infer_tier(self, tier):
        label = self._infer_tier_labels[tier]
        self.infer_tier_counts[label] = self.infer_tier_counts.get(label, 0) + 1

    def _sdf_buffer(self, name, shape, dtype):
        """取得批次维可增长的内部工作数组视图。"""
        dtype = np.dtype(dtype)
        key = (name, tuple(shape[1:]), dtype.str)
        buf = self._sdf_buffers.get(key)
        if buf is None or buf.shape[0] < shape[0]:
            buf = np.empty(shape, dtype=dtype)
            self._sdf_buffers[key] = buf
        return buf[:shape[0]]

    def _point_group_index(self, gslice):
        """拼接点云中每个点所属的监控组索引（按各档点云尺寸缓存）。"""
        sizes = tuple(int(x) for x in np.diff(gslice))
        index = self._point_group_indices.get(sizes)
        if index is None:
            index = np.repeat(
                np.arange(len(sizes), dtype=np.intp), sizes)
            self._point_group_indices[sizes] = index
        return index

    # ---------- 场景 ----------

    def specs_to_arrays(self, specs, inflate=True):
        """复刻 env._specs_to_arrays (含膨胀语义), 返回 3 插槽数组 + n 数"""
        pos = np.zeros((3, 3), np.float32)
        pos[:, 2] = -1000.0
        shape = np.zeros(3, np.int32)
        he = np.zeros((3, 3), np.float32)
        r = np.zeros(3, np.float32)
        cyl = np.zeros((3, 2), np.float32)
        for s, spec in enumerate(specs[:3]):
            pos[s] = spec['pos']
            if spec['shape'] == 'box':
                shape[s] = 0
                he[s] = spec['size']['halfExtents']
            elif spec['shape'] == 'sphere':
                shape[s] = 1
                r[s] = spec['size']['radius']
            else:
                shape[s] = 2
                cyl[s] = [spec['size']['radius'], spec['size']['height'] / 2]
        m = self.train_inflate_margin
        if inflate and m > 0:
            he = he + m
            r = r + m
            cyl = cyl + m
        return pos, shape, he, r, cyl, len(specs[:3])

    def set_scene(self, target_pos, target_quat, start_q, obstacle_specs=None,
                  specs_inflated=False):
        """布置场景并返回初始观测 (对齐 env._setup_world 语义)

        specs_inflated=True: specs 已含膨胀量 (从参考 env 张量还原而来),
        跳过二次膨胀"""
        self.qpos = np.clip(np.asarray(start_q, np.float32),
                            self.joint_lo, self.joint_hi).copy()
        self.target_pos = np.asarray(target_pos, np.float32).copy()
        qn = np.asarray(target_quat, np.float32)
        self.target_quat = (qn / (np.linalg.norm(qn) + 1e-12)).astype(np.float32)
        if obstacle_specs is None:
            obstacle_specs = []
        (self.obs_pos, self.obs_shape, self.obs_he, self.obs_r,
         self.obs_cyl, self.n_obs) = self.specs_to_arrays(
            obstacle_specs, inflate=not specs_inflated)
        self.step_count = 0
        self.infer_tier_counts = {}
        self.prev_min_dist = 0.5                 # env 初值; obs_batch 中不使用
        obs, pd, od, mon_d = self._obs(self.qpos, self.step_count,
                                       self.prev_min_dist)
        self.prev_pos_dist = float(pd[0])
        self.prev_orient_dist = float(od[0])
        self.prev_mon_dist = mon_d[0].copy()
        return obs[0]

    # ---------- 正运动学 ----------

    def _fk(self, q):
        """q (N,7) -> 与 env._fk_batch 同序的 10 元组"""
        if self.fused_fk:
            return self._fk_fused(q)
        N = q.shape[0]
        pos = np.zeros((N, 3), np.float32)
        quat = np.tile(np.array([1, 0, 0, 0.0], np.float32), (N, 1))
        link_pos, link_quat, j_pos, j_axis = [], [], [], []
        z_unit = np.array([0.0, 0.0, 1.0], np.float32)
        for i in range(7):
            p_off, q_off = self._chain[i]
            pos = pos + self._rotate(quat, np.broadcast_to(p_off, (N, 3)))
            quat = _q_mul_w(quat, np.broadcast_to(q_off, (N, 4)))
            j_pos.append(pos.copy())
            j_axis.append(self._rotate(quat, np.broadcast_to(z_unit, (N, 3))))
            ang = q[:, i]
            qj = np.stack([np.cos(ang / 2), np.zeros(N, np.float32),
                           np.zeros(N, np.float32), np.sin(ang / 2)], axis=-1)
            quat = _q_mul_w(quat, qj)
            link_pos.append(pos.copy())
            link_quat.append(quat)
        p_off, q_off = self._hand_off
        hand_pos = pos + self._rotate(quat, np.broadcast_to(p_off, (N, 3)))
        hand_quat_w = _q_mul_w(quat, np.broadcast_to(q_off, (N, 4)))
        fl_pos = hand_pos + self._rotate(hand_quat_w, np.broadcast_to(
            self._finger_off, (N, 3)))
        fr_pos = fl_pos
        fl_quat_w = hand_quat_w
        fr_quat_w = _q_mul_w(hand_quat_w, np.broadcast_to(self._qz180, (N, 4)))
        return (np.stack(link_pos, axis=1).astype(np.float32),
                np.stack(link_quat, axis=1).astype(np.float32),
                hand_pos.astype(np.float32), _to_xyzw(hand_quat_w).astype(np.float32),
                fl_pos.astype(np.float32), _to_xyzw(fl_quat_w).astype(np.float32),
                fr_pos.astype(np.float32), _to_xyzw(fr_quat_w).astype(np.float32),
                np.stack(j_axis, axis=1).astype(np.float32),
                np.stack(j_pos, axis=1).astype(np.float32))

    def _fk_fused(self, q):
        """分量化 FK：保持原运算顺序，消除关节循环内 stack/broadcast。"""
        N = q.shape[0]
        px = np.zeros(N, np.float32)
        py = np.zeros(N, np.float32)
        pz = np.zeros(N, np.float32)
        w = np.ones(N, np.float32)
        x = np.zeros(N, np.float32)
        y = np.zeros(N, np.float32)
        z = np.zeros(N, np.float32)
        link_pos = np.empty((N, 7, 3), np.float32)
        link_quat = np.empty((N, 7, 4), np.float32)
        j_pos = np.empty((N, 7, 3), np.float32)
        j_axis = np.empty((N, 7, 3), np.float32)
        zero = np.float32(0.0)
        one = np.float32(1.0)

        for i in range(7):
            p_off, q_off = self._chain[i]
            rx, ry, rz = _q_rot_w_components(
                w, x, y, z, p_off[0], p_off[1], p_off[2])
            px = px + rx
            py = py + ry
            pz = pz + rz
            w, x, y, z = _q_mul_w_components(
                w, x, y, z, q_off[0], q_off[1], q_off[2], q_off[3])
            j_pos[:, i, 0] = px
            j_pos[:, i, 1] = py
            j_pos[:, i, 2] = pz
            ax, ay, az = _q_rot_w_components(
                w, x, y, z, zero, zero, one)
            j_axis[:, i, 0] = ax
            j_axis[:, i, 1] = ay
            j_axis[:, i, 2] = az
            ang = q[:, i]
            half = ang / 2
            cw = np.cos(half)
            sz = np.sin(half)
            w, x, y, z = _q_mul_w_components(
                w, x, y, z, cw, zero, zero, sz)
            link_pos[:, i, 0] = px
            link_pos[:, i, 1] = py
            link_pos[:, i, 2] = pz
            link_quat[:, i, 0] = w
            link_quat[:, i, 1] = x
            link_quat[:, i, 2] = y
            link_quat[:, i, 3] = z

        p_off, q_off = self._hand_off
        rx, ry, rz = _q_rot_w_components(
            w, x, y, z, p_off[0], p_off[1], p_off[2])
        hpx, hpy, hpz = px + rx, py + ry, pz + rz
        hw, hx, hy, hz = _q_mul_w_components(
            w, x, y, z, q_off[0], q_off[1], q_off[2], q_off[3])
        rx, ry, rz = _q_rot_w_components(
            hw, hx, hy, hz,
            self._finger_off[0], self._finger_off[1], self._finger_off[2])
        fpx, fpy, fpz = hpx + rx, hpy + ry, hpz + rz
        rw, rxq, ryq, rzq = _q_mul_w_components(
            hw, hx, hy, hz,
            self._qz180[0], self._qz180[1],
            self._qz180[2], self._qz180[3])

        hand_pos = np.stack([hpx, hpy, hpz], axis=-1).astype(np.float32)
        hand_quat = np.stack([hx, hy, hz, hw], axis=-1).astype(np.float32)
        finger_pos = np.stack([fpx, fpy, fpz], axis=-1).astype(np.float32)
        finger_quat = np.stack([hx, hy, hz, hw], axis=-1).astype(np.float32)
        right_quat = np.stack([rxq, ryq, rzq, rw], axis=-1).astype(np.float32)
        return (link_pos, link_quat, hand_pos, hand_quat,
                finger_pos, finger_quat, finger_pos, right_quat,
                j_axis, j_pos)

    def _group_frames(self, fk):
        """各组 (pos, quat_wxyz), 对齐 env._group_frames"""
        frames = [(fk[0][:, g], fk[1][:, g]) for g in range(7)]
        frames.append((fk[2], _to_wxyz(fk[3])))
        frames.append((fk[4], _to_wxyz(fk[5])))
        frames.append((fk[6], _to_wxyz(fk[7])))
        return frames

    # ---------- SDF ----------

    def _slot_dist_dir(self, pw, s):
        """世界点 pw (...,3) 到插槽 s 的 (dist, dir) — 逐式对齐 env._sdf_and_dir"""
        c = self.obs_pos[s]
        shape = int(self.obs_shape[s])
        he = self.obs_he[s]
        p = pw
        if shape == 1:                                     # sphere
            cp = c - p
            n_cp = np.linalg.norm(cp, axis=-1, keepdims=True)
            dist = np.linalg.norm(p - c, axis=-1) - self.obs_r[s]
            dvec = cp / np.maximum(n_cp, 1e-9)
            return dist, dvec
        if shape == 0:                                     # box (轴对齐)
            qc = np.abs(p - c) - he
            dist = (np.linalg.norm(np.maximum(qc, 0), axis=-1)
                    + np.minimum(np.max(qc, axis=-1), 0))
            s_box = np.clip(p, c - he, c + he)
            n_sb = np.linalg.norm(s_box - p, axis=-1, keepdims=True)
            outside = n_sb > 1e-9
            k = np.argmax(qc, axis=-1)
            e_k = np.eye(3, dtype=np.float32)[k]
            dir_in = -np.sign(p - c) * e_k
            dvec = np.where(outside,
                            (s_box - p) / np.maximum(n_sb, 1e-9), dir_in)
            return dist, dvec
        # cylinder (z 轴)
        rr, hh = self.obs_cyl[s][0], self.obs_cyl[s][1]
        dxy = p[..., :2] - c[:2]
        rho = np.linalg.norm(dxy, axis=-1)
        q1 = rho - rr
        q2 = np.abs(p[..., 2] - c[2]) - hh
        dist = (np.linalg.norm(np.maximum(np.stack([q1, q2], axis=-1), 0),
                               axis=-1)
                + np.minimum(np.maximum(q1, q2), 0))
        dir_xy = np.where(rho[..., None] > 1e-9,
                          -dxy / np.maximum(rho[..., None], 1e-9), 0.0)
        dir_raw = np.concatenate([
            dir_xy * np.maximum(q1, 0)[..., None],
            (-np.sign(p[..., 2] - c[2])
             * np.maximum(q2, 0))[..., None]], axis=-1)
        dir_out = dir_raw / np.maximum(
            np.linalg.norm(dir_raw, axis=-1, keepdims=True), 1e-9)
        inside = (q1 < 0) & (q2 < 0)
        dir_in = np.where((q1 > q2)[..., None], dir_out,
                          np.stack([np.zeros_like(q1), np.zeros_like(q1),
                                    -np.sign(p[..., 2] - c[2])], axis=-1))
        return dist, np.where(inside[..., None], dir_in, dir_out)

    def _slot_dist(self, pw, s):
        """仅计算世界点到插槽的距离；公式与 _slot_dist_dir 的 dist 完全一致。"""
        c = self.obs_pos[s]
        shape = int(self.obs_shape[s])
        p = pw
        if shape == 1:                                     # sphere
            return np.linalg.norm(p - c, axis=-1) - self.obs_r[s]
        if shape == 0:                                     # box (轴对齐)
            qc = np.abs(p - c) - self.obs_he[s]
            return (np.linalg.norm(np.maximum(qc, 0), axis=-1)
                    + np.minimum(np.max(qc, axis=-1), 0))
        # cylinder (z 轴)
        rr, hh = self.obs_cyl[s][0], self.obs_cyl[s][1]
        dxy = p[..., :2] - c[:2]
        rho = np.linalg.norm(dxy, axis=-1)
        q1 = rho - rr
        q2 = np.abs(p[..., 2] - c[2]) - hh
        return (np.linalg.norm(np.maximum(np.stack([q1, q2], axis=-1), 0),
                               axis=-1)
                + np.minimum(np.maximum(q1, q2), 0))

    def _link_dists(self, fk, full=False, infer_tier=None):
        """各组到最近障碍 (dist (N,L), dir (N,L,3), point (N,L,3)),
        对齐 env._link_dists

        向量化: 全部组点云拼接后单次旋转, 每插槽对全点云一次 SDF,
        组内 argmin 得到逐组最近 (与全量逐点计算严格等价)。
        full=True 走全量点云 (碰撞判定安全口径, point 有效);
        full=False 走抽稀点云 (策略输入提速, point 返回 None)"""
        N = fk[0].shape[0]
        L = len(self.mon_groups)
        if self.skip_inactive_obstacle_slots and self.n_obs == 0:
            # 无障碍时 link distance 经 clip 后恒为 1、方向又乘 has_obs=0。
            # 直接返回对应值可保持策略观测不变，并跳过旋转点云和 3 次假 SDF。
            out_d = np.full((N, L), self.dist_clip, np.float32)
            out_v = np.zeros((N, L, 3), np.float32)
            out_p = np.zeros((N, L, 3), np.float32) if full else None
            return out_d, out_v, out_p
        frames = self._group_frames(fk)
        if full:
            pts, pts_cat, gslice = (self._pts_full, self._pts_cat_full,
                                    self._group_slice_full)
        else:
            infer_tier = infer_tier or 'fixed'
            pts, pts_cat, gslice = self._infer_clouds[infer_tier]
            self._count_infer_tier(infer_tier)
        Psum = pts_cat.shape[0]
        # 各组帧 -> 拼接点云的世界坐标 (N,Psum,3)
        pw_shape = (N, Psum, 3)
        pw_cat = (self._sdf_buffer('pw_cat', pw_shape, np.float32)
                  if self.reuse_sdf_buffers else np.empty(pw_shape, np.float32))
        if self.batch_point_rotation and N == 1:
            # 单构型闭环：把 10 次小数组旋转合并为一次。批量轨迹复核不走此路，
            # 避免为大 N 展开 (N,Psum,4) 的逐点四元数。
            group_pos = np.stack(
                [frames[g][0] for g in self.mon_groups], axis=1)
            group_quat = np.stack(
                [frames[g][1] for g in self.mon_groups], axis=1)
            point_group = self._point_group_index(gslice)
            pw_cat[:] = (self._rotate(
                group_quat[:, point_group, :], pts_cat[None, ...])
                + group_pos[:, point_group, :])
        else:
            for gi, g in enumerate(self.mon_groups):
                gpos, gquat = frames[g]
                a, b = gslice[gi], gslice[gi + 1]
                pw_cat[:, a:b] = (self._rotate(gquat[:, None, :],
                                               pts[gi][None, ...])
                                  + gpos[:, None, :])
        n_slots = self.n_obs if self.skip_inactive_obstacle_slots else 3
        if self.defer_nearest_direction:
            # 所有点只算标量距离，同时记录胜出的障碍插槽；方向延迟到每组
            # argmin 后，仅为 N*L 个真正进入观测的最近点计算。
            d_all = np.empty((N, Psum), np.float32)
            work_shape = (N, Psum)
            if self.reuse_sdf_buffers:
                slot_all = self._sdf_buffer('slot_all', work_shape, np.uint8)
                slot_all.fill(0)
                better_buf = self._sdf_buffer('better', work_shape, bool)
            else:
                slot_all = np.zeros(work_shape, np.uint8)
                better_buf = None
            for s in range(n_slots):
                d_s = self._slot_dist(pw_cat, s)
                if s == 0:
                    d_all = d_s
                else:
                    if self.reuse_sdf_buffers:
                        np.less(d_s, d_all, out=better_buf)
                        np.copyto(d_all, d_s, where=better_buf)
                        np.copyto(slot_all, np.uint8(s), where=better_buf)
                    else:
                        better = d_s < d_all
                        d_all = np.where(better, d_s, d_all)
                        slot_all = np.where(better, np.uint8(s), slot_all)
        else:
            d_all = np.empty((N, Psum), np.float32)
            v_all = np.empty((N, Psum, 3), np.float32)
            for s in range(n_slots):
                d_s, v_s = self._slot_dist_dir(
                    pw_cat, s)                            # (N,Psum),(N,Psum,3)
                # 逐插槽叠加: 保留跨插槽最小 (与 jnp.where(shape==...) 展开后一致)
                if s == 0:
                    d_all = d_s
                    v_all = v_s
                else:
                    better = d_s < d_all
                    d_all = np.where(better, d_s, d_all)
                    v_all = np.where(better[..., None], v_s, v_all)
        out_d = np.empty((N, L), np.float32)
        out_v = np.empty((N, L, 3), np.float32)
        if self.defer_nearest_direction:
            near_p_shape = (N, L, 3)
            near_slot_shape = (N, L)
            near_p = (self._sdf_buffer('near_p', near_p_shape, np.float32)
                      if self.reuse_sdf_buffers
                      else np.empty(near_p_shape, np.float32))
            near_slot = (self._sdf_buffer('near_slot', near_slot_shape, np.uint8)
                         if self.reuse_sdf_buffers
                         else np.empty(near_slot_shape, np.uint8))
        else:
            near_p = None
            near_slot = None
        out_p = (near_p if full and self.defer_nearest_direction
                 else np.empty((N, L, 3), np.float32) if full else None)
        rows = np.arange(N)[:, None]
        for gi in range(L):
            a, b = gslice[gi], gslice[gi + 1]
            i_min = a + np.argmin(d_all[:, a:b], axis=-1, keepdims=True)
            out_d[:, gi] = d_all[rows, i_min][:, 0]
            selected_p = pw_cat[rows, i_min][:, 0]
            if self.defer_nearest_direction:
                near_p[:, gi] = selected_p
                near_slot[:, gi] = slot_all[rows, i_min][:, 0]
            else:
                out_v[:, gi] = v_all[rows, i_min][:, 0]
                if full:
                    out_p[:, gi] = selected_p
        if self.defer_nearest_direction:
            for s in range(n_slots):
                selected = near_slot == s
                if selected.any():
                    _, selected_v = self._slot_dist_dir(near_p[selected], s)
                    out_v[selected] = selected_v
        return out_d, out_v, out_p

    def group_sphere_dists(self, qs):
        """构型批 qs (T,7) 下各组包络球到最近障碍插槽的保守距离 (T,L)

        球心 FK 距离 − 半径 − 障碍几何: 对凸障碍为真实下界,
        仅供轨迹碰撞粗筛 (>0 必然无碰撞; <=0 需点云精确复检)"""
        fk = self._fk(np.asarray(qs, np.float32))
        frames = self._group_frames(fk)
        T = qs.shape[0]
        L = len(self.mon_groups)
        out = np.full((T, L), 1e9, np.float32)
        active = [s for s in range(3) if self.obs_pos[s][2] > -900.0]
        if not active:
            return out
        for gi, g in enumerate(self.mon_groups):
            gpos, gquat = frames[g]
            cw = (self._rotate(gquat, self._grp_local_c[gi]) + gpos)  # (T,3)
            for s in active:
                d_c, _ = self._slot_dist_dir(cw, s)                 # 球心 SDF
                # 球面距离下界: 心距−半径 (带符号 SDF 在球心位于障碍内时
                # 不再紧, 但此时 d_c<=0 必报可疑, 交精确复检, 不漏判)
                out[:, gi] = np.minimum(out[:, gi],
                                        d_c - self._grp_local_r[gi])
        return out

    # ---------- 观测 ----------

    def _obs(self, q, step_count, prev_md, full=False):
        """对齐 env._obs_batch: 返回 (obs (N,D), pd, od, mon_d (N,L))

        full=True: 全量点云 (碰撞判定口径); 默认抽稀点云 (策略输入)"""
        q = np.atleast_2d(np.asarray(q, np.float32))
        prev_md = np.atleast_1d(np.asarray(prev_md, np.float32))
        fk = self._fk(q)
        hand_pos, hand_quat = fk[2], fk[3]
        target_pos = self.target_pos[None] if self.target_pos.ndim == 1 \
            else self.target_pos
        target_quat = self.target_quat[None] if self.target_quat.ndim == 1 \
            else self.target_quat

        pos_diff = target_pos - hand_pos
        pos_dist = np.linalg.norm(pos_diff, axis=-1)
        pos_dir = np.where(pos_dist[..., None] > 1e-8,
                           pos_diff / np.maximum(pos_dist[..., None], 1e-9),
                           0.0)
        orient_aa = _quat_to_axis_angle(_quat_multiply(
            target_quat, _quat_conjugate(hand_quat)))
        orient_dist = np.linalg.norm(orient_aa, axis=-1)
        cosine_dist = 2.0 - np.cos(orient_dist)

        infer_tier = None
        if full:
            mon_d, mon_v, mon_p = self._link_dists(fk, full=True)
        else:
            infer_tier = self._infer_tier(float(prev_md.min()))
            mon_d, mon_v, mon_p = self._link_dists(
                fk, infer_tier=infer_tier)
            # 低档当前帧已经进入近区时立即重算，避免低精度观测继续驱动下一步。
            current_md = float(mon_d.min())
            if infer_tier == 'low' and current_md <= self.infer_near_distance:
                infer_tier = 'high'
                mon_d, mon_v, mon_p = self._link_dists(
                    fk, infer_tier=infer_tier)
                current_md = float(mon_d.min())
            if (self.adaptive_infer_points and infer_tier != 'full'
                    and current_md <= self.infer_full_distance):
                mon_d, mon_v, mon_p = self._link_dists(
                    fk, infer_tier='full')
        has_obs = np.float32(1.0) if self.n_obs > 0 else np.float32(0.0)
        d_norm = np.clip(mon_d, -0.05, self.dist_clip) / self.dist_clip
        link_feat = np.concatenate([
            d_norm[..., None], mon_v * has_obs], axis=-1)
        link_feat = link_feat.reshape(-1, 4 * self.n_link_obs)

        N = q.shape[0]
        parts = [
            q.astype(np.float32),
            pos_dir.astype(np.float32),
            pos_dist[..., None].astype(np.float32),
            orient_aa.astype(np.float32),
            cosine_dist[..., None].astype(np.float32),
            link_feat.astype(np.float32),
        ]
        if self.obs_count_hint:
            parts.append(np.full((N, 1), self.n_obs / 3.0, np.float32))
        sc = np.atleast_1d(np.asarray(step_count, np.float32))
        # The rollout horizon may be extended for diagnosis while preserving
        # the time-feature distribution seen during 100-step training.
        time_feature = np.minimum(sc / self.observation_max_steps, 1.0)
        parts.append(time_feature[..., None].astype(np.float32))
        if self.obs_obstacle_hint or self.obs_safe_joint_dir:
            l_min = np.argmin(mon_d, axis=-1)
            rows = np.arange(N)
        if self.obs_obstacle_hint:
            hint_dir = mon_v[rows, l_min] * has_obs
            hint_d = mon_d[rows, l_min]
            hint_dn = (np.clip(hint_d, -0.05, self.dist_clip) / self.dist_clip
                       if self.n_obs > 0 else np.ones(N, np.float32))
            parts.extend([hint_dir.astype(np.float32),
                          hint_dn[..., None].astype(np.float32)])
        if self.obs_safe_joint_dir:
            joint_axes, joint_pos = fk[8], fk[9]
            u = mon_v[rows, l_min] * has_obs
            if mon_p is None:      # 抽稀路径不产 point, 补一次全量最近点
                _, _, mon_p = self._link_dists(fk, full=True)
            p_near = mon_p[rows, l_min]
            jv = np.cross(joint_axes, p_near[:, None, :] - joint_pos)
            grad_q = -np.einsum('njk,nk->nj', jv, u)
            safe_dir = grad_q / np.maximum(
                np.linalg.norm(grad_q, axis=-1, keepdims=True), 1e-9)
            parts.append((safe_dir * has_obs).astype(np.float32))
        if self.obs_lookahead:
            cur_md = mon_d.min(axis=-1)
            md_delta = (np.where(sc > 0, cur_md - prev_md, 0.0) * has_obs)
            vel = np.clip(md_delta, -0.05, 0.05) / 0.05 * has_obs
            pred = np.clip(cur_md + md_delta, -0.05, self.dist_clip) \
                / self.dist_clip
            pred = pred if self.n_obs > 0 else np.ones(N, np.float32)
            parts.extend([vel[..., None].astype(np.float32),
                          pred[..., None].astype(np.float32)])
        if self.obs_link_delta:
            link_delta = np.where(
                sc[:, None] > 0,
                mon_d - self.prev_mon_dist[None, :], 0.0)
            link_vel = (np.clip(link_delta, -0.05, 0.05) / 0.05
                        * has_obs)
            parts.append(link_vel.astype(np.float32))
        obs = np.concatenate(parts, axis=-1)
        return obs, pos_dist, orient_dist, mon_d

    # ---------- step ----------

    def step(self, action):
        """对齐 env._step_batch (仅推理所需: 新构型 + 观测 + min_dist)"""
        action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        expected_dim = self.n_ctrl + int(self.learned_step_scale)
        if action.shape != (expected_dim,):
            raise ValueError(
                f'policy action shape {action.shape} != expected {(expected_dim,)}')
        if self.learned_step_scale:
            joint_action = action[:self.n_ctrl]
            if self.normalize_joint_direction:
                joint_action = joint_action / max(
                    float(np.max(np.abs(joint_action))), 1e-6)
            step_scale = (self.learned_step_min
                          + (1.0 - self.learned_step_min)
                          * (float(action[self.n_ctrl]) + 1.0) * 0.5)
        else:
            joint_action = action
            step_scale = 1.0
        scale_input = max(self.prev_pos_dist,
                          self.orient_scale_weight * self.prev_orient_dist)
        scale = (self.min_action_scale + self.max_action_span
                 * min(scale_input / 0.5, 1.0))
        has_obs = self.n_obs > 0
        if self.brake_margin > 0 and self.brake_mode == 'per_joint':
            d_pad = np.where(self._distal_mask[None],
                             self.prev_mon_dist[None, None, :], 1e9)
            d_distal = d_pad.min(axis=-1)                          # (1,7)
            brake_j = (np.clip(d_distal / self.brake_margin,
                               self.brake_floor, 1.0)
                       if has_obs else np.ones((1, 7), np.float32))
            new_q = np.clip(
                self.qpos[None] + joint_action[None] * scale
                * step_scale * brake_j,
                            self.joint_lo, self.joint_hi)
        else:
            if self.brake_margin > 0 and has_obs:
                brake = np.clip(self.prev_min_dist / self.brake_margin,
                                self.brake_floor, 1.0)
                scale = scale * brake
            new_q = np.clip(
                self.qpos[None] + joint_action[None] * scale * step_scale,
                            self.joint_lo, self.joint_hi)
        new_q = new_q[0].astype(np.float32)
        self.qpos = new_q
        self.step_count += 1
        obs, pd, od, mon_d = self._obs(new_q[None], self.step_count,
                                       self.prev_min_dist)
        self.prev_pos_dist = float(pd[0])
        self.prev_orient_dist = float(od[0])
        self.prev_mon_dist = mon_d[0].copy()
        self.prev_min_dist = float(mon_d[0].min())
        return obs[0], float(mon_d[0].min())

    def min_dist_batch(self, qs):
        """构型批 qs (T,7) 的各组最近距离 (T,L) — 轨迹复检/_pose_metrics/DLS
        复检用; 全量点云 (碰撞判定安全口径, 不得抽稀)"""
        fk = self._fk(np.asarray(qs, np.float32))
        d, _, _ = self._link_dists(fk, full=True)
        return d

    def trajectory_collisions(self, qs):
        """轨迹碰撞数 (min_dist<=0 口径): 包络球下界粗筛 + 可疑帧精确复检

        下界 >0 的帧必然无碰撞 (球内点距障碍 >= 下界); 其余帧走
        与训练同源的全量点云 SDF, 口径不变只是免全帧精确计算"""
        if self.n_obs == 0:
            return 0
        qs = np.asarray(qs, np.float32)
        lb = self.group_sphere_dists(qs)                    # (T,L) 保守下界
        suspect = np.nonzero((lb.min(axis=-1) <= 0.0))[0]
        if len(suspect) == 0:
            return 0
        d = self.min_dist_batch(qs[suspect])
        return int((d.min(axis=-1) <= 0.0).sum())


# ==================== 快速求解器 ====================

class FastRLAvoidSolverMJX:
    """RLAvoidSolverMJX 的 numpy 加速版 (接口/返回字段一致)

    hybrid 模式复用 AvoidBaselineMJX 的 refine_safe_dls; 其逐候选 SDF
    复检 (_link_metrics) 换成本端口的 numpy 实现 (3.4ms/次 -> ~0.3ms/次)。
    """

    def __init__(self, cfg=None, model_path=None, max_steps=None,
                 max_points=768, use_dls_refine=False, dls_refine_iters=100,
                 infer_max_points=None, policy_backend='numpy',
                 onnx_path=None, adaptive_infer_points=False,
                 infer_min_points=128, infer_near_distance=0.12,
                 infer_full_distance=0.04,
                 skip_inactive_obstacle_slots=True,
                 defer_nearest_direction=True,
                 reuse_sdf_buffers=False,
                 fast_quat_rotate=True,
                 fused_fk=True,
                 batch_point_rotation=True,
                 stop_pos_tolerance=0.010,
                 stop_orient_tolerance_deg=10.0,
                 observation_max_steps=None):
        from .avoid_solvers import AvoidBaselineMJX
        from .env_mjx import RobotIKAvoidEnvMJX

        # 参考 env: 提供几何常量与 specs/场景接口 (构建后仅保留常量)
        ref = RobotIKAvoidEnvMJX(cfg, n_worlds=1, max_steps=max_steps,
                                 max_points=max_points)
        self.cfg = ref.cfg
        # infer_max_points: step 闭环点云抽稀阈值 (None=不抽稀, 与全量
        # 一致); 碰撞判定类路径 (轨迹复检/终态/DLS) 始终全量点云
        self.fenv = FastEnvState(
            ref, infer_max_points=infer_max_points,
            adaptive_infer_points=adaptive_infer_points,
            infer_min_points=infer_min_points,
            infer_near_distance=infer_near_distance,
            infer_full_distance=infer_full_distance,
            skip_inactive_obstacle_slots=skip_inactive_obstacle_slots,
            defer_nearest_direction=defer_nearest_direction,
            reuse_sdf_buffers=reuse_sdf_buffers,
            fast_quat_rotate=fast_quat_rotate,
            fused_fk=fused_fk,
            batch_point_rotation=batch_point_rotation,
            stop_pos_tolerance=stop_pos_tolerance,
            stop_orient_tolerance_deg=stop_orient_tolerance_deg,
            observation_max_steps=observation_max_steps)
        self._ref_env = ref                       # specs 还原/可视化兼容
        self.use_dls_refine = use_dls_refine
        self.dls_refine_iters = dls_refine_iters
        self._baseline = None
        if use_dls_refine:
            self._baseline = AvoidBaselineMJX(self.cfg, max_points=max_points)
            fs = self.fenv
            # 逐候选 SDF 复检: jax _link_metrics -> numpy min_dist_batch
            self._baseline._link_metrics = lambda q: (
                fs.min_dist_batch(np.asarray(q, np.float32)[None])[0],
                np.zeros((len(fs.mon_groups), 3), np.float32))

        if model_path is None:
            raise ValueError('FastRLAvoidSolverMJX 需显式指定 model_path')
        self.policy_backend = str(policy_backend).lower()
        print(f'  [RL-Avoid-Fast] 加载策略: backend={self.policy_backend} '
              f'model={model_path}')
        # numpy 直读 zip; ONNX 后端使用相同 predict 接口, 几何链路不变
        self.policy = load_policy(model_path, backend=self.policy_backend,
                                  onnx_path=onnx_path)

    @property
    def home_position(self):
        return self.fenv.home_position

    @staticmethod
    def _coupled(pd, od):
        cosine_dist = 2.0 - np.cos(od)
        return float(pd * cosine_dist + (cosine_dist - 1.0))

    def solve(self, target_pos, target_quat, obstacle_specs=None,
              n_obstacles=0, dist_range=(0.12, 0.30), clearance=0.05,
              start_q=None, record_trajectory=True):
        """口径与 RLAvoidSolverMJX.solve 一致 (best-so-far + 轨迹复检)"""
        fs = self.fenv
        specs_inflated = False
        if obstacle_specs is None:
            # 随机场景走参考 env 采样后取 specs (与 jax 版同路径);
            # 还原的 specs 已含膨胀量, 后续不得二次膨胀/回写
            self._ref_env.set_scene(
                0, target_pos, target_quat,
                start_q if start_q is not None else fs.home_position,
                n_obstacles=n_obstacles, dist_range=dist_range,
                clearance=clearance)
            obstacle_specs = self._ref_specs()
            start_q = np.asarray(self._ref_env.qpos)[0]
            specs_inflated = True
        obs = fs.set_scene(target_pos, target_quat, start_q, obstacle_specs,
                           specs_inflated=specs_inflated)
        if not specs_inflated:
            self._sync_ref_specs(obstacle_specs)
        specs = list(obstacle_specs)
        has_obs = fs.n_obs > 0

        target_pos = np.asarray(target_pos, dtype=np.float32)
        target_quat = quat_normalize(np.asarray(target_quat, dtype=np.float32))

        q0 = fs.qpos.copy()
        trajectory = [q0.copy()] if record_trajectory else None

        pd0, od0 = fs.prev_pos_dist, fs.prev_orient_dist
        md0 = float(fs.prev_mon_dist.min()) if has_obs else 0.5
        best = {'joints': q0.copy(), 'coupled': self._coupled(pd0, od0),
                'pos': pd0, 'ori': od0, 'min_dist': md0}

        t0 = time.perf_counter()
        steps = 0
        collision = False
        margin = fs.success_margin
        for _ in range(fs.max_steps):
            action, _ = self.policy.predict(obs, deterministic=True)
            obs, md = fs.step(action)
            steps += 1
            q = fs.qpos.copy()
            if record_trajectory:
                trajectory.append(q)
            pd, od = fs.prev_pos_dist, fs.prev_orient_dist
            if has_obs and md <= 0.0:
                # 抽稀距离只能做“确定碰撞”的早停: md_sparse 上偏,
                # <=0 时全量必 <=0; 反之不能用抽稀值判“安全”
                collision = True
                break
            coupled = self._coupled(pd, od)
            if coupled < best['coupled']:
                best.update(joints=q.copy(), coupled=coupled,
                            pos=pd, ori=od, min_dist=md)
            if (pd < fs.pos_tolerance and od < fs.orient_tolerance):
                if not has_obs:
                    # The current frame is the certified solution.  Returning
                    # an older coupled-distance "best" can otherwise discard
                    # a frame that satisfies both independent tolerances.
                    best.update(joints=q.copy(), coupled=coupled,
                                pos=pd, ori=od, min_dist=md)
                    break
                # 安全口径早停: 抽稀 md 只是粗筛, 达标候选帧用全量
                # 点云复核一次 (每步至多一次, 仅在接近成功时触发)
                if md >= margin:
                    _, _, md_full = self._pose_metrics(q)
                    if md_full >= margin:
                        best.update(joints=q.copy(), coupled=coupled,
                                    pos=pd, ori=od, min_dist=md_full)
                        break
        t_ms = (time.perf_counter() - t0) * 1000

        traj_collisions = self._trajectory_collision_count(trajectory)

        # 回退到历史最优构型 (与 jax 版同步 ref env, 供可视化/DLS 场景复用)
        fs.qpos = best['joints'].copy()
        self._ref_env.qpos = self._ref_env.qpos.at[0].set(best['joints'])

        refine_traj = []
        if self.use_dls_refine and traj_collisions == 0:
            ref = self._refine_with_dls(best['joints'], target_pos,
                                        target_quat)
            if ref is not None:
                best['joints'], refine_traj = ref
                fs.qpos = best['joints'].copy()
                self._ref_env.qpos = self._ref_env.qpos.at[0].set(
                    best['joints'])

        pos_err, ori_err, min_dist = self._pose_metrics(best['joints'])
        n_final_contacts = 1 if (has_obs and min_dist <= 0) else 0
        success = (pos_err < fs.pos_tolerance
                   and ori_err < fs.orient_tolerance
                   and n_final_contacts == 0
                   and (not has_obs or min_dist >= margin))

        return {
            'success': bool(success),
            'method': ('rl_dls_hybrid' if self.use_dls_refine
                       else 'rl_ppo_avoid_mjx_fast'),
            'joint_angles': [float(x) for x in best['joints'].tolist()],
            'position_error': float(pos_err),
            'orientation_error': float(ori_err),
            'orientation_error_deg': float(np.degrees(ori_err)),
            'min_distance': float(min_dist),
            'n_collisions': int(n_final_contacts),
            'trajectory_collisions': int(traj_collisions),
            'iterations': steps + len(refine_traj),
            'solve_time_ms': round((time.perf_counter() - t0) * 1000, 2),
            'n_obstacles': int(fs.n_obs),
            'obstacle_specs': specs,
            'target_position': target_pos.tolist(),
            'target_orientation': target_quat.tolist(),
            'trajectory': trajectory,
            'refine_trajectory': refine_traj,
        }

    def _sync_ref_specs(self, specs):
        """障碍物张量同步进参考 env 世界 0 (可视化 get_scene_state 兼容)"""
        env = self._ref_env
        pos, shape, he, r, cyl, n = self.fenv.specs_to_arrays(specs or [])
        env.obs_pos = env.obs_pos.at[0].set(pos)
        env.obs_shape = env.obs_shape.at[0].set(shape)
        env.obs_he = env.obs_he.at[0].set(he)
        env.obs_r = env.obs_r.at[0].set(r)
        env.obs_cyl = env.obs_cyl.at[0].set(cyl)
        env.n_obs = env.n_obs.at[0].set(n)

    def _ref_specs(self):
        """从参考 env 世界 0 还原障碍物 specs (随机场景路径用)"""
        env = self._ref_env
        n = int(np.asarray(env.n_obs)[0])
        pos = np.asarray(env.obs_pos)[0]
        shape = np.asarray(env.obs_shape)[0]
        he = np.asarray(env.obs_he)[0]
        r = np.asarray(env.obs_r)[0]
        cyl = np.asarray(env.obs_cyl)[0]
        specs = []
        names = {0: 'box', 1: 'sphere', 2: 'cylinder'}
        for s in range(n):
            spec = {'shape': names[int(shape[s])],
                    'pos': [float(x) for x in pos[s]]}
            if shape[s] == 0:
                spec['size'] = {'halfExtents': [float(x) for x in he[s]]}
            elif shape[s] == 1:
                spec['size'] = {'radius': float(r[s])}
            else:
                spec['size'] = {'radius': float(cyl[s][0]),
                                'height': float(2 * cyl[s][1])}
            specs.append(spec)
        return specs

    def _pose_metrics(self, q):
        """构型 q 的 (pos_err, orient_err, min_dist) — 终态碰撞判定,
        单次批量 FK+SDF (全量点云口径)"""
        fs = self.fenv
        obs, pd, od, mon_d = fs._obs(np.asarray(q, np.float32)[None],
                                     fs.step_count, fs.prev_min_dist,
                                     full=True)
        md = float(mon_d[0].min()) if fs.n_obs > 0 else 0.5
        return float(pd[0]), float(od[0]), md

    def _trajectory_collision_count(self, trajectory):
        """逐帧碰撞复检 (min_dist<=0 口径, 包络球粗筛加速)"""
        fs = self.fenv
        if not trajectory or fs.n_obs == 0:
            return 0
        return fs.trajectory_collisions(np.asarray(trajectory, np.float32))

    def _refine_with_dls(self, q_start, target_pos, target_quat):
        """RL 粗解 → 碰撞安全 DLS 精调 (语义对齐 RLAvoidSolverMJX)"""
        fs = self.fenv
        bl = self._baseline
        specs = self._ref_specs()
        bl.set_scene(target_pos, target_quat, obstacle_specs=specs,
                     start_q=np.asarray(q_start, dtype=np.float32))
        res = bl.refine_safe_dls(
            start_q=np.asarray(q_start, dtype=np.float32),
            max_iterations=self.dls_refine_iters)
        q_new = np.asarray(res['joint_angles'], dtype=np.float32)
        pd, od, md = self._pose_metrics(q_new)
        coupled_new = self._coupled(pd, od)
        has_obs = fs.n_obs > 0
        ok = (res['trajectory_collisions'] == 0
              and (not has_obs or md >= fs.success_margin))
        if not ok:
            return None
        pd0, od0, _ = self._pose_metrics(q_start)
        if coupled_new >= self._coupled(pd0, od0):
            return None
        return q_new, res['trajectory'][1:]

    def get_scene_state(self):
        """返回当前场景状态 (可视化用, 与 jax 版字段一致)"""
        fs = self.fenv
        fk = fs._fk(fs.qpos[None])
        ee_pos = fk[2][0]
        ee_quat = fk[3][0]
        return {
            'joints': fs.qpos.tolist(),
            'ee_position': ee_pos.tolist(),
            'ee_orientation': ee_quat.tolist(),
            'target_position': fs.target_pos.tolist(),
            'target_orientation': fs.target_quat.tolist(),
            'obstacles': self._ref_specs(),
        }

    def close(self):
        pass
