"""7 轴机械臂无碰撞 IK 强化学习环境 — JAX/MJX 向量化实现

与已移除的 PyBullet 版避障环境语义完全对齐, 用于 GPU 大规模
并行训练。架构:

- 观测/动作/奖励/成功判定: 逐项复刻 RobotIKAvoidEnv
- 正运动学: JAX 实现的串联链 FK (与 MuJoCo/PyBullet 数值对齐, 见 tests)
- link-障碍物距离: 监控 link 用碰撞网格凸包点云, 障碍物 (box/sphere/
  cylinder, 轴对齐) 用闭式 SDF; 方向 = -∇SDF (指向障碍物), 对齐
  PyBullet getClosestPoints 的 dir 约定 (link 最近点 -> 障碍物)
- 场景采样 (reset): CPU MuJoCo 自碰撞过滤 + numpy 走廊采样, 逐项复刻
  ObstacleManager.spawn_scene 的接受/拒绝逻辑

已知近似 (对齐测试中量化):
- 点云采样间距引入的距离误差 (毫米级)
- 自碰撞过滤: MuJoCo 凸包接触 vs PyBullet 网格接触 (同一套网格, 差异极小)

状态张量 (leading dim = n_worlds):
    qpos (N,7), target_pos (N,3), target_quat (N,4 xyzw),
    obs_pos (N,3,3), obs_shape (N,3) int, obs_he (N,3,3), obs_r (N,3),
    obs_cyl (N,3,2), n_obs (N,), step_count (N,),
    prev_pos_dist/prev_orient_dist/prev_coupled (N,)
"""
import os

import numpy as np
import jax
import jax.numpy as jnp
import mujoco

from ik_avoid.config import AvoidConfig, load_config

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXED_XML = os.path.join(_BASE, 'mjx', 'assets', 'panda_fixed.xml')
MESH_DIR = os.path.join(_BASE, 'urdf', 'meshes', 'collision')

# PyBullet AABB 包含 Bullet 内建 margin, 采样包络球时补偿
PB_AABB_PAD = 0.004

HAND_Z = 0.107
HAND_YAW = -np.pi / 4
FINGER_Z = 0.0584

# pybullet link 索引 -> 点云组索引 (link1..7, hand, finger_l, finger_r)
PB_LINK_TO_GROUP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
                    8: 7, 9: 8, 10: 9}
GROUP_NAMES = ['link1', 'link2', 'link3', 'link4', 'link5', 'link6',
               'link7', 'hand', 'finger_l', 'finger_r']
GROUP_MESHES = ['link1.obj', 'link2.obj', 'link3.obj', 'link4.obj',
                'link5.obj', 'link6.obj', 'link7.obj', 'hand.obj',
                'finger.obj', 'finger.obj']
GROUP_FRAMES = ['link'] * 7 + ['hand', 'finger_l', 'finger_r']


# ==================== 四元数工具 (xyzw, 对齐 ik_6d.rl_env_6d) ====================

def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return jnp.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], axis=-1)


def quat_conjugate(q):
    return q * jnp.array([-1.0, -1.0, -1.0, 1.0])


def quat_to_axis_angle(q):
    """xyzw 四元数 -> 轴角向量 (短路径), 支持批量"""
    n = jnp.linalg.norm(q, axis=-1, keepdims=True)
    q = jnp.where(n < 1e-10, jnp.array([0, 0, 0, 1.0]), q / n)
    q = jnp.where(q[..., 3:4] < 0, -q, q)
    w = jnp.clip(q[..., 3], -1.0, 1.0)
    angle = 2.0 * jnp.arccos(jnp.abs(w))
    sin_half = jnp.sqrt(jnp.maximum(1.0 - w * w, 1e-16))
    scale = jnp.where(sin_half < 1e-8, 2.0, angle / sin_half)
    return q[..., :3] * scale[..., None]


def quat_angle(q1, q2):
    return jnp.linalg.norm(quat_to_axis_angle(
        quat_multiply(q2, quat_conjugate(q1))), axis=-1)


# ==================== 点云准备 ====================

def _load_group_points(max_points=768):
    """各监控组的本地坐标系点云 (凸包网格顶点+面中心, 等距抽样封顶)"""
    import trimesh
    clouds = []
    for name, mesh_file in zip(GROUP_NAMES, GROUP_MESHES):
        mesh = trimesh.load(os.path.join(MESH_DIR, mesh_file), force='mesh')
        pts = np.concatenate([mesh.vertices,
                              mesh.triangles.mean(axis=1)], axis=0)
        if name == 'finger_r':
            # 右手指 = 左手指网格绕 z 旋转 180°
            pts = pts @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        if len(pts) > max_points:
            idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
            pts = pts[idx]
        clouds.append(pts.astype(np.float32))
    return clouds


# ==================== 环境 ====================

class RobotIKAvoidEnvMJX:
    """PyBullet RobotIKAvoidEnv 的 JAX 向量化等价实现"""

    def __init__(self, cfg: AvoidConfig = None, n_worlds=1, config_path=None,
                 max_steps=None, stage=None, max_points=768):
        self.cfg = cfg or (load_config(config_path) if config_path else AvoidConfig())
        ec = self.cfg.env
        self.n_worlds = int(n_worlds)
        self.max_steps = max_steps or ec.max_steps
        self.pos_tolerance = ec.pos_tolerance
        self.orient_tolerance = self.cfg.orient_tolerance_rad
        self.home_position = np.array(
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)

        stages = self.cfg.obstacle.stages
        self._stage = len(stages) - 1 if stage is None else stage

        # ---------- 模型常量 ----------
        self._mj_model = mujoco.MjModel.from_xml_path(FIXED_XML)
        self.n_ctrl = 7
        # 关节限位 (MJCF range 与 URDF 一致)
        jlo, jhi = [], []
        for i in range(self.n_ctrl):
            lo = float(self._mj_model.jnt_range[i][0])
            hi = float(self._mj_model.jnt_range[i][1])
            jlo.append(lo); jhi.append(hi)
        self.joint_lo = jnp.array(jlo, dtype=jnp.float32)
        self.joint_hi = jnp.array(jhi, dtype=jnp.float32)

        # FK 链: 各 body 相对父体的固定位姿 (pos, quat wxyz)
        self._chain = []
        for bname in [f'panda_link{i}' for i in range(1, 8)]:
            bid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, bname)
            self._chain.append((
                jnp.array(self._mj_model.body_pos[bid], dtype=jnp.float32),
                jnp.array(self._mj_model.body_quat[bid], dtype=jnp.float32),
            ))
        c45, s45 = np.cos(HAND_YAW / 2), np.sin(HAND_YAW / 2)
        self._hand_off = (jnp.array([0, 0, HAND_Z], dtype=jnp.float32),
                          jnp.array([c45, 0, 0, s45], dtype=jnp.float32))
        self._finger_off = jnp.array([0, 0, FINGER_Z], dtype=jnp.float32)
        self._qz180 = jnp.array([0, 0, 0, 1.0], dtype=jnp.float32)  # wxyz

        # 监控组
        self.monitored_links = list(ec.monitored_links)
        self.mon_groups = [PB_LINK_TO_GROUP[i] for i in self.monitored_links]
        clouds = _load_group_points(max_points)
        self._pts = [jnp.array(c, dtype=jnp.float32) for c in clouds]
        # 包络球在 link 局部系的球心/半径 (点云刚体连接, 不随 q 变化;
        # 等价于世界系 AABB 外接球的保守上界, 仅供 reset 采样保护/自碰撞粗估)
        sph_c, sph_r = [], []
        for g in self.mon_groups:
            p = np.asarray(clouds[g])
            mn = p.min(axis=0) - PB_AABB_PAD
            mx = p.max(axis=0) + PB_AABB_PAD
            sph_c.append(((mn + mx) / 2).astype(np.float32))
            sph_r.append(float(np.linalg.norm(mx - mn)) / 2 * 0.75)
        self._sph_local_c = jnp.stack(sph_c)
        self._sph_local_r = jnp.asarray(sph_r, dtype=jnp.float32)
        self._jit_spheres = jax.jit(self._spheres_centers_jax)

        # 观测维度 (基础: n_ctrl + 8 + 4L + 2; obs_obstacle_hint +4;
        # obs_safe_joint_dir +7; obs_lookahead +2)
        self.n_link_obs = len(self.mon_groups)
        self.obs_dim = self.n_ctrl + 8 + 4 * self.n_link_obs + 2 \
            + (4 if ec.obs_obstacle_hint else 0) \
            + (self.n_ctrl if ec.obs_safe_joint_dir else 0) \
            + (2 if ec.obs_lookahead else 0)

        # v13 逐关节刹车: 关节 k 驱动 pb link >= k 的监控连杆组 (静态掩码 (7,L))
        pb_links = [sorted(PB_LINK_TO_GROUP, key=PB_LINK_TO_GROUP.get)[g]
                    for g in self.mon_groups]
        self._distal_mask = jnp.asarray(
            [[pl >= k for pl in pb_links] for k in range(self.n_ctrl)],
            dtype=jnp.bool_)

        # ---------- 设备状态 ----------
        N = self.n_worlds
        self.qpos = jnp.zeros((N, 7), dtype=jnp.float32)
        self.target_pos = jnp.zeros((N, 3), dtype=jnp.float32)
        self.target_quat = jnp.tile(jnp.array([0, 0, 0, 1.0]), (N, 1))
        self.obs_pos = jnp.zeros((N, 3, 3), dtype=jnp.float32).at[:, :, 2].set(-1000.0)
        self.obs_shape = jnp.zeros((N, 3), dtype=jnp.int32)
        self.obs_he = jnp.zeros((N, 3, 3), dtype=jnp.float32)
        self.obs_r = jnp.zeros((N, 3), dtype=jnp.float32)
        self.obs_cyl = jnp.zeros((N, 3, 2), dtype=jnp.float32)
        self.n_obs = jnp.zeros((N,), dtype=jnp.int32)
        self.step_count = jnp.zeros((N,), dtype=jnp.int32)
        self.prev_pos_dist = jnp.zeros((N,), dtype=jnp.float32)
        self.prev_orient_dist = jnp.zeros((N,), dtype=jnp.float32)
        self.prev_coupled = jnp.zeros((N,), dtype=jnp.float32)
        self.prev_min_dist = jnp.full((N,), 0.5, dtype=jnp.float32)
        # v13: 上一步各监控连杆组最近距离 (逐关节刹车用)
        self.prev_mon_dist = jnp.full((N, self.n_link_obs), 0.5,
                                      dtype=jnp.float32)
        self.prev_proj = jnp.zeros((N,), dtype=jnp.float32)

        # CPU MuJoCo 采样用数据
        self._mj_data = mujoco.MjData(self._mj_model)
        self._ee_site = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, 'ee')
        self.rng = np.random.default_rng()

        self._jit_fk = jax.jit(self._fk_batch)
        self._jit_step = jax.jit(self._step_batch)
        self._jit_obs = jax.jit(self._obs_batch)

        # 全量步进时免 scatter 写回快路径 (step_batch); 置 False 可回退
        # 旧路径做 A/B 对比 (等价性见 logs/SPEED_OPT_LOG.md 任务 #7)
        self._fast_writeback = True

    # ==================== 正运动学 (JAX) ====================

    @staticmethod
    def _q_mul_w(q1, q2):
        """wxyz 四元数乘法"""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        return jnp.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], axis=-1)

    @staticmethod
    def _q_rot_w(q, v):
        """wxyz 四元数旋转向量 (批量 q: (...,4), v: (...,3))"""
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        t = 2.0 * jnp.cross(q[..., 1:4], v)
        return v + w[..., None] * t + jnp.cross(q[..., 1:4], t)

    def _fk_batch(self, q):
        """FK: q (N,7) -> (link_pos (N,7,3), link_quat_w (N,7,4),
                          hand_pos (N,3), hand_quat_xyzw (N,4),
                          fingL_pos, fingL_quat_xyzw, fingR_pos, fingR_quat_xyzw,
                          joint_axes (N,7,3), joint_pos (N,7,3))
        返回的四元数: link 为 wxyz (内部用), hand/finger 转 xyzw (对外);
        joint_axes/joint_pos 为各关节在世界系的轴方向与铰点,
        供解析几何 Jacobian (关节空间安全方向观测) 使用"""
        N = q.shape[0]
        pos = jnp.zeros((N, 3), dtype=jnp.float32)
        quat = jnp.tile(jnp.array([1, 0, 0, 0.0]), (N, 1))
        link_pos, link_quat = [], []
        j_pos, j_axis = [], []
        z_unit = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
        for i in range(7):
            p_off, q_off = self._chain[i]
            pos = pos + self._q_rot_w(quat, jnp.broadcast_to(p_off, (N, 3)))
            quat = self._q_mul_w(quat, jnp.broadcast_to(q_off, (N, 4)))
            # 关节 i 铰点与轴 (固定偏移后、关节旋转前的坐标系)
            j_pos.append(pos)
            j_axis.append(self._q_rot_w(quat, jnp.broadcast_to(z_unit, (N, 3))))
            ang = q[:, i]
            qj = jnp.stack([jnp.cos(ang / 2), jnp.zeros(N),
                            jnp.zeros(N), jnp.sin(ang / 2)], axis=-1)
            quat = self._q_mul_w(quat, qj)
            link_pos.append(pos)
            link_quat.append(quat)
        # hand = link7 ∘ (z+0.107, rot_z(-45°))
        p_off, q_off = self._hand_off
        hand_pos = pos + self._q_rot_w(quat, jnp.broadcast_to(p_off, (N, 3)))
        hand_quat_w = self._q_mul_w(quat, jnp.broadcast_to(q_off, (N, 4)))
        # fingers = hand ∘ z+0.0584 [右手指再绕 z 转 180°]
        fl_pos = hand_pos + self._q_rot_w(hand_quat_w, jnp.broadcast_to(
            self._finger_off, (N, 3)))
        fr_pos = fl_pos
        fl_quat_w = hand_quat_w
        fr_quat_w = self._q_mul_w(hand_quat_w, jnp.broadcast_to(self._qz180, (N, 4)))

        def to_xyzw(qw):
            return jnp.stack([qw[..., 1], qw[..., 2], qw[..., 3], qw[..., 0]], axis=-1)

        return (jnp.stack(link_pos, axis=1),
                jnp.stack(link_quat, axis=1),
                hand_pos, to_xyzw(hand_quat_w),
                fl_pos, to_xyzw(fl_quat_w),
                fr_pos, to_xyzw(fr_quat_w),
                jnp.stack(j_axis, axis=1), jnp.stack(j_pos, axis=1))

    def _group_frames(self, fk):
        """各组 (pos, quat_wxyz), 批量维 (N,...)"""
        link_pos, link_quat = fk[0], fk[1]

        def to_wxyz(qx):
            return jnp.stack([qx[..., 3], qx[..., 0], qx[..., 1], qx[..., 2]], axis=-1)

        frames = [(link_pos[:, g], link_quat[:, g]) for g in range(7)]
        frames.append((fk[2], to_wxyz(fk[3])))   # hand
        frames.append((fk[4], to_wxyz(fk[5])))   # finger_l
        frames.append((fk[6], to_wxyz(fk[7])))   # finger_r
        return frames

    # ==================== 距离计算 (点云 SDF) ====================

    def _sdf_and_dir(self, pw, obs_pos, obs_shape, obs_he, obs_r, obs_cyl):
        """世界点 pw (N,P,3) 到 3 个障碍物插槽的带符号距离与指向障碍物方向

        输入均为世界批量维: obs_* (N,3,...)
        Returns: dist (N,P,3), dir (N,P,3,3)
        """
        def one_slot(p, c, shape, he, r, cyl):
            # --- sphere ---
            ds = jnp.linalg.norm(p - c, axis=-1) - r
            u = (c - p) / jnp.maximum(jnp.linalg.norm(c - p, axis=-1,
                                                      keepdims=True), 1e-9)
            # --- box (轴对齐) ---
            qc = jnp.abs(p - c) - he
            db = jnp.linalg.norm(jnp.maximum(qc, 0), axis=-1) \
                + jnp.minimum(jnp.max(qc, axis=-1), 0)
            # box 方向: 外部=最近面投影, 内部=最深穿透轴反向
            s_box = jnp.clip(p, c - he, c + he)
            n_sb = jnp.linalg.norm(s_box - p, axis=-1, keepdims=True)
            outside = n_sb > 1e-9
            k = jnp.argmax(qc, axis=-1)
            e_k = jax.nn.one_hot(k, 3)
            dir_in = -jnp.sign(p - c) * e_k
            dir_box = jnp.where(outside, (s_box - p) / jnp.maximum(n_sb, 1e-9),
                                dir_in)
            # --- cylinder (z 轴) ---
            rr, hh = cyl[..., 0], cyl[..., 1]
            dxy = p[..., :2] - c[..., :2]
            rho = jnp.linalg.norm(dxy, axis=-1)
            q1 = rho - rr
            q2 = jnp.abs(p[..., 2] - c[..., 2]) - hh
            dc = jnp.linalg.norm(jnp.maximum(jnp.stack([q1, q2], axis=-1), 0),
                                 axis=-1) \
                + jnp.minimum(jnp.maximum(q1, q2), 0)
            dir_xy = jnp.where(rho[..., None] > 1e-9,
                               -dxy / jnp.maximum(rho[..., None], 1e-9),
                               jnp.zeros(2))
            # 外部方向: 侧面分量(权 q1+) + 上下底面分量(权 q2+) 合成后归一
            dir_cyl_raw = jnp.concatenate([
                dir_xy * jnp.maximum(q1, 0)[..., None],
                (-jnp.sign(p[..., 2] - c[..., 2])
                 * jnp.maximum(q2, 0))[..., None]], axis=-1)
            dir_cyl_out = dir_cyl_raw / jnp.maximum(
                jnp.linalg.norm(dir_cyl_raw, axis=-1, keepdims=True), 1e-9)
            inside_c = (q1 < 0) & (q2 < 0)
            dir_cyl_in = jnp.where((q1 > q2)[..., None],
                                   dir_cyl_out,
                                   jnp.stack([jnp.zeros_like(q1),
                                              jnp.zeros_like(q1),
                                              -jnp.sign(p[..., 2] - c[..., 2])], axis=-1))
            dir_cyl = jnp.where(inside_c[..., None], dir_cyl_in, dir_cyl_out)

            dist = jnp.where(shape == 0, db, jnp.where(shape == 1, ds, dc))
            dvec = jnp.where(shape[..., None] == 0, dir_box,
                             jnp.where(shape[..., None] == 1, u, dir_cyl))
            return dist, dvec

        dists, dirs = [], []
        for s in range(3):
            # 插槽参数扩维到 (N,1,...) 与 pw (N,P,3) 广播
            d, dv = one_slot(pw,
                             obs_pos[:, s][:, None, :],
                             obs_shape[:, s][:, None],
                             obs_he[:, s][:, None, :],
                             obs_r[:, s][:, None],
                             obs_cyl[:, s][:, None, :])
            dists.append(d)
            dirs.append(dv)
        return jnp.stack(dists, axis=-1), jnp.stack(dirs, axis=-2)

    def _link_dists(self, fk, obs_pos, obs_shape, obs_he, obs_r, obs_cyl):
        """批量: 各世界各监控点云组到最近障碍物的 (dist, dir, 最近点世界坐标)

        Returns: dist (N,L), dir (N,L,3), point (N,L,3)
        """
        frames = self._group_frames(fk)
        N = obs_pos.shape[0]
        idx = jnp.arange(N)
        out_d, out_v, out_p = [], [], []
        for g in self.mon_groups:
            gpos, gquat = frames[g]                # (N,3) (N,4)
            pts = self._pts[g]                     # (P,3)
            pw = self._q_rot_w(gquat[:, None, :], pts[None, ...]) \
                + gpos[:, None, :]                 # (N,P,3)
            d_all, dv_all = self._sdf_and_dir(pw, obs_pos, obs_shape, obs_he,
                                              obs_r, obs_cyl)  # (N,P,3) (N,P,3,3)
            P3 = d_all.shape[1] * d_all.shape[2]
            flat = d_all.reshape(N, P3)
            i_min = jnp.argmin(flat, axis=-1)
            out_d.append(flat[idx, i_min])
            out_v.append(dv_all.reshape(N, P3, 3)[idx, i_min])
            # 最近点世界坐标 (P 点 × 3 插槽展平, 点索引 = i_min // 3)
            out_p.append(pw[idx, i_min // 3])
        return (jnp.stack(out_d, axis=-1), jnp.stack(out_v, axis=-2),
                jnp.stack(out_p, axis=-2))

    # ==================== 观测 ====================

    @staticmethod
    def _coupled_distance(pos_dist, orient_dist):
        cosine_dist = 2.0 - jnp.cos(orient_dist)
        return pos_dist * cosine_dist + (cosine_dist - 1.0)

    @staticmethod
    def _proj_of(pos_diff, target_pos):
        """末端在 基座→目标 轴上的投影坐标 (绕行中性, 仅沿轴进度计入)

        pos_diff = target - hand, hand = target - pos_diff
        """
        base = jnp.array([0.0, 0.0, 0.33], dtype=jnp.float32)
        axis = target_pos - base
        axis = axis / jnp.maximum(
            jnp.linalg.norm(axis, axis=-1, keepdims=True), 1e-9)
        hand = target_pos - pos_diff
        return jnp.sum((hand - base) * axis, axis=-1)

    def _obs_batch(self, qpos, target_pos, target_quat, step_count, n_obs,
                   obs_pos, obs_shape, obs_he, obs_r, obs_cyl, prev_md):
        """计算观测 (N, obs_dim) 与距离中间量"""
        ec = self.cfg.env
        fk = self._fk_batch(qpos)
        _, _, hand_pos, hand_quat = fk[0], fk[1], fk[2], fk[3]

        pos_diff = target_pos - hand_pos
        pos_dist = jnp.linalg.norm(pos_diff, axis=-1)
        pos_dir = jnp.where(pos_dist[..., None] > 1e-8,
                            pos_diff / jnp.maximum(pos_dist[..., None], 1e-9),
                            jnp.zeros(3))
        orient_aa = quat_to_axis_angle(quat_multiply(target_quat,
                                                     quat_conjugate(hand_quat)))
        orient_dist = jnp.linalg.norm(orient_aa, axis=-1)
        cosine_dist = 2.0 - jnp.cos(orient_dist)

        # 批量 link 距离 (无嵌套 vmap, 编译图更小)
        mon_d, mon_v, mon_p = self._link_dists(fk, obs_pos, obs_shape,
                                               obs_he, obs_r, obs_cyl)
        # mon_d: (N,L) mon_v: (N,L,3) mon_p: (N,L,3)
        has_obs = (n_obs > 0).astype(jnp.float32)
        d_norm = jnp.clip(mon_d, -0.05, ec.dist_clip) / ec.dist_clip
        link_feat = jnp.concatenate([
            d_norm[..., None], mon_v * has_obs[:, None, None]], axis=-1)
        link_feat = link_feat.reshape(-1, 4 * self.n_link_obs)

        parts = [
            qpos,                                   # n
            pos_dir,                                # 3
            pos_dist[..., None],                    # 1
            orient_aa,                              # 3
            cosine_dist[..., None],                 # 1
            link_feat,                              # 4L
            (n_obs / 3.0)[..., None],               # 1
            (step_count / self.max_steps)[..., None],  # 1
        ]
        if ec.obs_obstacle_hint or ec.obs_safe_joint_dir:
            # 全局最近监控点 (link, 点, 障碍插槽) 索引
            N = mon_d.shape[0]
            l_min = jnp.argmin(mon_d, axis=-1)
            idx = jnp.arange(N)
        if ec.obs_obstacle_hint:
            # 最近障碍提示: 全局最近监控点处的 SDF 方向(指向障碍) + 归一化距离
            hint_dir = mon_v[idx, l_min] * has_obs[:, None]        # (N,3)
            hint_d = mon_d[idx, l_min]
            hint_dn = jnp.where(
                n_obs > 0,
                jnp.clip(hint_d, -0.05, ec.dist_clip) / ec.dist_clip,
                1.0)
            parts.extend([hint_dir, hint_dn[..., None]])           # +4
        if ec.obs_safe_joint_dir:
            # 关节空间安全方向: 全局最近点处 min_dist 对 q 的梯度方向 (7 维)
            # ∇_q d = J_v^T · ∂d/∂p, ∂d/∂p = -u (u 指向障碍),
            # J_v 列 = z_i × (p - p_i) (旋转关节几何 Jacobian)
            joint_axes, joint_pos = fk[8], fk[9]                   # (N,7,3)
            u = mon_v[idx, l_min] * has_obs[:, None]               # (N,3)
            p_near = mon_p[idx, l_min]                             # (N,3)
            jv = jnp.cross(joint_axes,
                           p_near[:, None, :] - joint_pos)         # (N,7,3)
            grad_q = -jnp.einsum('njk,nk->nj', jv, u)              # (N,7)
            safe_dir = grad_q / jnp.maximum(
                jnp.linalg.norm(grad_q, axis=-1, keepdims=True), 1e-9)
            parts.append(safe_dir * has_obs[:, None])              # +7
        if ec.obs_lookahead:
            # v11 前瞻 2 维: 碰撞失败全是"单帧穿透", 瞬时距离快照缺刹车依据,
            # 补时间导数信号: 上一步接近速度 + 照当前趋势的下一步预测余量
            # (reset 帧 step_count=0 无历史, 掩码归零)
            cur_md = mon_d.min(axis=-1)
            md_delta = jnp.where(step_count > 0, cur_md - prev_md,
                                 0.0) * has_obs
            vel = jnp.clip(md_delta, -0.05, 0.05) / 0.05 * has_obs
            pred = jnp.clip(cur_md + md_delta,
                            -0.05, ec.dist_clip) / ec.dist_clip
            pred = jnp.where(n_obs > 0, pred, 1.0)
            parts.extend([vel[..., None], pred[..., None]])        # +2
        obs = jnp.concatenate(parts, axis=-1)
        return obs, pos_dist, orient_dist, mon_d, pos_diff

    # ==================== step ====================

    def _step_batch(self, qpos, target_pos, target_quat, step_count, n_obs,
                    obs_pos, obs_shape, obs_he, obs_r, obs_cyl,
                    prev_pos_dist, prev_orient_dist, prev_coupled,
                    prev_min_dist, prev_mon_dist, prev_proj, action):
        rc = self.cfg.reward
        ec = self.cfg.env
        N = qpos.shape[0]
        has_obs = n_obs > 0

        action = jnp.clip(action, -1.0, 1.0)
        scale_input = jnp.maximum(prev_pos_dist,
                                  ec.orient_scale_weight * prev_orient_dist)
        scale = ec.min_action_scale + ec.max_action_span \
            * jnp.minimum(scale_input / 0.5, 1.0)
        if ec.brake_margin > 0 and ec.brake_mode == 'per_joint':
            # v13 逐关节刹车: 关节 k 只看它驱动的远端连杆组 (pb link >= k) 的
            # 最近障碍距离, 线性压步 (下限 floor)——margin 取 0.49*远端力臂上界
            # 以上时, 最近点单步位移 ≤ 剩余距离, 穿透在物理上被堵;
            # 非危险关节保持全速, 降低全局刹车的超时代价
            d_pad = jnp.where(self._distal_mask[None],
                              prev_mon_dist[:, None, :], 1e9)
            d_distal = d_pad.min(axis=-1)                        # (N,7)
            brake_j = jnp.where(
                has_obs[:, None],
                jnp.clip(d_distal / ec.brake_margin,
                         ec.brake_floor, 1.0),
                1.0)
            new_q = jnp.clip(qpos + action * scale[:, None] * brake_j,
                             self.joint_lo, self.joint_hi)
        else:
            if ec.brake_margin > 0:
                # 接近障碍时线性压低步长 (下限 brake_floor), 给策略反应时间
                brake = jnp.where(
                    has_obs,
                    jnp.clip(prev_min_dist / ec.brake_margin,
                             ec.brake_floor, 1.0),
                    1.0)
                scale = scale * brake
            new_q = jnp.clip(qpos + action * scale[:, None],
                             self.joint_lo, self.joint_hi)

        obs, pos_dist, orient_dist, mon_d, pos_diff = self._obs_batch(
            new_q, target_pos, target_quat, step_count + 1, n_obs,
            obs_pos, obs_shape, obs_he, obs_r, obs_cyl, prev_min_dist)
        coupled = self._coupled_distance(pos_dist, orient_dist)
        min_dist = mon_d.min(axis=-1)
        proj = self._proj_of(pos_diff, target_pos)

        # ---- 奖励 ----
        if rc.task_diff:
            # v8 势能差 (potential-based shaping): 回报沿轨迹望远镜相消,
            # episode 总任务回报只取决于终态 coupled, 长度中性,
            # 消除绝对势下"早撞止损"的套利空间
            reward = rc.task_scale * (prev_coupled - coupled)
        else:
            reward = rc.task_base - rc.task_scale * coupled
        if ec.progress_mode == 'projection':
            # 目标轴投影进度: 横向绕行不被惩罚, 消除与避障的行为冲突
            prog = rc.w_progress * (proj - prev_proj)
        else:
            prog = rc.w_progress * (prev_coupled - coupled)
        if rc.gate_progress:
            # v7 危险区进度豁免: 贴障时进度项只奖不罚, 移除绕行税
            near = has_obs & (prev_min_dist < ec.safe_margin)
            prog = jnp.where(near, jnp.maximum(prog, 0.0), prog)
        reward = reward + prog
        reward = reward + rc.w_orient_progress * (prev_orient_dist - orient_dist)
        if rc.w_rep > 0:
            # 方向性斥力: prev_min_dist 进入 rep_margin 内, 按逼近速度(min_dist 下降量)惩罚
            approach = jnp.clip(min_dist - prev_min_dist, -0.05, 0.0)
            rep_on = has_obs & (prev_min_dist < ec.rep_margin)
            reward = reward + jnp.where(rep_on, rc.w_rep * approach, 0.0)
        safe = jnp.where(has_obs,
                         rc.w_safe * jnp.clip(min_dist, 0.0, ec.safe_margin)
                         / ec.safe_margin, 0.0)
        reward = reward + safe
        if rc.w_align > 0 and ec.obs_safe_joint_dir:
            # v7 安全方向对齐奖励: 危险区内沿"间隙增大方向"(safe_dir, obs 末 7 维)
            # 行动直接发奖, 与任务吸引同量级对抗, 给避障行为正向动机
            align_on = has_obs & (prev_min_dist < ec.align_margin)
            safe_dir = obs[..., -7:]
            align = jnp.maximum((action * safe_dir).sum(axis=-1), 0.0)
            reward = reward + jnp.where(align_on, rc.w_align * align, 0.0)
        reward = reward - rc.w_smooth * (action ** 2).sum(axis=-1)
        at_limit = ((new_q <= self.joint_lo + 1e-6)
                    | (new_q >= self.joint_hi - 1e-6)).sum(axis=-1)
        reward = reward - rc.w_joint_limit * at_limit
        reward = reward - rc.step_penalty

        collision = has_obs & (min_dist <= 0.0)
        success = (pos_dist < self.pos_tolerance) \
            & (orient_dist < self.orient_tolerance) \
            & ((~has_obs) | (min_dist >= ec.success_margin))

        reward = reward + jnp.where(collision, -rc.w_collision, 0.0)
        reward = reward + jnp.where((~collision) & success, rc.success_reward, 0.0)

        terminated = collision | ((~collision) & success)
        truncated = (step_count + 1 >= self.max_steps) & ~terminated

        return (obs, reward, terminated, truncated, new_q,
                pos_dist, orient_dist, coupled, min_dist, proj, mon_d,
                step_count + 1)

    # ==================== 场景采样 (CPU, 复刻 PyBullet 版语义) ====================

    def _mj_ee_pose(self, q):
        """CPU MuJoCo FK -> (pos, quat_xyzw)"""
        d = self._mj_data
        d.qpos[:7] = np.asarray(q)
        mujoco.mj_forward(self._mj_model, d)
        pos = np.array(d.site_xpos[self._ee_site], dtype=np.float32)
        R = d.site_xmat[self._ee_site].reshape(3, 3)
        return pos, self._mat_to_quat_xyzw(R)

    @staticmethod
    def _mat_to_quat_xyzw(R):
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        q = np.array([x, y, z, w], dtype=np.float32)
        return q / (np.linalg.norm(q) + 1e-12)

    def _mj_self_collision(self, q):
        d = self._mj_data
        d.qpos[:7] = np.asarray(q)
        mujoco.mj_forward(self._mj_model, d)
        return d.ncon > 0

    def _mj_link_spheres(self, q):
        """构型 q 下监控组包络球 (对齐 PyBullet _link_spheres:
        AABB 外接球 × 0.75 收缩; PyBullet AABB 含 4mm margin 故补偿)"""
        fk_np = self._fk_numpy(np.asarray(q, dtype=np.float32))
        spheres = []
        for g in self.mon_groups:
            gpos, gquat_w = fk_np[g]
            pts = np.asarray(self._pts[g])
            R = self._quat_w_to_mat(gquat_w)
            pw = pts @ R.T + gpos
            mn = pw.min(axis=0) - PB_AABB_PAD
            mx = pw.max(axis=0) + PB_AABB_PAD
            spheres.append(((mn + mx) / 2,
                            float(np.linalg.norm(mx - mn)) / 2 * 0.75))
        return spheres

    @staticmethod
    def _quat_w_to_mat(q):
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])

    def _fk_numpy(self, q):
        """与 _fk_batch 同语义的 numpy 版 (采样阶段用), 返回各组 (pos, quat_wxyz)"""
        q = np.asarray(q)
        pos = np.zeros(3, dtype=np.float64)
        quat = np.array([1, 0, 0, 0], dtype=np.float64)
        frames = []

        def qmul(a, b):
            w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2])

        def qrot(qq, v):
            w, x, y, z = qq
            t = 2.0 * np.cross(qq[1:4], v)
            return v + w * t + np.cross(qq[1:4], t)

        for i in range(7):
            p_off = np.asarray(self._chain[i][0], dtype=np.float64)
            q_off = np.asarray(self._chain[i][1], dtype=np.float64)
            pos = pos + qrot(quat, p_off)
            quat = qmul(quat, q_off)
            ang = q[i]
            quat = qmul(quat, np.array([np.cos(ang/2), 0, 0, np.sin(ang/2)]))
            frames.append((pos.copy(), quat.copy()))
        p_off = np.array([0, 0, HAND_Z], dtype=np.float64)
        c, s = np.cos(HAND_YAW/2), np.sin(HAND_YAW/2)
        q_off = np.array([c, 0, 0, s], dtype=np.float64)
        hand_pos = pos + qrot(quat, p_off)
        hand_quat = qmul(quat, q_off)
        frames.append((hand_pos.copy(), hand_quat.copy()))
        f_off = np.array([0, 0, FINGER_Z], dtype=np.float64)
        fl_pos = hand_pos + qrot(hand_quat, f_off)
        frames.append((fl_pos.copy(), hand_quat.copy()))
        frames.append((fl_pos.copy(), qmul(hand_quat, np.array([0, 0, 0, 1.0]))))
        return frames

    def _sample_target_config(self):
        """随机关节角 -> FK -> 可达 6D 目标 (自碰撞过滤), 对齐 PyBullet 版"""
        margin = self.cfg.env.target_margin
        lo = np.asarray(self.joint_lo); hi = np.asarray(self.joint_hi)
        for _ in range(30):
            m = (hi - lo) * margin
            q = self.rng.uniform(lo + m, hi - m).astype(np.float32)
            if self._mj_self_collision(q):
                continue
            pos, quat = self._mj_ee_pose(q)
            noise = self.rng.uniform(-0.005, 0.005, size=3).astype(np.float32)
            return pos + noise, quat, q
        pos, quat = self._mj_ee_pose(q)
        return pos, quat, q

    # ---------- 障碍物采样 (复刻 obstacles.ObstacleManager) ----------

    def _random_spec(self):
        cfg = self.cfg.obstacle
        shape = self.rng.choice(cfg.shapes)
        if shape == 'box':
            lo, hi = cfg.box_half_extents
            size = {'halfExtents': [float(self.rng.uniform(lo, hi)) for _ in range(3)]}
        elif shape == 'sphere':
            size = {'radius': float(self.rng.uniform(*cfg.sphere_radius))}
        else:
            size = {'radius': float(self.rng.uniform(*cfg.cylinder_radius)),
                    'height': float(self.rng.uniform(*cfg.cylinder_height))}
        return {'shape': shape, 'size': size}

    @staticmethod
    def _bound_radius(shape, size):
        if shape == 'box':
            return float(np.linalg.norm(size['halfExtents']))
        if shape == 'sphere':
            return float(size['radius'])
        return float(np.sqrt(size['radius']**2 + (size['height']/2)**2))

    def _spawn_scene_specs(self, n, target_pos, protected_spheres,
                           dist_range, clearance, base_pos=(0.0, 0.0, 0.33)):
        """复刻 ObstacleManager.spawn_scene 的走廊采样, 返回 specs 列表"""
        cfg = self.cfg.obstacle
        target_pos = np.asarray(target_pos, dtype=float)
        base_pos = np.asarray(base_pos, dtype=float)
        seg = target_pos - base_pos
        seg_dir = seg / (np.linalg.norm(seg) + 1e-9)

        specs = []
        bounds = []
        positions = []
        for _ in range(n):
            for _ in range(cfg.max_attempts):
                spec = self._random_spec()
                bound = self._bound_radius(spec['shape'], spec['size'])
                t = float(self.rng.uniform(0.25, 0.85))
                axis_point = base_pos + t * seg
                v = self.rng.normal(size=3)
                v = v - np.dot(v, seg_dir) * seg_dir
                v /= (np.linalg.norm(v) + 1e-9)
                dd = float(self.rng.uniform(dist_range[0], dist_range[1]))
                c = axis_point + v * dd
                if c[2] < cfg.region_z[0]:
                    continue
                if np.linalg.norm(c - target_pos) < cfg.target_clearance + bound:
                    continue
                ok = True
                for (pp, rr) in protected_spheres:
                    if np.linalg.norm(c - np.asarray(pp)) < rr + bound + clearance:
                        ok = False
                        break
                if not ok:
                    continue
                for (bpos, bbound) in zip(positions, bounds):
                    if np.linalg.norm(c - bpos) < bound + bbound + 0.02:
                        ok = False
                        break
                if not ok:
                    continue
                spec['pos'] = [float(c[0]), float(c[1]), float(c[2])]
                specs.append(spec)
                bounds.append(bound)
                positions.append(c)
                break
        return specs

    def _specs_to_arrays(self, specs):
        """specs -> 单世界障碍物张量 (3 插槽)"""
        pos = np.zeros((3, 3), dtype=np.float32)
        pos[:, 2] = -1000.0
        shape = np.zeros(3, dtype=np.int32)
        he = np.zeros((3, 3), dtype=np.float32)
        r = np.zeros(3, dtype=np.float32)
        cyl = np.zeros((3, 2), dtype=np.float32)
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
        # v12 训练期障碍膨胀: SDF/碰撞判定/观测 hint/safe_dir 全部基于膨胀后几何,
        # 把 -w_collision 重罚前移 margin, 治"单帧穿透"(碰撞信号晚一帧的信用分配问题);
        # 评测配置必须 margin=0 用真实几何
        m = self.cfg.env.train_inflate_margin
        if m > 0:
            he = he + m          # box 半边长各向 +m
            r = r + m            # 球半径 +m
            cyl = cyl + m        # 柱: 半径与半高均 +m
        return pos, shape, he, r, cyl, len(specs[:3])

    # ==================== Gymnasium 风格接口 ====================

    def _setup_world(self, w, target_pos, target_quat, start_q, specs):
        self.qpos = self.qpos.at[w].set(np.clip(start_q, self.joint_lo, self.joint_hi))
        self.target_pos = self.target_pos.at[w].set(target_pos)
        qn = np.asarray(target_quat, dtype=np.float32)
        qn = qn / (np.linalg.norm(qn) + 1e-12)
        self.target_quat = self.target_quat.at[w].set(qn)
        pos, shape, he, r, cyl, n = self._specs_to_arrays(specs)
        self.obs_pos = self.obs_pos.at[w].set(pos)
        self.obs_shape = self.obs_shape.at[w].set(shape)
        self.obs_he = self.obs_he.at[w].set(he)
        self.obs_r = self.obs_r.at[w].set(r)
        self.obs_cyl = self.obs_cyl.at[w].set(cyl)
        self.n_obs = self.n_obs.at[w].set(n)
        self.step_count = self.step_count.at[w].set(0)

        # 单世界切片计算初始观测与 prev 量 (避免全批量重算)
        obs, pos_dist, orient_dist, mon_d, pos_diff = self._jit_obs(
            self.qpos[w:w+1], self.target_pos[w:w+1], self.target_quat[w:w+1],
            self.step_count[w:w+1], self.n_obs[w:w+1],
            self.obs_pos[w:w+1], self.obs_shape[w:w+1], self.obs_he[w:w+1],
            self.obs_r[w:w+1], self.obs_cyl[w:w+1],
            self.prev_min_dist[w:w+1])
        self.prev_pos_dist = self.prev_pos_dist.at[w].set(pos_dist[0])
        self.prev_orient_dist = self.prev_orient_dist.at[w].set(orient_dist[0])
        self.prev_coupled = self.prev_coupled.at[w].set(
            self._coupled_distance(pos_dist[0], orient_dist[0]))
        self.prev_min_dist = self.prev_min_dist.at[w].set(mon_d[0].min())
        self.prev_mon_dist = self.prev_mon_dist.at[w].set(mon_d[0])
        self.prev_proj = self.prev_proj.at[w].set(
            self._proj_of(pos_diff, self.target_pos[w:w+1])[0])
        return np.asarray(obs[0])

    def set_scene(self, w, target_pos, target_quat, start_q,
                  obstacle_specs=None, n_obstacles=0,
                  dist_range=(0.15, 0.35), clearance=0.06):
        """外部指定场景 (评测/对齐测试用)"""
        if obstacle_specs is None:
            protected = self._mj_link_spheres(np.asarray(start_q))
            obstacle_specs = self._spawn_scene_specs(
                n_obstacles, target_pos, protected, dist_range, clearance)
        return self._setup_world(w, target_pos, target_quat, start_q, obstacle_specs)

    def _sample_scene(self):
        """随机场景采样 (复刻 RobotIKAvoidEnv.reset 采样语义)

        返回 (target_pos, target_quat, start_q, obstacle_specs)
        """
        target_pos, target_quat, q_seed = self._sample_target_config()
        ec = self.cfg.env
        lo = np.asarray(self.joint_lo); hi = np.asarray(self.joint_hi)
        if self.rng.random() < ec.start_random_prob:
            start_q = self.rng.uniform(lo, hi).astype(np.float32)
        else:
            start_q = (self.home_position
                       + self.rng.uniform(-0.8, 0.8, size=7)).astype(np.float32)

        stage = self.cfg.obstacle.stages[self._stage]
        specs = []
        if stage.n_max > 0:
            n = int(self.rng.integers(stage.n_min, stage.n_max + 1))
            if n > 0:
                protected = (self._mj_link_spheres(q_seed)
                             + self._mj_link_spheres(start_q))
                specs = self._spawn_scene_specs(
                    n, target_pos, protected,
                    (stage.dist_min, stage.dist_max), stage.clearance)
        return target_pos, target_quat, start_q, specs

    def reset(self, w=0, seed=None):
        """随机场景 (训练用), 复刻 RobotIKAvoidEnv.reset 语义"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        target_pos, target_quat, start_q, specs = self._sample_scene()
        return self._setup_world(w, target_pos, target_quat, start_q, specs)

    def generate_scenes(self, n, seed, with_obstacles=True):
        """原生评测场景生成 (无 PyBullet 依赖)

        与 PyBullet 版同构: 场景 i 用 rng(seed+i), 目标 FK 采样保证可达,
        障碍物走廊采样保护种子构型与起始构型 (保证存在无碰撞解)。
        """
        stage = len(self.cfg.obstacle.stages) - 1 if with_obstacles else 0
        self.set_obstacle_stage(stage)
        scenes = []
        for i in range(n):
            self.rng = np.random.default_rng(seed + i)
            target_pos, target_quat, start_q, specs = self._sample_scene()
            scenes.append({
                'target_pos': [float(x) for x in np.asarray(target_pos)],
                'target_quat': [float(x) for x in np.asarray(target_quat)],
                'start_q': [float(x) for x in np.asarray(start_q)],
                'obstacle_specs': specs,
            })
        return scenes

    # ==================== 批量 reset (训练吞吐 优化, GPU 采样) ====================

    def set_obstacle_stage(self, stage):
        """课程阶段切换 (下次 reset 生效), 对齐 PyBullet 版接口"""
        stages = self.cfg.obstacle.stages
        self._stage = int(np.clip(stage, 0, len(stages) - 1))

    def _batch_link_spheres(self, q_batch):
        """q_batch (m,7) -> 包络球 center (m,L,3), radius (m,L)

        局部系预计算球心/半径, 仅变换球心 (免点云旋转)
        Returns 附加: hand_pos (m,3), hand_quat_wxyz (m,4)"""
        q_batch = np.asarray(q_batch)
        centers, hpos, hquat = self._jit_spheres(jnp.asarray(q_batch))
        return (np.asarray(centers),
                np.tile(np.asarray(self._sph_local_r)[None, :],
                        (q_batch.shape[0], 1)),
                np.asarray(hpos), np.asarray(hquat))

    def _spheres_centers_jax(self, q):
        """纯 JAX: q (m,7) -> centers (m,L,3), hand_pos (m,3),
        hand_quat_wxyz (m,4)"""
        fk = self._fk_batch(q)
        frames = self._group_frames(fk)
        centers = []
        for gi, g in enumerate(self.mon_groups):
            gpos, gquat = frames[g]
            c_local = self._sph_local_c[gi]
            centers.append(self._q_rot_w(
                gquat, jnp.broadcast_to(c_local, gpos.shape)) + gpos)
        return jnp.stack(centers, axis=1), frames[7][0], frames[7][1]

    def _sample_targets_batch(self, n):
        """批量版 _sample_target_config (定形 jit, 自碰撞用包络球近似)

        每轮全量 n 世界重采 2 候选, 未中者下轮再试, 最多 3 轮
        Returns: target_pos (n,3), target_quat_xyzw (n,4), q_seed (n,7)
        """
        lo = np.asarray(self.joint_lo); hi = np.asarray(self.joint_hi)
        margin = self.cfg.env.target_margin
        m = (hi - lo) * margin
        L = len(self.mon_groups)
        crad = np.asarray(self._sph_local_r)          # (L,)
        tgt_pos = np.zeros((n, 3), dtype=np.float32)
        tgt_quat = np.zeros((n, 4), dtype=np.float32)
        tgt_quat[:, 3] = 1.0
        q_seed = np.zeros((n, 7), dtype=np.float32)
        unfilled = np.ones(n, dtype=bool)
        last = None
        ar = np.arange(L)
        for _ in range(3):
            if not unfilled.any():
                break
            cand = self.rng.uniform(lo + m, hi - m,
                                    size=(n, 2, 7)).astype(np.float32)
            flat = cand.reshape(n * 2, 7)
            centers, hpos, hquat = self._jit_spheres(flat)
            cpos = np.asarray(centers)                # (2n,L,3)
            hpos = np.asarray(hpos); hquat = np.asarray(hquat)
            diff = cpos[:, :, None, :] - cpos[:, None, :, :]
            dd = np.linalg.norm(diff, axis=-1)
            dd[:, ar, ar] = 1e9
            ok = ((dd - (crad[None, :, None] + crad[None, None, :])).min(
                axis=(1, 2)) > 0.005)
            ok2 = ok.reshape(n, 2)
            fi = np.arange(n) * 2 + np.where(ok2[:, 0], 0, 1)
            last = (hpos[fi], hquat[fi], flat[fi])
            new = ok2.any(axis=1) & unfilled
            if new.any():
                noise = self.rng.uniform(-0.005, 0.005,
                                         size=(int(new.sum()), 3))
                tq = hquat[fi][new]                   # wxyz
                tgt_pos[new] = (hpos[fi][new] + noise).astype(np.float32)
                tgt_quat[new] = np.stack([tq[:, 1], tq[:, 2],
                                          tq[:, 3], tq[:, 0]],
                                         axis=-1).astype(np.float32)
                q_seed[new] = flat[fi][new]
                unfilled[new] = False
        if unfilled.any() and last is not None:
            noise = self.rng.uniform(-0.005, 0.005,
                                     size=(int(unfilled.sum()), 3))
            tq = last[1][unfilled]
            tgt_pos[unfilled] = (last[0][unfilled] + noise).astype(np.float32)
            tgt_quat[unfilled] = np.stack(
                [tq[:, 1], tq[:, 2], tq[:, 3], tq[:, 0]],
                axis=-1).astype(np.float32)
            q_seed[unfilled] = last[2][unfilled]
        return tgt_pos, tgt_quat, q_seed

    def _spawn_specs_batch(self, n, n_w, target_pos, prot_c, prot_r,
                           dist_range, clearance):
        """批量版走廊采样 (复刻 _spawn_scene_specs)

        n_w (n,) 各世界障碍数; prot_c (n,P,3) prot_r (n,P) 保护球
        Returns: 3 插槽张量 (n,3,...) + n_obs (n,)
        """
        cfg = self.cfg.obstacle
        base = np.array([0.0, 0.0, 0.33], dtype=np.float32)
        pool = list(cfg.shapes)
        code_of = {name: i for i, name in
                   enumerate(['box', 'sphere', 'cylinder'])}
        # 每世界每插槽形状与尺寸 (先全量采样, 保持与单版随机流结构相近)
        shp = self.rng.choice(np.array(pool), size=(n, 3))
        he_s = self.rng.uniform(*cfg.box_half_extents,
                                size=(n, 3, 3)).astype(np.float32)
        sr_s = self.rng.uniform(*cfg.sphere_radius,
                                size=(n, 3)).astype(np.float32)
        cr_s = self.rng.uniform(*cfg.cylinder_radius,
                                size=(n, 3)).astype(np.float32)
        ch_s = self.rng.uniform(*cfg.cylinder_height,
                                size=(n, 3)).astype(np.float32)
        shape_code = np.zeros((n, 3), dtype=np.int32)
        for name, code in code_of.items():
            shape_code[shp == name] = code
        bound = np.where(shape_code == 0,
                         np.linalg.norm(he_s, axis=-1),
                         np.where(shape_code == 1, sr_s,
                                  np.sqrt(cr_s**2 + (ch_s / 2)**2)))
        pos_out = np.zeros((n, 3, 3), dtype=np.float32)
        pos_out[:, :, 2] = -1000.0
        shape_out = np.zeros((n, 3), dtype=np.int32)
        he_out = np.zeros((n, 3, 3), dtype=np.float32)
        r_out = np.zeros((n, 3), dtype=np.float32)
        cyl_out = np.zeros((n, 3, 2), dtype=np.float32)
        b_out = np.zeros((n, 3), dtype=np.float32)
        n_obs = np.zeros(n, dtype=np.int32)
        for slot in range(3):
            idx = np.where(n_w > slot)[0]
            if len(idx) == 0:
                break
            na = len(idx)
            K = cfg.max_attempts
            tp = target_pos[idx]                      # (na,3) float32
            seg = tp - base
            seg_dir = seg / (np.linalg.norm(seg, axis=-1, keepdims=True)
                             + 1e-9)
            t = self.rng.uniform(0.25, 0.85, size=(na, K)).astype(np.float32)
            axis_pt = base + t[..., None] * seg[:, None, :]
            v = self.rng.normal(size=(na, K, 3)).astype(np.float32)
            v -= (v * seg_dir[:, None, :]).sum(-1, keepdims=True) \
                * seg_dir[:, None, :]
            v /= (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
            dlat = self.rng.uniform(dist_range[0], dist_range[1],
                                    size=(na, K)).astype(np.float32)
            c = axis_pt + v * dlat[..., None]         # (na,K,3)
            b = bound[idx, slot][:, None]             # (na,1)
            valid = c[..., 2] >= cfg.region_z[0]
            valid &= (np.linalg.norm(c - tp[:, None, :], axis=-1)
                      >= cfg.target_clearance + b)
            dc = np.linalg.norm(c[:, :, None, :] - prot_c[idx][:, None, :, :],
                                axis=-1)              # (na,K,P)
            valid &= (dc >= prot_r[idx][:, None, :]
                      + bound[idx, slot][:, None, None] + clearance).all(-1)
            for prev in range(slot):
                dprev = np.linalg.norm(c - pos_out[idx, prev][:, None, :],
                                       axis=-1)
                valid &= dprev >= b + b_out[idx, prev][:, None] + 0.02
            has = valid.any(axis=1)
            if not has.any():
                continue
            take = valid.argmax(axis=1)
            c_sel = c[np.arange(na), take]
            ii = idx[has]
            pos_out[ii, slot] = c_sel[has]
            b_out[ii, slot] = bound[idx[has], slot]
            shape_out[ii, slot] = shape_code[ii, slot]
            he_out[ii, slot] = he_s[ii, slot]
            r_out[ii, slot] = sr_s[ii, slot]
            cyl_out[ii, slot] = np.stack(
                [cr_s[ii, slot], ch_s[ii, slot] / 2], axis=-1)
            n_obs[ii] += 1
        return pos_out, shape_out, he_out, r_out, cyl_out, n_obs

    def _setup_worlds_batch(self, w_idx, start_q, target_pos, target_quat,
                            obs_pos, obs_shape, obs_he, obs_r, obs_cyl,
                            n_obs):
        """批量写入状态并计算初始观测, 返回 obs (len, obs_dim)"""
        w_idx = np.asarray(w_idx, dtype=np.int32)
        self.qpos = self.qpos.at[w_idx].set(
            np.clip(start_q, self.joint_lo, self.joint_hi))
        self.target_pos = self.target_pos.at[w_idx].set(target_pos)
        qn = target_quat / (np.linalg.norm(target_quat, axis=-1,
                                           keepdims=True) + 1e-12)
        self.target_quat = self.target_quat.at[w_idx].set(qn)
        self.obs_pos = self.obs_pos.at[w_idx].set(obs_pos)
        self.obs_shape = self.obs_shape.at[w_idx].set(obs_shape)
        self.obs_he = self.obs_he.at[w_idx].set(obs_he)
        self.obs_r = self.obs_r.at[w_idx].set(obs_r)
        self.obs_cyl = self.obs_cyl.at[w_idx].set(obs_cyl)
        self.n_obs = self.n_obs.at[w_idx].set(n_obs)
        self.step_count = self.step_count.at[w_idx].set(0)
        obs, pd, od, md, pdiff = self._jit_obs(
            self.qpos[w_idx], self.target_pos[w_idx],
            self.target_quat[w_idx], self.step_count[w_idx],
            self.n_obs[w_idx], self.obs_pos[w_idx], self.obs_shape[w_idx],
            self.obs_he[w_idx], self.obs_r[w_idx], self.obs_cyl[w_idx],
            self.prev_min_dist[w_idx])
        self.prev_pos_dist = self.prev_pos_dist.at[w_idx].set(pd)
        self.prev_orient_dist = self.prev_orient_dist.at[w_idx].set(od)
        self.prev_coupled = self.prev_coupled.at[w_idx].set(
            self._coupled_distance(pd, od))
        self.prev_min_dist = self.prev_min_dist.at[w_idx].set(
            md.min(axis=-1))
        self.prev_mon_dist = self.prev_mon_dist.at[w_idx].set(md)
        self.prev_proj = self.prev_proj.at[w_idx].set(
            self._proj_of(pdiff, self.target_pos[w_idx]))
        return np.asarray(obs)

    def reset_batch(self, w_idx):
        """批量随机重置指定世界 (训练用), 返回 obs (len(w_idx), obs_dim)

        内部 padding 到分桶定形 (N/8~N), 避免 jit 重编译且小批量省采样"""
        w_idx = np.asarray(w_idx, dtype=np.int32)
        n_real = len(w_idx)
        N = self.n_worlds
        if N <= 128:
            n = N
        else:
            n = next(b for b in (max(N // 8, 128), N // 4, N // 2, N)
                     if n_real <= b)
        if n_real < n:
            w_idx = np.concatenate([
                w_idx, np.full(n - n_real, w_idx[0], np.int32)])
        tgt_pos, tgt_quat, q_seed = self._sample_targets_batch(n)
        ec = self.cfg.env
        lo = np.asarray(self.joint_lo); hi = np.asarray(self.joint_hi)
        start_q = np.empty((n, 7), dtype=np.float32)
        full = self.rng.random(n) < ec.start_random_prob
        if full.any():
            start_q[full] = self.rng.uniform(lo, hi, size=(int(full.sum()), 7))
        if (~full).any():
            start_q[~full] = (self.home_position
                              + self.rng.uniform(-0.8, 0.8,
                                                 size=(int((~full).sum()), 7)))
        start_q = start_q.astype(np.float32)
        stage = self.cfg.obstacle.stages[self._stage]
        if stage.n_max == 0:
            obs_pos = np.zeros((n, 3, 3), dtype=np.float32)
            obs_pos[:, :, 2] = -1000.0
            obs_shape = np.zeros((n, 3), dtype=np.int32)
            obs_he = np.zeros((n, 3, 3), dtype=np.float32)
            obs_r = np.zeros((n, 3), dtype=np.float32)
            obs_cyl = np.zeros((n, 3, 2), dtype=np.float32)
            n_obs = np.zeros(n, dtype=np.int32)
        else:
            n_w = self.rng.integers(stage.n_min, stage.n_max + 1, size=n)
            both = np.concatenate([q_seed, start_q], axis=0)
            pc, pr, _, _ = self._batch_link_spheres(both)
            prot_c = np.concatenate(
                [pc[:n], pc[n:]], axis=1)             # (n,2L,3)
            prot_r = np.concatenate(
                [pr[:n], pr[n:]], axis=1)             # (n,2L)
            obs_pos, obs_shape, obs_he, obs_r, obs_cyl, n_obs = \
                self._spawn_specs_batch(n, n_w, tgt_pos, prot_c, prot_r,
                                        (stage.dist_min, stage.dist_max),
                                        stage.clearance)
        obs = self._setup_worlds_batch(w_idx, start_q, tgt_pos, tgt_quat,
                                       obs_pos, obs_shape, obs_he, obs_r,
                                       obs_cyl, n_obs)
        return obs[:n_real]

    def step(self, action, w=0, return_min_dist=False):
        """单世界接口 (兼容 PyBullet 版调用方式)

        return_min_dist=True 时额外返回 min_dist (float), 口径与
        get_min_dist 一致 (_jit_step 内部 mon_d.min, 新 qpos), 可免除
        solve 循环每步冗余的 get_min_dist(_jit_obs) 重算。
        """
        out = self.step_batch(
            np.asarray(action, dtype=np.float32)[None], worlds=[w],
            return_min_dist=return_min_dist)
        if return_min_dist:
            obs, reward, term, trunc, min_dist = out
            return (obs[0], float(reward[0]), bool(term[0]), bool(trunc[0]),
                    {}, float(min_dist[0]))
        obs, reward, term, trunc = out
        return obs[0], float(reward[0]), bool(term[0]), bool(trunc[0]), {}

    def step_batch(self, actions, worlds=None, return_min_dist=False):
        """批量 step: actions (K,7) 作用于 worlds 列表 (默认全部)

        Returns: obs (K,D), reward (K,), terminated (K,), truncated (K,)
                 [, min_dist (K,) 当 return_min_dist=True]

        快路径 (_fast_writeback=True 且全量步进): 免 [idx] gather 切片与
        8 次 .at[idx].set() scatter 写回, 直接全数组重绑定 —— 实测单世界
        口径省 ~2.5ms/步 (未入 jit 的 scatter 单次 ~0.2ms)。子集 worlds
        步进自动回退旧路径。
        """
        actions = jnp.asarray(actions, dtype=jnp.float32)
        full = (worlds is None) or (self.n_worlds == 1)

        if self._fast_writeback and full:
            # n_worlds==1 或 worlds=None 时 actions (N,7) 即全量
            (obs, reward, term, trunc, new_q,
             pos_dist, orient_dist, coupled, min_dist, proj, mon_d,
             new_cnt) = self._jit_step(
                self.qpos, self.target_pos, self.target_quat,
                self.step_count, self.n_obs,
                self.obs_pos, self.obs_shape, self.obs_he,
                self.obs_r, self.obs_cyl,
                self.prev_pos_dist, self.prev_orient_dist,
                self.prev_coupled, self.prev_min_dist,
                self.prev_mon_dist, self.prev_proj, actions)
            # 全量重绑定: 零 scatter/零额外 JAX dispatch
            self.qpos = new_q
            self.step_count = new_cnt
            self.prev_pos_dist = pos_dist
            self.prev_orient_dist = orient_dist
            self.prev_coupled = coupled
            self.prev_min_dist = min_dist
            self.prev_mon_dist = mon_d
            self.prev_proj = proj
            return (np.asarray(obs), np.asarray(reward),
                    np.asarray(term), np.asarray(trunc)) + (
                    (np.asarray(min_dist),) if return_min_dist else ())

        # ---- 旧路径: 子集 worlds 步进 (scatter 写回) ----
        if worlds is None:
            idx = jnp.arange(self.n_worlds)
        else:
            idx = jnp.asarray(worlds, dtype=jnp.int32)

        (obs, reward, term, trunc, new_q,
         pos_dist, orient_dist, coupled, min_dist, proj, mon_d,
         _new_cnt) = self._jit_step(
            self.qpos[idx], self.target_pos[idx], self.target_quat[idx],
            self.step_count[idx], self.n_obs[idx],
            self.obs_pos[idx], self.obs_shape[idx], self.obs_he[idx],
            self.obs_r[idx], self.obs_cyl[idx],
            self.prev_pos_dist[idx], self.prev_orient_dist[idx],
            self.prev_coupled[idx], self.prev_min_dist[idx],
            self.prev_mon_dist[idx],
            self.prev_proj[idx], actions)

        self.qpos = self.qpos.at[idx].set(new_q)
        self.step_count = self.step_count.at[idx].add(1)
        self.prev_pos_dist = self.prev_pos_dist.at[idx].set(pos_dist)
        self.prev_orient_dist = self.prev_orient_dist.at[idx].set(orient_dist)
        self.prev_coupled = self.prev_coupled.at[idx].set(coupled)
        self.prev_min_dist = self.prev_min_dist.at[idx].set(min_dist)
        self.prev_mon_dist = self.prev_mon_dist.at[idx].set(mon_d)
        self.prev_proj = self.prev_proj.at[idx].set(proj)
        return (np.asarray(obs), np.asarray(reward),
                np.asarray(term), np.asarray(trunc)) + (
                (np.asarray(min_dist),) if return_min_dist else ())

    def get_obs(self, w=0):
        obs, _, _, _, _ = self._jit_obs(
            self.qpos, self.target_pos, self.target_quat, self.step_count,
            self.n_obs, self.obs_pos, self.obs_shape, self.obs_he,
            self.obs_r, self.obs_cyl, self.prev_min_dist)
        return np.asarray(obs[w])

    def get_obs_batch(self):
        """全部世界观测 (N, obs_dim)"""
        obs, _, _, _, _ = self._jit_obs(
            self.qpos, self.target_pos, self.target_quat, self.step_count,
            self.n_obs, self.obs_pos, self.obs_shape, self.obs_he,
            self.obs_r, self.obs_cyl, self.prev_min_dist)
        return np.asarray(obs)

    def get_min_dist_batch(self):
        """全部世界最小障碍距离 (N,)"""
        _, _, _, mon_d, _ = self._jit_obs(
            self.qpos, self.target_pos, self.target_quat, self.step_count,
            self.n_obs, self.obs_pos, self.obs_shape, self.obs_he,
            self.obs_r, self.obs_cyl, self.prev_min_dist)
        return np.asarray(mon_d.min(axis=-1))

    def get_min_dist(self, w=0):
        _, _, _, mon_d, _ = self._jit_obs(
            self.qpos, self.target_pos, self.target_quat, self.step_count,
            self.n_obs, self.obs_pos, self.obs_shape, self.obs_he,
            self.obs_r, self.obs_cyl, self.prev_min_dist)
        return float(mon_d[w].min())
