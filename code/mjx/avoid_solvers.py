"""MJX 版避障求解器 (替代 ik_avoid.rl_solver_avoid / baselines, 无 PyBullet 依赖)

- RLAvoidSolverMJX:   PPO 推理, 基于 RobotIKAvoidEnvMJX (单世界) 复刻
                      RLAvoidSolver.solve 的 best-so-far 口径; 碰撞判定用
                      点云 SDF min_dist<=0 (等价 PyBullet getContactPoints,
                      毫米级近似)
- AvoidBaselineMJX:   传统 IK baseline (DLS 6D / 零空间斥力避障), 雅可比来自
                      mujoco.mj_jac, 障碍距离/方向复用 MJX env 的 SDF 原语
"""
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import mujoco

from ik_avoid.config import load_config
from ik_6d.quat_utils import (
    quat_multiply, quat_conjugate, quat_normalize,
    quat_to_axis_angle, quat_angle,
)
from .env_mjx import RobotIKAvoidEnvMJX, FIXED_XML
from .mj_solver import mat_to_quat_xyzw


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==================== RL 避障求解器 ====================

class RLAvoidSolverMJX:
    """无碰撞 6D 位姿 IK 的 PPO 推理求解器 (MJX 单世界环境)"""

    def __init__(self, cfg=None, model_path=None, max_steps=None, max_points=768,
                 use_dls_refine=False, dls_refine_iters=100):
        """
        Args:
            cfg: AvoidConfig (None=默认)
            model_path: PPO 模型路径 (不含 .zip), None=自动查找 mjx best/最终模型
            max_steps: 最大迭代步数
            use_dls_refine: v13+ 混合模式 —— RL 粗调后接碰撞安全 DLS 精调
        """
        self.cfg = cfg or self._default_paired_config()
        self.env = RobotIKAvoidEnvMJX(self.cfg, n_worlds=1,
                                      max_steps=max_steps, max_points=max_points)
        self.use_dls_refine = use_dls_refine
        self.dls_refine_iters = dls_refine_iters
        self._baseline = (AvoidBaselineMJX(self.cfg, max_points=max_points)
                          if use_dls_refine else None)

        if model_path is None:
            model_name = self.cfg.train.model_name
            cands = [
                os.path.join(_BASE_DIR, 'models', 'panda_ppo_ik_avoid_mjx_best', 'best_model'),
                os.path.join(_BASE_DIR, 'models', 'panda_ppo_ik_avoid_mjx'),
                os.path.join(_BASE_DIR, 'models', f'{model_name}_best', 'best_model'),
                os.path.join(_BASE_DIR, 'models', model_name),
            ]
            for cand in cands:
                if os.path.exists(cand + '.zip'):
                    model_path = cand
                    break
            else:
                raise FileNotFoundError(
                    f'未找到模型: {[c + ".zip" for c in cands]}, 请先训练')
        print(f'  [RL-Avoid-MJX] 加载模型: {model_path}.zip')
        # 直读 zip 内 policy.pth (ik_avoid/policy_io.py, 零 SB3 依赖;
        # predict 签名兼容 SB3, rollout/评测链路无需感知)
        from ik_avoid.policy_io import load_policy
        self.model = load_policy(model_path)

    @staticmethod
    def _default_paired_config():
        """默认配置: 与候选模型首位 (v3 best, 49 维) 配套的 avoid_remote30m.yaml;
        直接 load_config() 会落到 avoid.yaml (57 维含手指) 导致 shape 不匹配"""
        cand = os.path.join(_BASE_DIR, 'configs', 'avoid_remote30m.yaml')
        return load_config(cand if os.path.exists(cand) else None)

    @property
    def home_position(self):
        return self.env.home_position

    def _coupled(self, pd, od):
        # 与 env._coupled_distance 同式的 numpy 标量版: 求解循环内避免
        # 标量再入 GPU 的往返开销 (实测单次 ~4ms)
        cosine_dist = 2.0 - np.cos(od)
        return float(pd * cosine_dist + (cosine_dist - 1.0))

    def solve(self, target_pos, target_quat, obstacle_specs=None,
              n_obstacles=0, dist_range=(0.12, 0.30), clearance=0.05,
              start_q=None, record_trajectory=True):
        """求解无碰撞 6D IK (口径与 ik_avoid.rl_solver_avoid.RLAvoidSolver 一致)"""
        env = self.env
        start = start_q if start_q is not None else env.home_position
        obs = env.set_scene(0, target_pos, target_quat, start,
                            obstacle_specs=obstacle_specs,
                            n_obstacles=n_obstacles,
                            dist_range=dist_range, clearance=clearance)
        specs = self._current_specs()
        n_obs = int(np.asarray(env.n_obs)[0])
        has_obs = n_obs > 0

        target_pos = np.asarray(target_pos, dtype=np.float32)
        target_quat = quat_normalize(np.asarray(target_quat, dtype=np.float32))

        q0 = np.asarray(env.qpos)[0].copy()
        trajectory = [q0.copy()] if record_trajectory else None

        # best-so-far: 按耦合距离记录无碰撞的历史最优构型
        # (md0 直接读 set_scene 缓存的 prev_mon_dist, 免一次 _jit_obs 重算)
        pd0 = float(np.asarray(env.prev_pos_dist)[0])
        od0 = float(np.asarray(env.prev_orient_dist)[0])
        md0 = (float(np.min(np.asarray(env.prev_mon_dist)[0]))
               if has_obs else 0.5)
        best = {'joints': q0.copy(), 'coupled': self._coupled(pd0, od0),
                'pos': pd0, 'ori': od0, 'min_dist': md0}

        t0 = time.perf_counter()
        steps = 0
        collision = False
        margin = self.cfg.env.success_margin
        for _ in range(env.max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            # min_dist 随 step 一次算出 (step_batch 复用同一次 _jit_obs),
            # 不再事后 get_min_dist 重算 FK+SDF
            obs_b, _, _, _, md_b = env.step_batch(
                np.asarray(action, dtype=np.float32)[None], worlds=[0],
                return_min_dist=True)
            obs = obs_b[0]
            steps += 1
            q = np.asarray(env.qpos)[0]
            if record_trajectory:
                trajectory.append(q.copy())
            pd = float(np.asarray(env.prev_pos_dist)[0])
            od = float(np.asarray(env.prev_orient_dist)[0])
            md = float(md_b[0]) if has_obs else 0.5
            if has_obs and md <= 0.0:
                collision = True
                break
            coupled = self._coupled(pd, od)
            if coupled < best['coupled']:
                best.update(joints=q.copy(), coupled=coupled,
                            pos=pd, ori=od, min_dist=md)
            if (pd < env.pos_tolerance and od < env.orient_tolerance
                    and (not has_obs or md >= margin)):
                break

        t_ms = (time.perf_counter() - t0) * 1000

        # 轨迹碰撞逐帧复检 (min_dist<=0 口径)
        traj_collisions = self._trajectory_collision_count(trajectory)

        # 回退到历史最优构型 (抗末期振荡)
        env.qpos = env.qpos.at[0].set(best['joints'])

        # v13+ 混合模式: RL 粗调结果接碰撞安全 DLS 精调 (仅当 RL 轨迹无碰撞,
        # 否则精调段会把轨迹碰撞带进终态口径之外的指标, 得不偿失)
        refine_traj = []
        if (self.use_dls_refine and traj_collisions == 0):
            ref = self._refine_with_dls(best['joints'], target_pos, target_quat)
            if ref is not None:
                best['joints'], refine_traj = ref
                env.qpos = env.qpos.at[0].set(best['joints'])

        pos_err, ori_err, min_dist = self._pose_metrics(best['joints'])
        n_final_contacts = 1 if (has_obs and min_dist <= 0) else 0
        success = (pos_err < env.pos_tolerance
                   and ori_err < env.orient_tolerance
                   and n_final_contacts == 0
                   and (not has_obs or min_dist >= margin))

        return {
            'success': bool(success),
            'method': ('rl_dls_hybrid' if self.use_dls_refine
                       else 'rl_ppo_avoid_mjx'),
            'joint_angles': [float(x) for x in best['joints'].tolist()],
            'position_error': float(pos_err),
            'orientation_error': float(ori_err),
            'orientation_error_deg': float(np.degrees(ori_err)),
            'min_distance': float(min_dist),
            'n_collisions': int(n_final_contacts),
            'trajectory_collisions': int(traj_collisions),
            'iterations': steps + len(refine_traj),
            'solve_time_ms': round(t_ms, 2),
            'n_obstacles': n_obs,
            'obstacle_specs': specs,
            'target_position': target_pos.tolist(),
            'target_orientation': target_quat.tolist(),
            'trajectory': trajectory,
            'refine_trajectory': refine_traj,
        }

    def _pose_metrics(self, q):
        """构型 q 的 (pos_err, orient_err, min_dist), 基于 env 世界 0 场景"""
        env = self.env
        q_save = np.asarray(env.qpos)[0].copy()
        env.qpos = env.qpos.at[0].set(np.asarray(q, dtype=np.float32))
        obs, pd, od, md, _ = env._jit_obs(
            env.qpos[0:1], env.target_pos[0:1], env.target_quat[0:1],
            env.step_count[0:1], env.n_obs[0:1], env.obs_pos[0:1],
            env.obs_shape[0:1], env.obs_he[0:1], env.obs_r[0:1],
            env.obs_cyl[0:1], env.prev_min_dist[0:1])
        env.qpos = env.qpos.at[0].set(q_save)
        has_obs = int(np.asarray(env.n_obs)[0]) > 0
        return (float(np.asarray(pd)[0]), float(np.asarray(od)[0]),
                float(np.asarray(md)[0].min()) if has_obs else 0.5)

    def _refine_with_dls(self, q_start, target_pos, target_quat):
        """RL 粗解 → 碰撞安全 DLS 精调; 精调不改善或引入碰撞则返回 None

        障碍场景从 env 世界 0 同步到 baseline 自己的 env (set_scene 后
        逐帧 SDF 复检: 穿入碰撞的步被拒绝, best-so-far 保底不退化)"""
        env = self.env
        bl = self._baseline
        specs = self._current_specs()
        bl.set_scene(target_pos, target_quat, obstacle_specs=specs,
                     start_q=np.asarray(q_start, dtype=np.float32))
        res = bl.refine_safe_dls(
            start_q=np.asarray(q_start, dtype=np.float32),
            max_iterations=self.dls_refine_iters)
        q_new = np.asarray(res['joint_angles'], dtype=np.float32)
        # 终态指标基于 env 场景复检 (与 RL 口径同源)
        pd, od, md = self._pose_metrics(q_new)
        coupled_new = self._coupled(pd, od)
        has_obs = int(np.asarray(env.n_obs)[0]) > 0
        ok = (res['trajectory_collisions'] == 0
              and (not has_obs or md >= self.cfg.env.success_margin))
        if not ok:
            return None
        pd0, od0, _ = self._pose_metrics(q_start)
        if coupled_new >= self._coupled(pd0, od0):
            return None                      # 精调无改善, 保留 RL 粗解
        return q_new, res['trajectory'][1:]

    def _current_specs(self):
        """从 env 世界 0 的张量还原障碍物 specs (前端渲染用)"""
        env = self.env
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

    def _min_dist_at(self, q):
        """构型 q 下的最小障碍距离 (临时写入世界 0)"""
        env = self.env
        q_save = np.asarray(env.qpos)[0].copy()
        env.qpos = env.qpos.at[0].set(np.asarray(q, dtype=np.float32))
        md = env.get_min_dist(0)
        env.qpos = env.qpos.at[0].set(q_save)
        return md

    def _trajectory_collision_count(self, trajectory):
        """逐帧检查轨迹碰撞 (min_dist<=0 口径, 与终态一致)

        全部帧堆叠成批一次 _jit_obs 复检 (相对逐帧 _min_dist_at,
        ~30 次 jit 分发并为 1 次); 定长 padding 到 max_steps+1 避免
        不同轨迹长度触发 jit 重编译"""
        env = self.env
        n_traj = len(trajectory)
        if n_traj == 0 or int(np.asarray(env.n_obs)[0]) == 0:
            return 0
        T = env.max_steps + 1
        qs = np.asarray(trajectory, dtype=np.float32)
        if qs.shape[0] < T:
            qs = np.concatenate([qs, np.tile(qs[-1:], (T - qs.shape[0], 1))])
        _, _, _, mon_d, _ = env._jit_obs(
            qs, np.tile(np.asarray(env.target_pos)[0:1], (T, 1)),
            np.tile(np.asarray(env.target_quat)[0:1], (T, 1)),
            np.tile(np.asarray(env.step_count)[0:1], (T,)),
            np.tile(np.asarray(env.n_obs)[0:1], (T,)),
            np.tile(np.asarray(env.obs_pos)[0:1], (T, 1, 1)),
            np.tile(np.asarray(env.obs_shape)[0:1], (T, 1)),
            np.tile(np.asarray(env.obs_he)[0:1], (T, 1, 1)),
            np.tile(np.asarray(env.obs_r)[0:1], (T, 1)),
            np.tile(np.asarray(env.obs_cyl)[0:1], (T, 1, 1)),
            np.tile(np.asarray(env.prev_min_dist)[0:1], (T,)))
        return int((np.asarray(mon_d).min(axis=-1)[:n_traj] <= 0.0).sum())

    def get_scene_state(self):
        """返回当前场景状态 (可视化用)"""
        env = self.env
        q = np.asarray(env.qpos)[0]
        ee_pos, ee_quat = env._mj_ee_pose(q)
        return {
            'joints': q.tolist(),
            'ee_position': ee_pos.tolist(),
            'ee_orientation': ee_quat.tolist(),
            'target_position': np.asarray(env.target_pos)[0].tolist(),
            'target_orientation': np.asarray(env.target_quat)[0].tolist(),
            'obstacles': self._current_specs(),
        }

    def close(self):
        pass


# ==================== 传统 baseline (DLS / 零空间避障) ====================

class AvoidBaselineMJX:
    """传统 IK baseline (DLS / 零空间避障), MuJoCo + MJX SDF 实现

    接口对齐 ik_avoid.baselines.AvoidBaselineSolver:
    set_scene / solve_dls / solve_nullspace / _link_spheres / spawn 场景
    """

    def __init__(self, cfg=None, max_points=768):
        self.cfg = cfg or load_config()
        ec = self.cfg.env

        # MuJoCo 刚体 (FK + 雅可比)
        self.model = mujoco.MjModel.from_xml_path(FIXED_XML)
        self.data = mujoco.MjData(self.model)
        self.n_ctrl = 7
        self.ndof = self.model.nv
        self.joint_limits = [(float(self.model.jnt_range[i][0]),
                              float(self.model.jnt_range[i][1]))
                             for i in range(self.n_ctrl)]
        self.ee_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, 'ee')
        self.link7_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'panda_link7')
        self.home_position = np.array(
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)

        # MJX 几何引擎 (点云 SDF 距离/方向 + 场景采样)
        self.env = RobotIKAvoidEnvMJX(self.cfg, n_worlds=1, max_points=max_points)
        self.monitored_links = list(ec.monitored_links)
        self._mon_groups = self.env.mon_groups
        # 各组雅可比锚点: (body_name, 'frame'|'hand'|'finger')
        self._jac_bodies = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                              f'panda_link{i + 1}') for i in range(7)]
        self._fl_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'panda_leftfinger')
        self._fr_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'panda_rightfinger')
        self._jit_link_dists = jax.jit(self._link_dists_jax)

        # 求解参数 (与 PyBullet 版一致)
        self.max_iterations = 300
        self.pos_tol = ec.pos_tolerance
        self.orient_tol = self.cfg.orient_tolerance_rad
        self.orient_weight = 0.3
        self.damping = 0.05

        self.target_pos = None
        self.target_quat = None

    # ==================== 基础工具 ====================

    def _apply(self, q):
        self.data.qpos[:self.n_ctrl] = np.asarray(q, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

    def _clip(self, q):
        for i in range(self.n_ctrl):
            lo, hi = self.joint_limits[i]
            q[i] = np.clip(q[i], lo, hi)
        return q

    def _ee_pose(self):
        pos = np.array(self.data.site_xpos[self.ee_site])
        quat = mat_to_quat_xyzw(self.data.site_xmat[self.ee_site].reshape(3, 3))
        return pos, quat.astype(np.float32)

    def _orient_error(self, target_quat):
        _, cur_quat = self._ee_pose()
        q_err = quat_multiply(target_quat, quat_conjugate(cur_quat))
        return quat_to_axis_angle(q_err)

    def _jacobian_6d(self, q):
        """末端 6x7 雅可比 (线速度 + 角速度, mj_jac)"""
        self._apply(q)
        jacp = np.zeros((3, self.ndof))
        jacr = np.zeros((3, self.ndof))
        point = self.data.site_xpos[self.ee_site].copy()
        mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.link7_body)
        return jacp[:, :self.n_ctrl], jacr[:, :self.n_ctrl]

    def _jacobian_group(self, q, gi):
        """监控组 gi 锚点的 3x7 线速度雅可比 (零空间避障用)"""
        self._apply(q)
        g = self._mon_groups[gi]
        if g < 7:                                  # link1..7
            bid, point = self._jac_bodies[g], self.data.xpos[self._jac_bodies[g]].copy()
        elif g == 7:                               # hand = ee site 帧位置
            bid, point = self.link7_body, self.data.site_xpos[self.ee_site].copy()
        elif g == 8:                               # finger_l
            bid, point = self._fl_body, self.data.xpos[self._fl_body].copy()
        else:                                      # finger_r
            bid, point = self._fr_body, self.data.xpos[self._fr_body].copy()
        jacp = np.zeros((3, self.ndof))
        mujoco.mj_jac(self.model, self.data, jacp, None, point, bid)
        return jacp[:, :self.n_ctrl]

    def _link_dists_jax(self, q, obs_pos, obs_shape, obs_he, obs_r, obs_cyl):
        """构型 q 下各监控组到最近障碍物的 (dist, dir) — 单世界批量维"""
        fk = self.env._fk_batch(q[None])
        d, v, _ = self.env._link_dists(
            fk, obs_pos[None], obs_shape[None], obs_he[None],
            obs_r[None], obs_cyl[None])
        return d[0], v[0]

    def _link_metrics(self, q):
        """(dist (L,), dir (L,3)) — dir 指向障碍物 (对齐 PyBullet 约定)"""
        env = self.env
        d, v = self._jit_link_dists(
            jnp.asarray(np.asarray(q, dtype=np.float32)),
            env.obs_pos[0], env.obs_shape[0], env.obs_he[0],
            env.obs_r[0], env.obs_cyl[0])
        return np.asarray(d), np.asarray(v)

    def _n_obstacles(self):
        return int(np.asarray(self.env.n_obs)[0])

    def _task_error(self, target_pos, target_quat):
        """6D 任务误差 [pos_err(3); w_o * rot_err(3)]"""
        ee_pos, _ = self._ee_pose()
        pos_err = np.asarray(target_pos) - ee_pos
        rot_err = self._orient_error(target_quat)
        return np.concatenate([pos_err, self.orient_weight * rot_err]), \
            float(np.linalg.norm(pos_err)), float(np.linalg.norm(rot_err))

    def set_scene(self, target_pos, target_quat, obstacle_specs=None,
                  start_q=None):
        """布置场景并返回起始构型"""
        q0 = np.asarray(start_q if start_q is not None else self.home_position,
                        dtype=np.float32).copy()
        self.env.set_scene(0, target_pos, target_quat, self._clip(q0.copy()),
                           obstacle_specs=obstacle_specs or [])
        self._apply(q0)
        self.target_pos = np.asarray(target_pos, dtype=float)
        self.target_quat = quat_normalize(np.asarray(target_quat, dtype=np.float32))
        return q0

    def spawn_scene(self, n, target_pos, dist_range=(0.12, 0.30), clearance=0.05):
        """随机放置障碍物 (走廊采样, 保护当前构型)"""
        protected = self._link_spheres(np.asarray(self.data.qpos[:7]))
        return self.env._spawn_scene_specs(
            n, np.asarray(target_pos), protected, dist_range, clearance)

    def _link_spheres(self, q):
        """构型 q 下监控组包络球 (复用 MJX env 实现)"""
        return self.env._mj_link_spheres(q)

    def get_specs(self):
        """当前障碍物 specs (从 env 张量还原)"""
        env = self.env
        n = self._n_obstacles()
        pos = np.asarray(env.obs_pos)[0]
        shape = np.asarray(env.obs_shape)[0]
        he = np.asarray(env.obs_he)[0]
        r = np.asarray(env.obs_r)[0]
        cyl = np.asarray(env.obs_cyl)[0]
        names = {0: 'box', 1: 'sphere', 2: 'cylinder'}
        specs = []
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

    def _final_metrics(self, q):
        """终态指标: 末端误差 + 最小安全距离 + 碰撞数"""
        self._apply(q)
        ee_pos, ee_quat = self._ee_pose()
        pos_err = float(np.linalg.norm(ee_pos - self.target_pos))
        orient_err = float(quat_angle(ee_quat, self.target_quat))
        if self._n_obstacles() > 0:
            dists, _ = self._link_metrics(q)
            min_dist = float(dists.min())
            n_contacts = 1 if min_dist <= 0 else 0
        else:
            min_dist, n_contacts = float('inf'), 0
        return pos_err, orient_err, min_dist, n_contacts

    def _result(self, q, iterations, t_ms, method):
        pos_err, orient_err, min_dist, n_contacts = self._final_metrics(q)
        # 与 RL 口径对齐: 有障碍时成功还要求 min_dist >= success_margin
        # (rollout_eval / RLAvoidSolverMJX.solve 同判据)
        margin_ok = (self._n_obstacles() == 0
                     or min_dist >= self.cfg.env.success_margin)
        success = (pos_err < self.pos_tol and orient_err < self.orient_tol
                   and n_contacts == 0 and margin_ok)
        return {
            'success': bool(success),
            'method': method,
            'joint_angles': [float(x) for x in np.asarray(q).tolist()],
            'position_error': pos_err,
            'orientation_error': orient_err,
            'orientation_error_deg': float(np.degrees(orient_err)),
            'min_distance': min_dist if np.isfinite(min_dist) else None,
            'n_collisions': n_contacts,
            'iterations': int(iterations),
            'solve_time_ms': round(t_ms, 2),
        }

    def _trajectory_collision_count(self, trajectory):
        """逐帧检查轨迹碰撞 (min_dist<=0 口径)"""
        if not trajectory or self._n_obstacles() == 0:
            return 0
        n_hit = 0
        for q in trajectory:
            dists, _ = self._link_metrics(np.asarray(q, dtype=np.float32))
            if dists.min() <= 0:
                n_hit += 1
        return n_hit

    # ==================== Baseline 1: DLS IK ====================

    def solve_dls(self, target_pos=None, target_quat=None, start_q=None,
                  max_iterations=None):
        """阻尼最小二乘 6D IK: Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ e"""
        if target_pos is None:
            target_pos, target_quat = self.target_pos, self.target_quat
        q = np.asarray(start_q if start_q is not None else self.home_position,
                       dtype=float).copy()
        max_iter = max_iterations or self.max_iterations

        t0 = time.perf_counter()
        w = self.orient_weight
        trajectory = [q.copy()]
        it = 0
        for it in range(max_iter):
            self._apply(q)
            e, pos_d, orient_d = self._task_error(target_pos, target_quat)
            if pos_d < self.pos_tol and orient_d < self.orient_tol:
                break
            Jl, Ja = self._jacobian_6d(q)
            J = np.vstack([Jl, w * Ja])
            JJT = J @ J.T + self.damping**2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJT, e)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = self._clip(q + dq)
            trajectory.append(q.copy())
        t_ms = (time.perf_counter() - t0) * 1000
        result = self._result(q, it + 1, t_ms, 'dls_6d')
        result['trajectory'] = trajectory
        result['trajectory_collisions'] = self._trajectory_collision_count(trajectory)
        return result

    # ==================== Baseline 1b: 碰撞安全 DLS 精调 (RL 混合用) ====================

    def refine_safe_dls(self, start_q=None, max_iterations=100):
        """碰撞安全 DLS 精调: 从近解构型出发收敛到毫米级精度

        与 solve_dls 的差异 (适配 RL 粗解 → 精调场景):
        - 每步候选构型先过 SDF 碰撞复检, 穿入的步长按 1/0.5/0.25 逐级
          缩小重试, 仍碰撞则弃步; 连续 3 次弃步提前终止 (安全帧 best-so-far 保底)
        - 步长上限 0.1 rad (精调段小步走, 避免大步跨过安全区)
        - 轨迹碰撞逐帧复检 (与 RL 终态口径同源)
        """
        q = np.asarray(start_q if start_q is not None else self.home_position,
                       dtype=float).copy()
        max_iter = max_iterations or self.max_iterations
        has_obs = self._n_obstacles() > 0

        t0 = time.perf_counter()
        w = self.orient_weight
        trajectory = [q.copy()]
        self._apply(q)
        e, pos_d, orient_d = self._task_error(self.target_pos, self.target_quat)
        best_q, best_err = q.copy(), float(np.linalg.norm(e))
        n_reject = 0
        it = 0
        for it in range(max_iter):
            if pos_d < self.pos_tol and orient_d < self.orient_tol:
                break
            Jl, Ja = self._jacobian_6d(q)
            J = np.vstack([Jl, w * Ja])
            JJT = J @ J.T + self.damping**2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJT, e)
            n = np.linalg.norm(dq)
            if n > 0.1:
                dq *= 0.1 / n
            accepted = False
            for shrink in (1.0, 0.5, 0.25):
                q_cand = self._clip(q + shrink * dq)
                if has_obs:
                    dists, _ = self._link_metrics(
                        np.asarray(q_cand, dtype=np.float32))
                    if dists.min() <= 0:
                        continue
                q = q_cand
                self._apply(q)
                e, pos_d, orient_d = self._task_error(
                    self.target_pos, self.target_quat)
                err = float(np.linalg.norm(e))
                if err < best_err:
                    best_q, best_err = q.copy(), err
                trajectory.append(q.copy())
                accepted = True
                break
            if accepted:
                n_reject = 0
            else:
                n_reject += 1
                if n_reject >= 3:            # 安全方向被障碍封死, 保留当前最优
                    break
        t_ms = (time.perf_counter() - t0) * 1000
        result = self._result(best_q, it + 1, t_ms, 'dls_refine')
        result['trajectory'] = trajectory
        result['trajectory_collisions'] = self._trajectory_collision_count(trajectory)
        return result

    # ==================== Baseline 2: Null-space avoidance ====================

    def _avoidance_gradient(self, q, rho=0.15, k=1e-3):
        """斥力势场关节空间梯度 g = Σ J_linkᵀ · F_link"""
        if self._n_obstacles() == 0:
            return np.zeros(self.n_ctrl)
        dists, dirs = self._link_metrics(q)
        g = np.zeros(self.n_ctrl)
        for gi in range(len(self._mon_groups)):
            d = dists[gi]
            if d >= rho or d <= 1e-4:
                continue
            # dir 指向障碍物, 斥力取反方向
            f_task = k * (1.0 / d - 1.0 / rho) / (d * d) * (-dirs[gi])
            J_link = self._jacobian_group(q, gi)
            g += J_link.T @ f_task
        return g

    def solve_nullspace(self, target_pos=None, target_quat=None, start_q=None,
                        max_iterations=None, rho=0.15):
        """零空间避障 IK: Δq = J⁺ e + α (I − J⁺ J) g"""
        if target_pos is None:
            target_pos, target_quat = self.target_pos, self.target_quat
        max_iter = max_iterations or self.max_iterations

        q0 = np.asarray(start_q if start_q is not None else self.home_position,
                        dtype=float).copy()

        t0 = time.perf_counter()
        q, iters, traj = self._nullspace_run(q0, target_pos, target_quat,
                                             max_iter, rho)
        t_ms = (time.perf_counter() - t0) * 1000
        result = self._result(q, iters, t_ms, 'nullspace_avoid')
        result['trajectory'] = traj
        result['trajectory_collisions'] = self._trajectory_collision_count(traj)
        return result

    def _nullspace_run(self, q0, target_pos, target_quat, max_iter, rho):
        w = self.orient_weight
        q = q0.copy()
        alpha_n = 1.0
        trajectory = [q.copy()]
        for it in range(max_iter):
            self._apply(q)
            e, pos_d, orient_d = self._task_error(target_pos, target_quat)
            Jl, Ja = self._jacobian_6d(q)
            J = np.vstack([Jl, w * Ja])
            JJT_reg = J @ J.T + self.damping**2 * np.eye(6)
            Jp = J.T @ np.linalg.inv(JJT_reg)          # 6D 阻尼伪逆 (7×6)
            dq_task = Jp @ e
            N = np.eye(self.n_ctrl) - Jp @ J
            g = self._avoidance_gradient(q, rho=rho)
            dq = dq_task + alpha_n * (N @ g)
            n = np.linalg.norm(dq)
            if n > 0.2:
                dq *= 0.2 / n
            q = self._clip(q + dq)
            trajectory.append(q.copy())
            if pos_d < self.pos_tol and orient_d < self.orient_tol:
                # 收敛后只做零空间避障精调
                if np.linalg.norm(N @ g) < 1e-4:
                    break
        return q, it + 1, trajectory

    def close(self):
        pass
