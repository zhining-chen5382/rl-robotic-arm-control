"""评测统计与批量 rollout 工具 (纯 numpy + MJX env 接口, 无 PyBullet 依赖)

rollout_eval 供 mjx/eval_mjx.py (MJX 批量评测) 与训练期
MJXBatchEvalCallback 复用, 两者共用同一套口径与汇总逻辑。
"""
import time

import numpy as np

POS_TOL_MM = [5.0, 10.0, 20.0]
ORIENT_TOL_DEG = [3.0, 5.0, 10.0]

# 严格成功率专用判据 (评测口径, 独立于训练配置)
STRICT_POS_MM = 10.0
STRICT_ORIENT_DEG = 10.0


def quat_normalize(q):
    """四元数归一化 (与 ik_6d.rl_env_6d.quat_normalize 完全一致), q 为 [x,y,z,w]"""
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q)
    if n < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return q / n


def summarize(results, label):
    """汇总单方法结果"""
    n = len(results)
    if n == 0:
        return None
    pos = np.array([r['position_error'] for r in results])
    ori = np.array([r['orientation_error'] for r in results])
    min_d = np.array([r['min_distance'] for r in results
                      if r['min_distance'] is not None])
    iters = np.array([r['iterations'] for r in results])
    t_ms = np.array([r['solve_time_ms'] for r in results])
    collisions = sum(1 for r in results if r['n_collisions'] > 0)
    # 轨迹碰撞 (含中途碰撞被 best-so-far 掩盖的案例; 旧结果无此字段时回退终态口径)
    traj_collisions = sum(1 for r in results
                          if r.get('trajectory_collisions',
                                   r['n_collisions']) > 0)

    strict_pos = STRICT_POS_MM / 1000
    strict_ori = np.radians(STRICT_ORIENT_DEG)
    summary = {
        'label': label,
        'n': n,
        'strict_success_rate': float(np.mean([
            (r['position_error'] < strict_pos
             and r['orientation_error'] < strict_ori
             and r['n_collisions'] == 0)
            for r in results])),
        'collision_rate': collisions / n,
        'trajectory_collision_rate': traj_collisions / n,
        'pos_err_mm_median': float(np.median(pos) * 1000),
        'orient_err_deg_median': float(np.degrees(np.median(ori))),
        'min_dist_median': float(np.median(min_d)) if len(min_d) else None,
        'iterations_mean': float(np.mean(iters)),
        'time_ms_mean': float(np.mean(t_ms)),
        'tolerance_matrix': {},
    }
    for pt in POS_TOL_MM:
        for ot in ORIENT_TOL_DEG:
            ok = np.mean([(r['position_error'] < pt / 1000
                           and r['orientation_error'] < np.radians(ot)
                           and r['n_collisions'] == 0)
                          for r in results])
            summary['tolerance_matrix'][f'<{pt:.0f}mm&<{ot:.0f}deg'] = float(ok)
    return summary


def print_summary(s):
    print(f"\n  [{s['label']}]  n={s['n']}")
    print(f"    严格成功率 (<{STRICT_POS_MM:.0f}mm&<{STRICT_ORIENT_DEG:.0f}°+无碰撞): {s['strict_success_rate']*100:.1f}%")
    print(f"    碰撞率 (终态): {s['collision_rate']*100:.1f}% | "
          f"轨迹碰撞率 (含中途): {s['trajectory_collision_rate']*100:.1f}%")
    print(f"    位置中位误差: {s['pos_err_mm_median']:.2f} mm | "
          f"姿态中位误差: {s['orient_err_deg_median']:.2f} deg")
    if s['min_dist_median'] is not None:
        print(f"    最小安全距离中位数: {s['min_dist_median']*1000:.1f} mm")
    print(f"    平均迭代: {s['iterations_mean']:.1f} | "
          f"平均耗时: {s['time_ms_mean']:.1f} ms")
    print("    容差成功率:")
    keys = list(s['tolerance_matrix'].keys())
    row = "      " + "".join(f"{k:>14}" for k in keys)
    vals = "      " + "".join(
        f"{s['tolerance_matrix'][k]*100:>13.1f}%" for k in keys)
    print(row)
    print(vals)


def inject_scenes(env, scenes):
    """把场景批量注入 MJX 环境 (世界 i <- scenes[i])"""
    n = len(scenes)
    idx = np.arange(n)
    start_q = np.array([sc['start_q'] for sc in scenes], dtype=np.float32)
    tgt_pos = np.array([sc['target_pos'] for sc in scenes], dtype=np.float32)
    tgt_quat = np.array([quat_normalize(np.asarray(sc['target_quat'],
                                                   dtype=np.float32))
                          for sc in scenes], dtype=np.float32)

    MAX_O = 3  # 障碍物插槽数 (与训练 stage3 多障碍上限一致)
    op = np.zeros((n, MAX_O, 3), dtype=np.float32)
    op[:, :, 2] = -1000.0
    osh = np.zeros((n, MAX_O), dtype=np.int32)
    ohe = np.zeros((n, MAX_O, 3), dtype=np.float32)
    orad = np.zeros((n, MAX_O), dtype=np.float32)
    ocyl = np.zeros((n, MAX_O, 2), dtype=np.float32)
    no = np.zeros(n, dtype=np.int32)
    for i, sc in enumerate(scenes):
        p, sh, he, r, cy, ni = env._specs_to_arrays(sc['obstacle_specs'])
        op[i], osh[i], ohe[i] = p, sh, he
        orad[i], ocyl[i], no[i] = r, cy, ni

    env._setup_worlds_batch(idx, start_q, tgt_pos, tgt_quat,
                            op, osh, ohe, orad, ocyl, no)


def rollout_eval(env, policy, scenes, max_steps):
    """批量 rollout 评测, 返回与 RLAvoidSolver.solve 同构的 results 列表

    复刻评测口径: best-so-far (非碰撞帧按耦合距离记录最优), 碰撞/成功帧终止;
    碰撞判定用终态 (best 帧) min_dist<=0, 与 PyBullet n_final_contacts 一致。
    policy 为 SB3 模型 (用 .predict) 或任意 obs->actions 的可调用对象。
    """
    n = len(scenes)
    cfg = env.cfg
    has_obs = np.asarray(env.n_obs)[:n] > 0
    start_q = np.asarray(env.qpos)[:n].copy()

    # 初始帧作为 best 起点 (对应 RLAvoidSolver 的初始 best)
    pd0 = np.asarray(env.prev_pos_dist)[:n].copy()
    od0 = np.asarray(env.prev_orient_dist)[:n].copy()
    md0 = env.get_min_dist_batch()[:n]
    md0 = np.where(has_obs, md0, 0.5)  # 无障碍世界哨兵值 -> 500mm
    cp0 = np.asarray(env._coupled_distance(pd0, od0))

    best_joints = start_q.copy()
    best_coupled = cp0.copy()
    best_pos, best_ori, best_md = pd0.copy(), od0.copy(), md0.copy()

    done = np.zeros(n, dtype=bool)          # 碰撞或成功后冻结
    traj_coll_frames = np.zeros(n, dtype=np.int32)
    iterations = np.full(n, max_steps, dtype=np.int32)

    obs = env.get_obs_batch()[:n]
    t0 = time.perf_counter()
    for t in range(max_steps):
        if done.all():
            break
        if hasattr(policy, 'predict'):
            actions, _ = policy.predict(obs, deterministic=True)
        else:
            actions = np.asarray(policy(obs))
        if done.any():
            actions[done] = 0.0
        obs, _, term, _, md = env.step_batch(
            actions, return_min_dist=True)
        md = np.where(has_obs, md, 0.5)
        pd = np.asarray(env.prev_pos_dist)[:n]
        od = np.asarray(env.prev_orient_dist)[:n]

        coll = (md <= 0.0) & has_obs
        succ = term & ~coll
        frame_ok = ~done & ~coll            # 非碰撞帧可入 best (含成功帧)
        coupled = np.asarray(env._coupled_distance(pd, od))
        upd = frame_ok & (coupled < best_coupled)
        best_joints[upd] = np.asarray(env.qpos)[:n][upd]
        best_coupled[upd] = coupled[upd]
        best_pos[upd], best_ori[upd], best_md[upd] = pd[upd], od[upd], md[upd]

        traj_coll_frames[coll & ~done] += 1
        newly = (coll | succ) & ~done
        iterations[newly] = t + 1
        done |= newly

    t_ms = (time.perf_counter() - t0) * 1000

    results = []
    tol_p, tol_o = env.pos_tolerance, env.orient_tolerance
    margin = cfg.env.success_margin
    for i in range(n):
        ho = bool(has_obs[i])
        n_final = 1 if best_md[i] <= 0 else 0
        success = (best_pos[i] < tol_p and best_ori[i] < tol_o
                   and n_final == 0
                   and (not ho or best_md[i] >= margin))
        results.append({
            'success': bool(success),
            'method': 'rl_ppo_avoid_mjx',
            'joint_angles': [float(x) for x in best_joints[i]],
            'position_error': float(best_pos[i]),
            'orientation_error': float(best_ori[i]),
            'orientation_error_deg': float(np.degrees(best_ori[i])),
            'min_distance': float(best_md[i]),
            'n_collisions': n_final,
            'trajectory_collisions': int(traj_coll_frames[i]),
            'iterations': int(iterations[i]),
            'solve_time_ms': round(t_ms / n, 3),
            'n_obstacles': int(min(len(scenes[i]['obstacle_specs']), 3)),
            'obstacle_specs': scenes[i]['obstacle_specs'],
            'target_position': scenes[i]['target_pos'],
            'target_orientation': scenes[i]['target_quat'],
        })
    return results
