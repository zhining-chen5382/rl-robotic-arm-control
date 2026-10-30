"""
大规模 6D IK 测试 — 1000 个随机可达位姿

测试内容:
  1. 随机生成 1000 个可达目标位姿 (随机关节角 → FK)
  2. 用最新模型从 home 位置求解每个目标
  3. 统计多容差组合下的成功率 + 端到端求解时间
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from ik_6d.rl_env_6d import RobotIK6DEnv, quat_angle

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'panda_ppo_ik_6d_map')

N_TARGETS = 1000
POS_TOLERANCES = [0.005, 0.010, 0.020, 0.050]   # 5mm, 10mm, 20mm, 50mm
ORIENT_TOLERANCES_DEG = [5.0, 10.0, 15.0]        # 5°, 10°, 15°


def generate_targets(env, n):
    """生成 n 个可达目标位姿: 随机关节角 (margin=0.15) → FK"""
    targets = []
    margin = 0.15
    for _ in range(n):
        q = np.zeros(env.n_ctrl, dtype=np.float32)
        for i in range(env.n_ctrl):
            lower, upper = env.joint_limits[i]
            m = (upper - lower) * margin
            q[i] = np.random.uniform(lower + m, upper - m)
        env._apply_joints(q)
        pos = env._get_ee_pos().copy()
        quat = env._get_ee_quat().copy()
        targets.append((pos, quat))
    return targets


def solve_target(env, model, target_pos, target_quat):
    """对单个目标位姿求解, 返回 (pos_dist, orient_dist, solve_ms, steps)"""
    # 手动设置目标 + home 起始 (不调用 reset, 避免随机采样覆盖目标)
    env.target_pos = target_pos.astype(np.float32)
    env.target_quat = quat_normalize(target_quat.astype(np.float32))
    env.current_joints = env.home_position.copy()
    env._clip_joints(env.current_joints)
    env._apply_joints(env.current_joints)
    env.step_count = 0

    ee_pos = env._get_ee_pos()
    env.prev_pos_dist = float(np.linalg.norm(ee_pos - env.target_pos))
    env.prev_orient_dist = quat_angle(env._get_ee_quat(), env.target_quat)
    env._prev_ee_pos = ee_pos.copy()

    obs = env._get_obs()

    # 预热模型 (仅第一次需要)
    t0 = time.perf_counter()
    steps = 0
    for _ in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    t_solve = (time.perf_counter() - t0) * 1000.0

    pos_dist = info.get('pos_distance', float('inf'))
    orient_dist = info.get('orient_distance', float('inf'))
    return pos_dist, orient_dist, t_solve, steps


def quat_normalize(q):
    n = np.linalg.norm(q)
    if n < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def main():
    np.random.seed(42)

    print("=" * 65)
    print(f"  大规模 6D IK 测试 — {N_TARGETS} 个随机可达位姿")
    print("=" * 65)

    # 加载模型
    if not os.path.exists(MODEL_PATH + '.zip'):
        print(f"  模型不存在: {MODEL_PATH}.zip")
        return
    print(f"  加载模型: {MODEL_PATH}.zip")
    model = PPO.load(MODEL_PATH, device='cpu')

    # 创建环境
    env = RobotIK6DEnv(robot_id='panda', max_steps=100,
                       pos_tolerance=0.005, orient_tolerance=0.175)
    env.reset(seed=42)

    # 生成目标
    print(f"  生成 {N_TARGETS} 个可达目标位姿...")
    targets = generate_targets(env, N_TARGETS)

    # 预热模型
    dummy_obs = env._get_obs()
    for _ in range(20):
        model.predict(dummy_obs, deterministic=True)

    # 求解所有目标
    print(f"  开始求解...\n")
    results = []
    t_start = time.perf_counter()
    for i, (tp, tq) in enumerate(targets):
        pd, od, t_ms, steps = solve_target(env, model, tp, tq)
        results.append((pd, od, t_ms, steps))
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"    [{i+1}/{N_TARGETS}]  已用 {elapsed:.1f}s")
    total_time = time.perf_counter() - t_start
    env.close()

    results = np.array(results)
    pos_dists = results[:, 0]
    orient_dists = results[:, 1]
    solve_times = results[:, 2]
    steps_arr = results[:, 3]

    # ============ 成功率矩阵 ============
    print("\n" + "=" * 65)
    print("  成功率矩阵 (%)")
    print("=" * 65)
    header = f"  {'':>10}"
    for ot_deg in ORIENT_TOLERANCES_DEG:
        header += f" {'姿态<'+str(int(ot_deg))+'°':>10}"
    print(header)
    print(f"  {'-' * (10 + len(ORIENT_TOLERANCES_DEG) * 11)}")
    for pt in POS_TOLERANCES:
        row = f"  {'位置<'+str(int(pt*1000))+'mm':>10}"
        for ot_deg in ORIENT_TOLERANCES_DEG:
            ot = np.radians(ot_deg)
            rate = np.mean((pos_dists < pt) & (orient_dists < ot)) * 100
            row += f" {rate:>9.1f}%"
        print(row)

    # ============ 边际成功率 ============
    print("\n" + "=" * 65)
    print("  边际成功率")
    print("=" * 65)
    for pt in POS_TOLERANCES:
        rate = np.mean(pos_dists < pt) * 100
        print(f"    位置 < {int(pt*1000)}mm:  {rate:.1f}%")
    for ot_deg in ORIENT_TOLERANCES_DEG:
        ot = np.radians(ot_deg)
        rate = np.mean(orient_dists < ot) * 100
        print(f"    姿态 < {ot_deg:.0f}°:  {rate:.1f}%")

    # ============ 误差统计 ============
    print("\n" + "=" * 65)
    print("  误差统计")
    print("=" * 65)
    print(f"    位置误差  — 平均 {np.mean(pos_dists)*1000:.2f} mm, 中位 {np.median(pos_dists)*1000:.2f} mm")
    print(f"                 最小 {np.min(pos_dists)*1000:.2f} mm, 最大 {np.max(pos_dists)*1000:.2f} mm")
    print(f"    姿态误差  — 平均 {np.degrees(np.mean(orient_dists)):.2f}°, 中位 {np.degrees(np.median(orient_dists)):.2f}°")
    print(f"                 最小 {np.degrees(np.min(orient_dists)):.2f}°, 最大 {np.degrees(np.max(orient_dists)):.2f}°")

    # ============ 求解时间统计 ============
    print("\n" + "=" * 65)
    print("  端到端求解时间 (目标确定 → 输出关节角)")
    print("=" * 65)
    print(f"    平均:   {np.mean(solve_times):.1f} ms")
    print(f"    中位数: {np.median(solve_times):.1f} ms")
    print(f"    最快:   {np.min(solve_times):.1f} ms")
    print(f"    最慢:   {np.max(solve_times):.1f} ms")
    print(f"    P95:    {np.percentile(solve_times, 95):.1f} ms")
    print(f"    总耗时: {total_time:.1f} s (平均 {total_time/N_TARGETS*1000:.1f} ms/个)")
    print(f"    迭代步数 — 平均 {np.mean(steps_arr):.1f} 步, 中位 {np.median(steps_arr):.0f} 步")

    # ============ 联合成功率 (主指标) ============
    print("\n" + "=" * 65)
    print("  联合成功率 (位置 AND 姿态 同时达标)")
    print("=" * 65)
    for pt in POS_TOLERANCES:
        for ot_deg in ORIENT_TOLERANCES_DEG:
            ot = np.radians(ot_deg)
            count = np.sum((pos_dists < pt) & (orient_dists < ot))
            print(f"    位置<{int(pt*1000)}mm 且 姿态<{ot_deg:.0f}°: {count}/{N_TARGETS} = {count/N_TARGETS*100:.1f}%")

    print("\n" + "=" * 65)
    print("  测试完成")
    print("=" * 65)


if __name__ == '__main__':
    main()
