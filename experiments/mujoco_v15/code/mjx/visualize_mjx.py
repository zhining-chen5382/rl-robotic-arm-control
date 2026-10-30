"""MuJoCo 原生可视化: 避障 IK 求解回放

用 mujoco.viewer 交互窗口渲染 Panda + 障碍物 + 目标位姿, 回放 RL 求解轨迹。
障碍物/目标用预留 geom 插槽运行时更新 (位置/尺寸/显隐)。

用法:
    python mjx/visualize_mjx.py --case 3                # 第 3 个随机障碍场景
    python mjx/visualize_mjx.py --case 0 --free         # 无障碍场景
    python mjx/visualize_mjx.py --model models/panda_ppo_ik_avoid_mjx_best/best_model
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer

from ik_avoid.config import load_config
from mjx.avoid_solvers import RLAvoidSolverMJX

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXED_XML = os.path.join(BASE_DIR, 'mjx', 'assets', 'panda_fixed.xml')

MAX_O = 3  # 障碍物插槽数


def quat_mul_wxyz(a, b):
    """四元数乘法 (wxyz)"""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw])


def quat_to_mat_wxyz(q):
    """四元数 (wxyz) -> 旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])


def build_model():
    """加载 Panda MJCF 并挂接障碍物/目标 geom 插槽"""
    spec = mujoco.MjSpec.from_file(FIXED_XML)
    wb = spec.worldbody

    # 障碍物插槽: 每插槽 3 种形状各一, 运行时只显示匹配形状
    obs_geoms = {'box': [], 'sphere': [], 'cylinder': []}
    colors = {'box': [0.9, 0.35, 0.2, 0.85],
              'sphere': [0.9, 0.55, 0.2, 0.85],
              'cylinder': [0.9, 0.35, 0.6, 0.85]}
    for i in range(MAX_O):
        for shape, (sz, cls) in {
                'box': ([0.05, 0.05, 0.05], None),
                'sphere': ([0.05, 0, 0], None),
                'cylinder': ([0.05, 0.1, 0], None)}.items():
            g = wb.add_geom()
            g.name = f'obs_{shape}_{i}'
            g.pos = [0, 0, -1000]          # 初始移出视野
            g.type = {'box': mujoco.mjtGeom.mjGEOM_BOX,
                      'sphere': mujoco.mjtGeom.mjGEOM_SPHERE,
                      'cylinder': mujoco.mjtGeom.mjGEOM_CYLINDER}[shape]
            g.size = sz
            g.rgba = colors[shape]
            g.contype = 0
            g.conaffinity = 0
            obs_geoms[shape].append(g)

    # 目标标记: 半透明球 + xyz 轴杆
    tg = wb.add_geom()
    tg.name = 'tgt_sph'
    tg.type = mujoco.mjtGeom.mjGEOM_SPHERE
    tg.size = [0.025, 0, 0]
    tg.rgba = [0.2, 0.8, 0.3, 0.55]
    tg.contype = 0
    tg.conaffinity = 0
    axis_geoms = []
    ax_colors = [[0.95, 0.2, 0.2, 1], [0.2, 0.95, 0.2, 1], [0.2, 0.4, 0.95, 1]]
    for k in range(3):
        ag = wb.add_geom()
        ag.name = f'tgt_ax_{k}'
        ag.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        ag.size = [0.004, 0.045, 0]
        ag.rgba = ax_colors[k]
        ag.contype = 0
        ag.conaffinity = 0
        axis_geoms.append(ag)

    model = spec.compile()
    ids = {
        'obs': {shape: [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                           f'obs_{shape}_{i}')
                        for i in range(MAX_O)] for shape in obs_geoms},
        'tgt': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'tgt_sph'),
        'axes': [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                   f'tgt_ax_{k}') for k in range(3)],
    }
    return model, ids


# 轴杆局部四元数 (wxyz): 圆柱沿 z, 转到 x/y/z 轴
_AXIS_LOCAL_QUAT = [
    np.array([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0]),   # z -> x
    np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0]),   # z -> y
    np.array([1.0, 0, 0, 0]),                             # z -> z
]
_AXIS_LOCAL_OFF = np.array([[0.045, 0, 0], [0, 0.045, 0], [0, 0, 0.045]])


def update_scene(model, ids, scene):
    """把场景 (目标+障碍物) 写入 geom 插槽"""
    # 目标
    tp = np.asarray(scene['target_pos'], dtype=np.float64)
    tq = np.asarray(scene['target_quat'], dtype=np.float64)  # xyzw
    tq_wxyz = np.array([tq[3], tq[0], tq[1], tq[2]])
    model.geom_pos[ids['tgt']] = tp
    R = quat_to_mat_wxyz(tq_wxyz)
    for k, gid in enumerate(ids['axes']):
        model.geom_pos[gid] = tp + R @ _AXIS_LOCAL_OFF[k]
        model.geom_quat[gid] = quat_mul_wxyz(tq_wxyz, _AXIS_LOCAL_QUAT[k])

    # 障碍物: 全部先隐藏, 再按 specs 显示
    for shape in ('box', 'sphere', 'cylinder'):
        for gid in ids['obs'][shape]:
            model.geom_pos[gid] = [0, 0, -1000]
    for i, spec in enumerate(scene['obstacle_specs'][:MAX_O]):
        shape = spec['shape']
        gid = ids['obs'][shape][i]
        model.geom_pos[gid] = np.asarray(spec['pos'], dtype=np.float64)
        if shape == 'box':
            model.geom_size[gid] = np.asarray(spec['size']['halfExtents'])
        elif shape == 'sphere':
            model.geom_size[gid] = [spec['size']['radius'], 0, 0]
        else:
            model.geom_size[gid] = [spec['size']['radius'],
                                    spec['size']['height'] / 2, 0]


def play_trajectory(model, data, trajectory, dt=0.04, loop=2):
    """在 viewer 中回放关节轨迹"""
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _ in range(loop):
            for q in trajectory:
                if not viewer.is_running():
                    return
                data.qpos[:7] = np.asarray(q)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(dt)
            time.sleep(0.8)


def main(args):
    # 默认优先 avoid_remote30m.yaml (49 维观测, 与 MJX best 模型配套)
    config = args.config
    if config is None:
        cand = os.path.join(BASE_DIR, 'configs', 'avoid_remote30m.yaml')
        config = cand if os.path.exists(cand) else None
    cfg = load_config(config)
    seed, case = args.seed, args.case

    solver = RLAvoidSolverMJX(cfg, model_path=args.model)
    scenes = solver.env.generate_scenes(case + 1, seed,
                                        with_obstacles=not args.free)
    scene = scenes[case]
    print(f"  场景 {case} (seed={seed}, {'无障碍' if args.free else '障碍'}): "
          f"{len(scene['obstacle_specs'])} 个障碍物")

    result = solver.solve(scene['target_pos'], scene['target_quat'],
                          obstacle_specs=scene['obstacle_specs'],
                          start_q=scene['start_q'], record_trajectory=True)
    solver.close()
    print(f"  求解: success={result['success']}, "
          f"pos={result['position_error']*1000:.1f}mm, "
          f"ori={result['orientation_error_deg']:.2f}deg, "
          f"碰撞={result['n_collisions']}, 迭代={result['iterations']}")

    model, ids = build_model()
    data = mujoco.MjData(model)
    update_scene(model, ids, scene)
    play_trajectory(model, data, result['trajectory'],
                    dt=args.dt, loop=args.loop)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MuJoCo 避障 IK 可视化')
    parser.add_argument('--case', type=int, default=0, help='场景序号')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--free', action='store_true', help='无障碍场景')
    parser.add_argument('--model', type=str, default=None,
                        help='RL 模型路径 (不含 .zip), 默认 avoid 配置 best')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dt', type=float, default=0.04)
    parser.add_argument('--loop', type=int, default=2, help='轨迹回放轮数')
    args = parser.parse_args()
    main(args)
