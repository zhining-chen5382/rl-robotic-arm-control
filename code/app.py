"""IK可视化系统 - Flask后端
Franka Panda (7轴) 避障逆解
基于 PPO 强化学习的 RL IK 求解器
"""
import os
# Web 单世界推理强制 CPU 后端 (必须在任何 jax 导入前设置):
# 实测 GPU 单步 20.8ms (n_worlds=1 时内核派发开销摊不薄) vs CPU 11.6ms,
# 且释放 GPU 显存给训练/评测进程
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

from flask import Flask, jsonify, request, send_from_directory, render_template, g
from mjx.mj_solver import MJRobotSolver
from robot_configs import get_robot_list, get_robot_config
from ik_3d.rl_ik_solver import RLIKSolver, is_model_available
from ik_6d.rl_ik_solver_6d import RLIK6DSolver
import threading
import numpy as np
import time
from deployment import (SOFTWARE_NAME, MODEL, CONFIG, MAX_STEPS,
                        OBSERVATION_MAX_STEPS, METHODS, check_files)
from ik_avoid.eval_stats import trajectory_safe_success

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORTED_ROBOT_ID = 'panda'

app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
_request_lock = threading.RLock()


@app.before_request
def serialize_solver_requests():
    # All legacy APIs share MuJoCo/JAX scene state, including FK animation.
    if request.path.startswith('/api/'):
        _request_lock.acquire()
        g.solver_lock_acquired = True


@app.teardown_request
def release_solver_requests(error):
    if g.pop('solver_lock_acquired', False):
        _request_lock.release()


@app.route('/api/software_info')
def software_info():
    return jsonify(name=SOFTWARE_NAME, model=MODEL.name,
                   model_path=str(MODEL.relative_to(_BASE_DIR)),
                   config=str(CONFIG.relative_to(_BASE_DIR)), max_steps=MAX_STEPS,
                   observation_max_steps=OBSERVATION_MAX_STEPS,
                   training_enabled=False, methods=METHODS,
                   robot_id=SUPPORTED_ROBOT_ID,
                   success_rule='位置<10mm，姿态<10°，终态及采样轨迹零障碍碰撞',
                   collision_scope='监控连杆点云与外部障碍物；非连续碰撞或实机安全认证')

# 全局IK求解器实例 (默认Panda, MuJoCo 原生后端)
solver = MJRobotSolver(SUPPORTED_ROBOT_ID)

# RL 求解器 (懒加载, 按机器人缓存)
rl_solvers = {}
rl_solver_lock = threading.Lock()

# 6D 位姿 RL 求解器 (复用主 solver 的 MuJoCo 实例)
rl6d_solver = None
rl6d_solver_lock = threading.Lock()

# 训练状态
training_status = {
    'training': False,
    'progress': '',
    'error': None,
}


def _legacy_rl_unavailable():
    """Explain which trained policy is included without loading a substitute."""
    return jsonify(
        success=False,
        error='本发布仅附带 v30 避障策略，不包含旧 3D/6D RL 权重。'
              '请使用 /api/solve_ik_avoid 的发布方法。',
        supported_endpoint='/api/solve_ik_avoid', methods=METHODS), 503


def _legacy_6d_model_available():
    return os.path.isfile(os.path.join(
        _BASE_DIR, 'models', 'panda_ppo_ik_6d_map.zip'))


def get_rl_solver(robot_id='panda'):
    """懒加载 RL 求解器"""
    if robot_id not in rl_solvers:
        with rl_solver_lock:
            if robot_id not in rl_solvers:
                rl_solvers[robot_id] = RLIKSolver(robot_id=robot_id)
    return rl_solvers[robot_id]


def get_rl6d_solver():
    """懒加载 6D 位姿 RL 求解器 (复用主 solver 的 MuJoCo 实例)"""
    global rl6d_solver
    if rl6d_solver is None:
        with rl6d_solver_lock:
            if rl6d_solver is None:
                rl6d_solver = RLIK6DSolver(solver)
    return rl6d_solver


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/urdf/meshes/visual/<path:filename>')
def serve_mesh(filename):
    """提供Panda mesh文件访问"""
    mesh_dir = os.path.join(os.path.dirname(__file__), 'urdf', 'meshes', 'visual')
    return send_from_directory(mesh_dir, filename)


@app.route('/urdf/meshes/collision/<path:filename>')
def serve_collision_mesh(filename):
    """提供collision mesh文件访问"""
    mesh_dir = os.path.join(os.path.dirname(__file__), 'urdf', 'meshes', 'collision')
    return send_from_directory(mesh_dir, filename)


@app.route('/api/robots')
def list_robots():
    """获取可用机械臂列表"""
    return jsonify(get_robot_list())


@app.route('/api/switch_robot', methods=['POST'])
def switch_robot():
    """切换当前机械臂

    请求体: {"robot_id": "panda"}
    """
    global solver
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error='请求必须是 JSON 对象', success=False), 400
    robot_id = data.get('robot_id', SUPPORTED_ROBOT_ID)
    if robot_id != SUPPORTED_ROBOT_ID:
        return jsonify(error='本发布仅支持 Panda 机械臂', success=False), 400

    try:
        solver = MJRobotSolver(robot_id)
        # 切换机器人时重置 6D 求解器缓存
        global rl6d_solver
        with rl6d_solver_lock:
            if rl6d_solver is not None:
                rl6d_solver = None
        info = solver.get_robot_info()
        fk = solver.forward_kinematics()
        return jsonify({
            'success': True,
            'robot_info': info,
            'ee_position': fk['position'],
            'ee_orientation': fk['orientation'],
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/robot_info')
def get_robot_info():
    """获取当前机器人信息"""
    info = solver.get_robot_info()
    fk = solver.forward_kinematics()
    return jsonify({
        **info,
        'current_angles': solver.joint_angles,
        'ee_position': fk['position'],
        'ee_orientation': fk['orientation'],
    })


@app.route('/api/solve_ik', methods=['POST'])
def solve_ik():
    """求解逆运动学
    
    请求体:
    {
        "target_pos": [x, y, z],
        "target_orn": [x, y, z, w] (可选),
        "method": "jacobian" | "dls" | "transpose" | "pseudoinverse" | "rl",
        "jac_mode": "pybullet" | "manual" (默认 "pybullet")
    }
    """
    data = request.json
    target_pos = data.get('target_pos')

    if target_pos is None:
        return jsonify({'error': '请提供target_pos [x, y, z]'}), 400

    target_orn = data.get('target_orn')
    method = data.get('method', 'jacobian')
    jac_mode = data.get('jac_mode', 'pybullet')

    try:
        # RL 方法单独处理
        if method == 'rl':
            global solver
            current_robot = solver.robot_id_str
            
            if not is_model_available(current_robot):
                return _legacy_rl_unavailable()
            
            import time as _time
            rl = get_rl_solver(current_robot)
            t0 = _time.perf_counter()
            result = rl.solve(target_pos)
            elapsed_ms = (_time.perf_counter() - t0) * 1000
            result['solve_time_ms'] = round(elapsed_ms, 2)
            # RL 求解器用自建 MuJoCo 实例, 用主 solver 获取 link_transforms
            solver.joint_angles = result['joint_angles']
            solver._apply_joint_angles()
            transforms = solver.get_all_link_transforms(result['joint_angles'])
            result['link_transforms'] = transforms
            return jsonify(result)

        import time as _time
        t0 = _time.perf_counter()
        result = solver.solve_ik(target_pos, target_orn, method=method, jac_mode=jac_mode)
        elapsed_ms = (_time.perf_counter() - t0) * 1000
        result['solve_time_ms'] = round(elapsed_ms, 2)
        transforms = solver.get_all_link_transforms(result['joint_angles'])
        result['link_transforms'] = transforms
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/forward_kinematics', methods=['POST'])
def forward_kinematics():
    """正运动学: 给定关节角度, 返回末端位姿"""
    data = request.json
    joint_angles = data.get('joint_angles')

    if joint_angles is None:
        return jsonify({'error': '请提供joint_angles'}), 400

    try:
        fk = solver.forward_kinematics(joint_angles)
        transforms = solver.get_all_link_transforms(joint_angles)
        return jsonify({
            'position': fk['position'],
            'orientation': fk['orientation'],
            'link_transforms': transforms,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_state')
def get_state():
    """获取当前机器人状态"""
    fk = solver.forward_kinematics()
    transforms = solver.get_all_link_transforms()
    return jsonify({
        'joint_angles': solver.joint_angles,
        'ee_position': fk['position'],
        'ee_orientation': fk['orientation'],
        'link_transforms': transforms,
    })


@app.route('/api/solve_path', methods=['POST'])
def solve_path():
    """求解运动路径
    
    请求体:
    {
        "target_pos": [x, y, z],
        "steps": 30,
        "method": "jacobian" (默认),
        "jac_mode": "pybullet" | "manual" (默认 "pybullet")
    }
    """
    data = request.json
    target_pos = data.get('target_pos')
    steps = data.get('steps', 30)
    method = data.get('method', 'jacobian')
    jac_mode = data.get('jac_mode', 'pybullet')

    if target_pos is None:
        return jsonify({'error': '请提供target_pos'}), 400

    try:
        start_angles = list(solver.joint_angles)
        start_fk = solver.forward_kinematics(start_angles)
        start_pos = start_fk['position']

        path_angles = solver.solve_path(start_angles, target_pos, steps, method=method, jac_mode=jac_mode)

        path_data = []
        for angles in path_angles:
            fk = solver.forward_kinematics(angles)
            path_data.append({
                'joint_angles': angles,
                'ee_position': fk['position'],
            })

        return jsonify({
            'success': True,
            'path': path_data,
            'start_position': start_pos,
            'target_position': target_pos,
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/reset', methods=['POST'])
def reset():
    """重置到初始位置"""
    home = list(solver.config.get('home_position', [0.0] * solver.n_ctrl))
    solver.joint_angles = list(home[:solver.n_ctrl])
    solver._apply_joint_angles()
    fk = solver.forward_kinematics()
    transforms = solver.get_all_link_transforms()
    return jsonify({
        'joint_angles': solver.joint_angles,
        'ee_position': fk['position'],
        'ee_orientation': fk['orientation'],
        'link_transforms': transforms,
    })


@app.route('/api/solve_ik_6d', methods=['POST'])
def solve_ik_6d():
    """6D 位姿 IK 求解 (RL MAPPO-IK)

    请求体:
    {
        "target_pos": [x, y, z],
        "target_orn": [x, y, z, w]   (四元数, 可选)
        "target_rpy": [r, p, y]      (RPY 角度, 可选, 优先级低于 target_orn)
    }
    """
    if not _legacy_6d_model_available():
        return _legacy_rl_unavailable()
    data = request.json
    target_pos = data.get('target_pos')
    if target_pos is None:
        return jsonify({'error': '请提供 target_pos [x, y, z]'}), 400

    # 目标姿态: 优先用四元数, 否则从 RPY 转换
    target_orn = data.get('target_orn')
    if target_orn is None:
        rpy = data.get('target_rpy', [0, 0, 0])
        target_orn = _rpy_to_quaternion(rpy)

    try:
        rl6d = get_rl6d_solver()
        import time as _time
        t0 = _time.perf_counter()
        result = rl6d.solve(target_pos, target_orn)
        elapsed_ms = (_time.perf_counter() - t0) * 1000
        result['solve_time_ms'] = round(elapsed_ms, 2)

        # 用主 solver 获取 link_transforms 供前端渲染
        solver.joint_angles = result['joint_angles']
        solver._apply_joint_angles()
        result['link_transforms'] = solver.get_all_link_transforms(result['joint_angles'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/random_pose')
def random_pose():
    """生成随机可达 6D 位姿 (用于自动演示)"""
    if not _legacy_6d_model_available():
        return _legacy_rl_unavailable()
    try:
        rl6d = get_rl6d_solver()
        pos, quat, q = rl6d.random_reachable_pose()
        return jsonify({
            'position': pos,
            'orientation': quat,
            'joint_angles': q,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== 避障 IK 接口 ====================

# V1.0 deployment is pinned in deployment.py; no silent model fallback.
_avoid_rl_solver = None
_avoid_rl_hybrid_solver = None
_avoid_baseline = None
_avoid_lock = threading.Lock()


def _pick_avoid_model():
    """选优先级最高的已存在模型+配置配套 (均无则返回 (None, None))"""
    return check_files()


def _load_avoid_config():
    from ik_avoid.config import load_config
    _, cfg_path = _pick_avoid_model()
    cfg = load_config(cfg_path)
    if cfg.env.robot_id != SUPPORTED_ROBOT_ID:
        raise ValueError('本发布的避障模型和几何仅支持 Panda')
    return cfg


def _avoid_deployment_kwargs(model_path):
    """Deployment-only overrides for the validated v28 horizon-150 variant."""
    return {'max_steps': MAX_STEPS,
            'observation_max_steps': OBSERVATION_MAX_STEPS}


def get_avoid_rl_solver():
    global _avoid_rl_solver
    if _avoid_rl_solver is None:
        with _avoid_lock:
            if _avoid_rl_solver is None:
                # numpy 快速推理链路 (与 jax 版统计等效, 纯 RL 提速 ~3x,
                # hybrid ~6x; 接口/返回字段与 RLAvoidSolverMJX 一致)
                # infer_max_points=256: step 闭环点云抽稀再提速 1.3x,
                # 碰撞判定类路径仍全量点云 (n=148 验收零漏判)；SDF 仅计算
                # 实际障碍物插槽，无障碍直接短路（逐案数值等价，§6.14）；
                # 全点只算距离，方向延迟到各组最近点后计算（§6.15）；
                # 四元数旋转使用末轴 3D 专用叉积（§6.17），FK 关节循环使用
                # 分量化融合实现，避免反复 stack/broadcast（§6.18）；单构型
                # 的 10 组监控点云合并为一次批量旋转（§6.19）
                from mjx.fast_inference import FastRLAvoidSolverMJX
                model_path, _ = _pick_avoid_model()
                _avoid_rl_solver = FastRLAvoidSolverMJX(
                    _load_avoid_config(), model_path=model_path,
                    infer_max_points=256,
                    **_avoid_deployment_kwargs(model_path))
    return _avoid_rl_solver


def get_avoid_rl_hybrid_solver():
    """RL 粗调 + 碰撞安全 DLS 精调的混合求解器 (numpy 快速链路)"""
    global _avoid_rl_hybrid_solver
    if _avoid_rl_hybrid_solver is None:
        with _avoid_lock:
            if _avoid_rl_hybrid_solver is None:
                from mjx.fast_inference import FastRLAvoidSolverMJX
                model_path, _ = _pick_avoid_model()
                _avoid_rl_hybrid_solver = FastRLAvoidSolverMJX(
                    _load_avoid_config(), model_path=model_path,
                    use_dls_refine=True, infer_max_points=256,
                    **_avoid_deployment_kwargs(model_path))
    return _avoid_rl_hybrid_solver


def get_avoid_baseline():
    global _avoid_baseline
    if _avoid_baseline is None:
        with _avoid_lock:
            if _avoid_baseline is None:
                from mjx.avoid_solvers import AvoidBaselineMJX
                _avoid_baseline = AvoidBaselineMJX(_load_avoid_config())
                _avoid_baseline.max_iterations = MAX_STEPS
    return _avoid_baseline


def _warmup_avoid_solvers():
    """启动预热: 首请求的 10+s 耗时来自 JAX JIT 编译与模型加载,
    后台各跑一次全链路 (RL/混合/baseline) 把编译提前, 用户首请求即稳态 ~10ms"""
    import time
    t0 = time.perf_counter()
    try:
        tgt = [0.4, 0.0, 0.5]
        quat = _rpy_to_quaternion([0, 0, 0])
        bl = get_avoid_baseline()
        bl._apply(np.asarray(bl.home_position))
        specs = bl.spawn_scene(2, np.asarray(tgt), dist_range=(0.12, 0.35),
                               clearance=0.06)
        for getter in (get_avoid_rl_solver, get_avoid_rl_hybrid_solver):
            rl = getter()
            rl.solve(tgt, quat, obstacle_specs=specs,
                     start_q=rl.home_position, record_trajectory=True)
        bl.set_scene(tgt, quat, obstacle_specs=specs,
                     start_q=bl.home_position)
        bl.solve_dls()
        print(f'  [预热] 避障求解器 JIT 编译完成, 耗时 '
              f'{time.perf_counter()-t0:.1f}s')
    except Exception as e:
        print(f'  [预热] 失败 (不影响服务, 首请求会现场编译): {e}')


@app.route('/api/generate_obstacles', methods=['POST'])
def generate_obstacles():
    """走廊采样保护实际起点的几何包络，不保证存在可行连接路径。

    请求体: {"count": 1~3, "target_pos": [x,y,z], "start_q": 可选七关节角}
    返回:   {"obstacles": [spec...], "count": n, "start_q": 实际保护起点}
    """
    data = request.get_json(silent=True)
    try:
        if not isinstance(data, dict):
            raise ValueError('请求必须是 JSON 对象')
        count = data.get('count', 1)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 3:
            raise ValueError('count 必须是 1 至 3 的整数')
        scene = validate_scene(dict(
            data, target_pos=data.get('target_pos', [0.4, 0.0, 0.5])))
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify(error=str(exc)), 400
    try:
        baseline = get_avoid_baseline()
        start_q = np.asarray(
            scene.get('start_q', baseline.home_position), dtype=float)
        baseline._apply(start_q)
        specs = baseline.spawn_scene(count, np.asarray(scene['target_pos'], dtype=float),
                                     dist_range=(0.12, 0.35), clearance=0.06)
        return jsonify({'obstacles': specs, 'count': len(specs),
                        'start_q': start_q.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/solve_ik_avoid', methods=['POST'])
def solve_ik_avoid():
    """避障 IK 求解

    请求体:
    {
        "target_pos": [x,y,z],
        "target_rpy": [r,p,y] (度),
        "method": "rl_avoid" | "rl_avoid_dls" | "dls_avoid" | "nullspace",
        "obstacles": [spec...]   # generate_obstacles 返回的规格
    }
    返回: joint_angles, link_transforms, trajectory, success,
          position_error, orientation_error_deg, min_distance,
          n_collisions, iterations, solve_time_ms
    """
    data = request.json or {}
    try:
        data = validate_scene(data)
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify(error=str(exc)), 400
    target_pos = data.get('target_pos')
    if target_pos is None:
        return jsonify({'error': '请提供 target_pos [x, y, z]'}), 400
    target_quat_raw = data.get('target_quat')
    if target_quat_raw is None:
        rpy = data.get('target_rpy', [0, 0, 0])
        target_quat = _rpy_to_quaternion(rpy)
    else:
        target_quat = np.asarray(target_quat_raw, dtype=float)
        if target_quat.shape != (4,) or not np.all(np.isfinite(target_quat)):
            return jsonify({'error': 'target_quat 必须是 4 个有限数'}), 400
        target_quat = target_quat / (np.linalg.norm(target_quat) + 1e-12)
    method = data.get('method', 'rl_avoid')
    obstacles = data.get('obstacles') or []
    start_q_raw = data.get('start_q')
    start_q = None
    if start_q_raw is not None:
        start_q = np.asarray(start_q_raw, dtype=float)
        if start_q.shape != (7,) or not np.all(np.isfinite(start_q)):
            return jsonify({'error': 'start_q 必须是 7 个有限关节角'}), 400

    try:
        if method in ('rl_avoid', 'rl_avoid_dls'):
            rl = (get_avoid_rl_hybrid_solver() if method == 'rl_avoid_dls'
                  else get_avoid_rl_solver())
            started = time.perf_counter()
            result = rl.solve(target_pos, target_quat,
                              obstacle_specs=obstacles,
                              start_q=(rl.home_position if start_q is None
                                       else start_q),
                              record_trajectory=True)
            joint_angles = result['joint_angles']
            trajectory = result.get('trajectory')
            # 混合模式: DLS 精调段轨迹拼接到 RL 轨迹后, 动画完整播放全程
            refine_traj = result.pop('refine_trajectory', None) or []
            if trajectory is not None and refine_traj:
                trajectory = list(trajectory) + [np.asarray(q).copy()
                                                 for q in refine_traj]
        else:
            baseline = get_avoid_baseline()
            baseline.set_scene(target_pos, target_quat,
                               obstacle_specs=obstacles,
                               start_q=(baseline.home_position if start_q is None
                                        else start_q))
            started = time.perf_counter()
            if method == 'dls_avoid':
                result = baseline.solve_dls(start_q=start_q)
            elif method == 'nullspace':
                result = baseline.solve_nullspace(start_q=start_q)
            else:
                return jsonify({'error': f'未知方法: {method}'}), 400
            joint_angles = result['joint_angles']
            trajectory = result.pop('trajectory', None)

        # Certify returned samples, including hybrid refinement and final pose.
        trajectory = list(trajectory or [])
        if not trajectory or not np.allclose(trajectory[-1], joint_angles, atol=1e-8, rtol=0):
            trajectory.append(np.asarray(joint_angles))
        checker = rl if method.startswith('rl') else baseline
        result['trajectory_collisions'] = checker._trajectory_collision_count(trajectory)
        result['solve_time_ms'] = round((time.perf_counter() - started) * 1000, 3)
        result['solver_success'] = bool(result['success'])
        result['success'] = trajectory_safe_success(result)
        result['requested_method'] = method
        result['software_name'] = SOFTWARE_NAME
        result['model_id'] = str(MODEL.relative_to(_BASE_DIR)) if method.startswith('rl') else None
        result['max_steps'] = MAX_STEPS
        result['observation_max_steps'] = OBSERVATION_MAX_STEPS
        result['start_q'] = np.asarray(start_q if start_q is not None else checker.home_position).tolist()
        result['refine_max_steps'] = 100 if method == 'rl_avoid_dls' else 0
        if result.get('min_distance') is not None and not np.isfinite(result['min_distance']):
            result['min_distance'] = None

        # 用主 solver 获取 link_transforms 供前端渲染
        solver.joint_angles = joint_angles
        solver._apply_joint_angles()
        result['link_transforms'] = solver.get_all_link_transforms(joint_angles)
        # 末端位姿 (供前端末端坐标轴渲染)
        fk = solver.forward_kinematics()
        result['ee_position'] = fk['position']
        result['ee_orientation'] = fk['orientation']
        # 轨迹 numpy -> list (jsonify 序列化)
        if trajectory is not None:
            trajectory = [np.asarray(q, dtype=float).tolist()
                          for q in trajectory]
        result['trajectory'] = trajectory
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


def _validated_start_q(value):
    """Use the loaded Panda's actual limits for every scene-based API."""
    q = np.asarray(value, dtype=float)
    if q.shape != (solver.n_ctrl,) or not np.isfinite(q).all():
        raise ValueError('start_q 必须包含 7 个有限关节角（弧度）')
    limits = np.asarray(solver.joint_limits, dtype=float)
    if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
        raise ValueError('start_q 超出 Panda 关节限位；请提供限位内起点')
    return q.tolist()


def validate_scene(data):
    """Validate portable JSON scenes before passing them to numeric solvers."""
    if not isinstance(data, dict):
        raise ValueError('场景必须是 JSON 对象')
    def vector(value, size, label):
        a = np.asarray(value, dtype=float)
        if a.shape != (size,) or not np.isfinite(a).all():
            raise ValueError(f'{label} 必须包含 {size} 个有限数')
        return a.tolist()
    out = {'schema_version': 1, 'robot_id': SUPPORTED_ROBOT_ID,
           'target_pos': vector(data.get('target_pos'), 3, 'target_pos'),
           'method': data.get('method', 'rl_avoid')}
    if (data.get('schema_version', 1) != 1
            or data.get('robot_id', SUPPORTED_ROBOT_ID) != SUPPORTED_ROBOT_ID):
        raise ValueError('仅支持 schema_version=1 的 Panda 场景')
    if out['method'] not in METHODS:
        raise ValueError('未知求解方法')
    if data.get('target_quat') is not None:
        q = np.asarray(vector(data['target_quat'], 4, 'target_quat'))
        if np.linalg.norm(q) < 1e-9:
            raise ValueError('目标四元数不能为零')
        out['target_quat'] = (q / np.linalg.norm(q)).tolist()
    out['target_rpy'] = vector(data.get('target_rpy', [0, 0, 0]), 3, 'target_rpy')
    if data.get('start_q') is not None:
        out['start_q'] = _validated_start_q(data['start_q'])
    obstacles = data.get('obstacles') or []
    if not isinstance(obstacles, list) or len(obstacles) > 3:
        raise ValueError('障碍物必须为数组，最多3个')
    out['obstacles'] = []
    for item in obstacles:
        shape = item['shape']
        pos = vector(item['pos'], 3, 'obstacle.pos')
        size = item['size']
        if shape == 'box':
            dims = vector(size['halfExtents'], 3, 'halfExtents')
            clean = {'halfExtents': dims}
        elif shape in ('sphere', 'cylinder'):
            keys = ['radius', 'height'] if shape == 'cylinder' else ['radius']
            dims = [float(size[k]) for k in keys]
            clean = dict(zip(keys, dims))
        else:
            raise ValueError('仅支持 box/sphere/cylinder 障碍物')
        if not np.isfinite(dims).all() or min(dims) <= 0:
            raise ValueError('障碍物尺寸必须为有限正数')
        out['obstacles'].append(dict(shape=shape, pos=pos, size=clean))
    return out


@app.route('/api/validate_scene', methods=['POST'])
def validate_scene_api():
    try:
        return jsonify(validate_scene(request.get_json()))
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify(error=str(exc)), 400


@app.route('/api/check_collision', methods=['POST'])
def check_collision():
    """检测给定关节构型与障碍物的碰撞情况 (供普通 IK 方法结果检查)

    请求体:
    {
        "joint_angles": [7 个关节角],
        "obstacles": [spec...]   # generate_obstacles 返回的规格
    }
    返回: {"n_collisions": int, "min_distance": float|None}
    """
    data = request.json or {}
    joint_angles = data.get('joint_angles')
    if joint_angles is None:
        return jsonify({'error': '请提供 joint_angles'}), 400
    obstacles = data.get('obstacles') or []
    try:
        baseline = get_avoid_baseline()
        q = np.asarray(joint_angles, dtype=np.float32)
        if obstacles:
            # 注入障碍物张量后查点云 SDF 最小距离 (min_dist<=0 判碰撞)
            baseline.env.set_scene(0, [0.5, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0], q,
                                   obstacle_specs=obstacles)
            dists, _ = baseline._link_metrics(q)
            min_dist = float(dists.min())
            n_contacts = 1 if min_dist <= 0 else 0
        else:
            min_dist, n_contacts = None, 0
        return jsonify({'n_collisions': int(n_contacts),
                        'min_distance': min_dist})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _rpy_to_quaternion(rpy):
    """RPY (度) → 四元数 [x, y, z, w] (URDF ZYX 约定)"""
    r = np.radians(rpy[0])
    p = np.radians(rpy[1])
    y = np.radians(rpy[2])
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_val = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [float(x), float(y_val), float(z), float(w)]


# ==================== RL IK 接口 ====================

@app.route('/api/rl_status')
def rl_status():
    """查询 RL 模型状态"""
    return jsonify({
        'model_available': MODEL.is_file() and CONFIG.is_file(),
        'training': training_status['training'],
        'progress': training_status['progress'],
        'error': training_status['error'],
    })


@app.route('/api/train_rl', methods=['POST'])
def train_rl():
    """触发 RL 模型训练 (异步, MJX 向量化环境)"""
    return jsonify(error='V1.0 演示版不提供训练入口'), 403




if __name__ == '__main__':
    print("=" * 60)
    check_files()
    print(SOFTWARE_NAME)
    print(f"  预热后访问 http://127.0.0.1:{os.environ.get('IK_PORT', '5000')} 开始使用")
    print("=" * 60)
    with _request_lock:
        _warmup_avoid_solvers()
    app.run(host='127.0.0.1', port=int(os.environ.get('IK_PORT', '5000')), debug=False)
