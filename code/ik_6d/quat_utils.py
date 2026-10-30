"""numpy 四元数工具 (xyzw 约定, 无 PyBullet 依赖)

原定义于 rl_env_6d.py, 抽出为独立模块, 使推理/可视化链路
(mj_solver, avoid_solvers, rl_ik_solver_6d) 不必经 rl_env_6d
间接引入 pybullet。rl_env_6d 保持 re-export 兼容。
"""
import numpy as np


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
