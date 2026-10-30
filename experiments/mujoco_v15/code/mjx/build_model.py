"""MJCF 修正模型构建: panda_raw.xml -> mjx/assets/panda_fixed.xml

URDF 转换器将 link8/hand 固定关节链合并进 panda_link7 时, 丢失了两段固定
变换 (panda_joint8: z+0.107 平移; panda_hand_joint: z 轴 -45° 旋转),
导致 hand 网格与手指 body 的位姿错误。本脚本将其补回:

1. hand 网格 geom:  pos=(0,0,0.107), quat=rot_z(-45°)
2. 左右手指 body:   pos=(0,0,0.107+0.0584), 手指 geom 朝向再乘 rot_z(-45°)
3. ee site: 在 link7 上定义 hand 坐标系 (z+0.107, z-45°),
   CPU MuJoCo 采样阶段用 site_xpos/site_xmat 直接读 EE 位姿

用法: python mjx/build_model.py
"""
import os

import numpy as np
import mujoco

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_XML = os.path.join(_BASE, 'mjx', 'assets', 'panda_raw.xml')
OUT_XML = os.path.join(_BASE, 'mjx', 'assets', 'panda_fixed.xml')

HAND_Z = 0.107            # panda_joint8 平移
HAND_YAW = -np.pi / 4     # panda_hand_joint rpy z 分量
FINGER_Z = 0.0584         # finger_joint 相对 hand 的平移


def qmul_wxyz(a, b):
    """wxyz 四元数乘法"""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]


def rot_z_wxyz(angle):
    return [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]


def build():
    spec = mujoco.MjSpec.from_file(RAW_XML)
    q_hand = rot_z_wxyz(HAND_YAW)

    # ---------- 1. hand 网格几何修正 ----------
    hand_geom = None
    for g in spec.geoms:
        if g.meshname == 'hand':
            hand_geom = g
            break
    if hand_geom is None:
        raise RuntimeError('hand geom not found')
    hand_geom.pos = [0.0, 0.0, HAND_Z]
    hand_geom.quat = q_hand

    # ---------- 2. 手指 body/geom 修正 ----------
    # 右手指 geom 的 rot_z(180°) 已由转换器保留, 这里统一左乘 q_hand
    for name in ('panda_leftfinger', 'panda_rightfinger'):
        body = spec.body(name)
        if body is None:
            raise RuntimeError(f'{name} body not found')
        body.pos = [0.0, 0.0, HAND_Z + FINGER_Z]
        for g in body.geoms:
            g.quat = qmul_wxyz(q_hand, list(g.quat))

    # ---------- 3. EE site (hand 坐标系) ----------
    link7 = spec.body('panda_link7')
    site = link7.add_site()
    site.name = 'ee'
    site.pos = [0.0, 0.0, HAND_Z]
    site.quat = q_hand

    xml = spec.to_xml()
    with open(OUT_XML, 'w', encoding='utf-8') as f:
        f.write(xml)

    # 编译验证
    m = mujoco.MjModel.from_xml_path(OUT_XML)
    d = mujoco.MjData(m)
    d.qpos[:7] = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    mujoco.mj_forward(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'ee')
    print(f'构建成功: {OUT_XML}')
    print(f'  nq={m.nq} nbody={m.nbody} ngeom={m.ngeom}')
    print(f'  home 构型 ee site pos = {np.round(d.site_xpos[sid], 5)}'
          f' (PyBullet 基准: [0.30702 0. 0.59027])')


if __name__ == '__main__':
    build()
