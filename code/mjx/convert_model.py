"""URDF -> MJCF 转换脚本 (Panda)

用法: python mjx/convert_model.py
输出: mjx/assets/panda_raw.xml (含碰撞网格的 MJCF)

说明:
- urdf/panda.urdf 中网格路径为 package://meshes/..., 实际位于 urdf/meshes/
- 转换前将 package:// 替换为 urdf/ 使相对路径可解析
- 产物 panda_raw.xml 供 env_mjx.py 进一步加工 (加障碍物体/设置 margin)
"""
import os
import re
import sys

import mujoco

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = os.path.join(_BASE, 'urdf', 'panda.urdf')
OUT_PATH = os.path.join(_BASE, 'mjx', 'assets', 'panda_raw.xml')


def main():
    with open(URDF_PATH, 'r', encoding='utf-8') as f:
        urdf_text = f.read()
    # package://meshes/... -> urdf/meshes/... (相对项目根目录)
    urdf_text = re.sub(r'package://', 'urdf/', urdf_text)

    # 写入项目根目录的临时文件, 使相对网格路径可解析
    tmp_urdf = os.path.join(_BASE, '_panda_conv_tmp.urdf')
    with open(tmp_urdf, 'w', encoding='utf-8') as f:
        f.write(urdf_text)
    try:
        spec = mujoco.MjSpec.from_file(tmp_urdf)
        model = spec.compile()
        xml = spec.to_xml()
    finally:
        os.remove(tmp_urdf)

    # mesh 路径改为相对 mjx/assets/ 的写法 (../../urdf/meshes)
    xml = xml.replace('urdf/meshes', '../../urdf/meshes')
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(xml)
    print(f'已保存: {OUT_PATH} ({len(xml)} 字符)')

    print('nbody =', model.nbody, ' ngeom =', model.ngeom,
          ' njnt =', model.njnt, ' nq =', model.nq)
    print('--- bodies ---')
    for i in range(model.nbody):
        print(f'  body {i:2d}  {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)}')
    print('--- geoms (name, type, body) ---')
    for i in range(model.ngeom):
        print(f'  geom {i:2d}  {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)!r:20s}'
              f' type={int(model.geom_type[i])} body={int(model.geom_bodyid[i])}')
    print('--- joints ---')
    for i in range(model.njnt):
        print(f'  joint {i:2d}  {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)}'
              f'  qposadr={int(model.jnt_qposadr[i])}')


if __name__ == '__main__':
    sys.exit(main())
