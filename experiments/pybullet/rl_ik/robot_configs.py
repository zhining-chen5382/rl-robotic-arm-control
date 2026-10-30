"""
机械臂配置文件
定义所有支持的机械臂及其参数
"""
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROBOT_CONFIGS = {
    'panda': {
        'name': 'Franka Panda',
        'description': '7轴冗余协作机械臂 (Franka Emika)',
        'urdf_path': os.path.join(BASE_DIR, 'urdf', 'panda.urdf'),
        'num_axes': 7,
        'home_position': [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        # mesh_dir 用于前端加载3D模型
        'mesh_dir': '/urdf/meshes',
        # 前端link定义: name -> mesh文件路径 + 颜色
        'link_meshes': [
            {'name': 'panda_link0', 'mesh': '/urdf/meshes/collision/link0.obj', 'color': 0xbbbbbb, 'format': 'obj'},
            {'name': 'panda_link1', 'mesh': '/urdf/meshes/visual/link1.obj', 'color': 0xf0f0f0, 'format': 'obj'},
            {'name': 'panda_link2', 'mesh': '/urdf/meshes/visual/link2.obj', 'color': 0xf0f0f0, 'format': 'obj'},
            {'name': 'panda_link3', 'mesh': '/urdf/meshes/visual/link3.obj', 'color': 0xf0d0d0, 'format': 'obj'},
            {'name': 'panda_link4', 'mesh': '/urdf/meshes/visual/link4.obj', 'color': 0xf0f0f0, 'format': 'obj'},
            {'name': 'panda_link5', 'mesh': '/urdf/meshes/visual/link5.obj', 'color': 0xf0f0f0, 'format': 'obj'},
            {'name': 'panda_link6', 'mesh': '/urdf/meshes/visual/link6.obj', 'color': 0xf0f0f0, 'format': 'obj'},
            {'name': 'panda_link7', 'mesh': '/urdf/meshes/collision/link7.obj', 'color': 0xf0f0f0, 'format': 'obj'},
            {'name': 'panda_hand',  'mesh': '/urdf/meshes/visual/hand.obj',  'color': 0xdddddd, 'format': 'obj'},
        ],
        # 可达空间边界 (pybullet坐标系)
        'reach_bounds': {
            'x_min': -1.0, 'x_max': 1.0,
            'y_min': -1.0, 'y_max': 1.0,
            'z_min': 0.0, 'z_max': 1.2,
        },
        # 预设目标点
        'presets': [
            {'name': '前方', 'pos': [0.5, 0.0, 0.5]},
            {'name': '右上方', 'pos': [0.3, 0.3, 0.7]},
            {'name': '右下方', 'pos': [0.3, -0.3, 0.3]},
            {'name': '左侧', 'pos': [0.0, 0.4, 0.6]},
        ],
    },

    'ur5': {
        'name': 'Universal Robots UR5',
        'description': '6轴工业协作机械臂 (Universal Robots)',
        'urdf_path': os.path.join(BASE_DIR, 'urdf', 'ur5.urdf'),
        'num_axes': 6,
        'home_position': [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
        # UR5 STL mesh文件
        'mesh_dir': '/urdf/ur5/meshes',
        'link_meshes': [
            {'name': 'base_link_inertia', 'mesh': '/urdf/ur5/meshes/visual/base.stl', 'color': 0x666666, 'format': 'stl'},
            {'name': 'shoulder_link',     'mesh': '/urdf/ur5/meshes/visual/shoulder.stl', 'color': 0x3366aa, 'format': 'stl'},
            {'name': 'upper_arm_link',    'mesh': '/urdf/ur5/meshes/visual/upperarm.stl', 'color': 0x3366aa, 'format': 'stl'},
            {'name': 'forearm_link',      'mesh': '/urdf/ur5/meshes/visual/forearm.stl', 'color': 0x3366aa, 'format': 'stl'},
            {'name': 'wrist_1_link',      'mesh': '/urdf/ur5/meshes/visual/wrist1.stl', 'color': 0x4477bb, 'format': 'stl'},
            {'name': 'wrist_2_link',      'mesh': '/urdf/ur5/meshes/visual/wrist2.stl', 'color': 0x4477bb, 'format': 'stl'},
            {'name': 'wrist_3_link',      'mesh': '/urdf/ur5/meshes/visual/wrist3.stl', 'color': 0x4477bb, 'format': 'stl'},
        ],
        'reach_bounds': {
            'x_min': -0.9, 'x_max': 0.9,
            'y_min': -0.9, 'y_max': 0.9,
            'z_min': -0.2, 'z_max': 1.0,
        },
        'presets': [
            {'name': '前方', 'pos': [0.5, 0.0, 0.4]},
            {'name': '右上方', 'pos': [0.3, 0.3, 0.6]},
            {'name': '左侧', 'pos': [-0.2, 0.4, 0.3]},
            {'name': '下方', 'pos': [0.3, 0.0, 0.1]},
        ],
    },
}


def get_robot_list():
    """返回可用机械臂列表"""
    result = []
    for key, cfg in ROBOT_CONFIGS.items():
        result.append({
            'id': key,
            'name': cfg['name'],
            'description': cfg['description'],
            'num_axes': cfg['num_axes'],
        })
    return result


def get_robot_config(robot_id):
    """获取指定机械臂的配置"""
    if robot_id not in ROBOT_CONFIGS:
        raise ValueError(f"Unknown robot: {robot_id}. Available: {list(ROBOT_CONFIGS.keys())}")
    return ROBOT_CONFIGS[robot_id]
