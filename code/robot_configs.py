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
