"""Single source of truth for the V1.0 demonstration deployment."""
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOFTWARE_NAME = '七自由度机械臂避障逆运动学仿真与性能评估软件 V1.0'
MODEL = BASE / 'models/panda_ppo_ik_avoid_v30_risk_head_ft_mjx_best/best_model.zip'
CONFIG = BASE / 'configs/avoid_mjx_v30_risk_head_ft.yaml'
MAX_STEPS = 150
# Preserve the observation scale on which this existing model was trained.
OBSERVATION_MAX_STEPS = 100
METHODS = ('rl_avoid', 'rl_avoid_dls', 'dls_avoid', 'nullspace')

def check_files():
    for path in (MODEL, CONFIG, BASE / 'static/demo_avoid_cases.json'):
        if not path.is_file():
            raise FileNotFoundError(f'缺少发布文件: {path}')
    return str(MODEL), str(CONFIG)

if __name__ == '__main__':
    check_files()
    import flask, mujoco, jax, torch, gymnasium, yaml, trimesh, scipy
    print(SOFTWARE_NAME)
    print(f'Model: {MODEL}\nConfig: {CONFIG}\nMax steps: {MAX_STEPS}')
