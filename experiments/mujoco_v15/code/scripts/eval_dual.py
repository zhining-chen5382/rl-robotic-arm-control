"""最终评测: best 与 final 模型同场景集对比, 取优者

背景: 训练中 best 模型由 n=20 周期评测选出, 噪声 ±5pp, 可能选错;
最终评测同时评 best 与 final (n=1000), 以权威大样本结果为准。

两个模型使用完全相同的场景集 (同 seed 生成一次), 逐场景可比。
赢家模型复制为 <model_name>_mjx_champion/champion_model.zip。

用法:
    python scripts/eval_dual.py --config configs/avoid_mjx_v6.yaml \
        --best models/panda_ppo_ik_avoid_v6_mjx_best/best_model \
        --final models/panda_ppo_ik_avoid_v6_mjx --n 1000
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO

from ik_avoid.config import load_config
from ik_avoid.eval_stats import inject_scenes, rollout_eval, summarize
from mjx.env_mjx import RobotIKAvoidEnvMJX

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')


def eval_model(model_path, envs_scenes, cfg):
    """在同一批已注入场景的环境上评测, 返回 {组别: summary}"""
    print(f"\n  [加载] {model_path}.zip")
    model = PPO.load(model_path, device='cpu')
    out = {}
    for kind, (env, scenes) in envs_scenes.items():
        inject_scenes(env, scenes)  # 重新注入同一场景集 (复位到初始构型)
        results = rollout_eval(env, model, scenes, env.max_steps)
        out[kind] = summarize(results, kind)
    return out, model


def main(args):
    cfg = load_config(args.config)
    model_name = f'{cfg.train.model_name}_mjx'

    # [1] 构建环境与固定场景集 (两组各生成一次, 双模型共用)
    envs_scenes = {}
    for kind, with_obs in [('rl_obstacles', True), ('rl_free', False)]:
        if kind == 'rl_free' and not args.include_free:
            continue
        env = RobotIKAvoidEnvMJX(cfg=cfg, n_worlds=args.n,
                                 max_points=args.max_points)
        scenes = env.generate_scenes(args.n, args.seed,
                                     with_obstacles=with_obs)
        envs_scenes[kind] = (env, scenes)

    # [2] 双模型评测
    sum_best, _ = eval_model(args.best, envs_scenes, cfg)
    sum_final, _ = eval_model(args.final, envs_scenes, cfg)

    # [3] 对比表
    print(f"\n{'='*74}")
    print(f"  双模型对比 (n={args.n}, seed={args.seed}, 同场景集)")
    print(f"{'='*74}")
    header = (f"  {'组别':<12}{'指标':<12}{'best':>10}{'final':>10}{'差值':>10}")
    print(header)
    score = {'best': 0.0, 'final': 0.0}   # 综合分 = 各组严格成功率均值
    for kind in envs_scenes:
        sb, sf = sum_best[kind], sum_final[kind]
        for label, key, pct in [
                ('严格成功率', 'strict_success_rate', True),
                ('轨迹碰撞率', 'trajectory_collision_rate', True),
                ('位置中位mm', 'pos_err_mm_median', False),
                ('姿态中位°', 'orient_err_deg_median', False)]:
            vb, vf = sb[key], sf[key]
            db = vf - vb
            fb = f'{vb*100:9.1f}%' if pct else f'{vb:9.1f}'
            ff = f'{vf*100:9.1f}%' if pct else f'{vf:9.1f}'
            fd = f'{db*100:+9.1f}' if pct else f'{db:+9.1f}'
            print(f'  {kind:<12}{label:<12}{fb}{ff}{fd}')
        score['best'] += sb['strict_success_rate']
        score['final'] += sf['strict_success_rate']
    n_grp = len(envs_scenes)
    score['best'] /= n_grp
    score['final'] /= n_grp
    winner = 'best' if score['best'] >= score['final'] else 'final'
    print(f"{'-'*74}")
    print(f"  综合分(各组严格均值): best {score['best']*100:.1f}% | "
          f"final {score['final']*100:.1f}%  ->  赢家: {winner}")

    # [4] 赢家模型归档为 champion
    win_path = args.best if winner == 'best' else args.final
    champ_dir = os.path.join(BASE_DIR, 'models', f'{model_name}_champion')
    os.makedirs(champ_dir, exist_ok=True)
    champ = os.path.join(champ_dir, 'champion_model.zip')
    shutil.copy(win_path + '.zip', champ)
    # norm 文件与主模型同名后缀 _norm.pkl (位于 models/ 下)
    norm = os.path.join(BASE_DIR, 'models', f'{model_name}_norm.pkl')
    if os.path.exists(norm):
        shutil.copy(norm, os.path.join(champ_dir, 'champion_norm.pkl'))
    print(f"  赢家模型已归档: {champ}")

    # [5] 保存 JSON
    os.makedirs(LOG_DIR, exist_ok=True)
    t_stamp = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(LOG_DIR, f'eval_dual_{t_stamp}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'args': vars(args), 'score': score, 'winner': winner,
                   'best': sum_best, 'final': sum_final},
                  f, ensure_ascii=False, indent=1, default=float)
    print(f"  对比结果已保存: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='best/final 双模型对比评测')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--best', type=str, required=True,
                        help='best 模型路径 (不含 .zip)')
    parser.add_argument('--final', type=str, required=True,
                        help='final 模型路径 (不含 .zip)')
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--include-free', action='store_true')
    parser.add_argument('--max_points', type=int, default=768)
    args = parser.parse_args()
    main(args)
