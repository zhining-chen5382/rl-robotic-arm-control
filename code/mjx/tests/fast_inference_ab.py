"""fast_inference 验收: 与 jax 链路同场景 A/B (一致性 + 统计等效 + 计时)

用法: CUDA_VISIBLE_DEVICES='' python -u mjx/tests/fast_inference_ab.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from ik_avoid.config import load_config
from mjx.env_mjx import RobotIKAvoidEnvMJX
from mjx.avoid_solvers import RLAvoidSolverMJX
from mjx.fast_inference import FastRLAvoidSolverMJX

CFG = 'configs/avoid_mjx_v16.yaml'
MP = 'models/panda_ppo_ik_avoid_v16_mjx_champion/champion_model'
N_OBS_CASES = 60     # 有障碍案数
N_FREE_CASES = 20    # 无障碍案数


def run(solver, scenes):
    out, ts = [], []
    for sc in scenes:
        kw = dict(obstacle_specs=sc['obstacle_specs'],
                  start_q=np.asarray(sc['start_q'], np.float32))
        t0 = time.perf_counter()
        r = solver.solve(np.asarray(sc['target_pos'], np.float32),
                         np.asarray(sc['target_quat'], np.float32), **kw)
        ts.append(time.perf_counter() - t0)
        out.append(r)
    return out, ts


def summarize(tag, results, ts):
    succ = sum(r['success'] for r in results)
    coll = sum(r['trajectory_collisions'] for r in results)
    n = len(results)
    print(f'{tag}: 成功 {succ}/{n} ({100*succ/n:.1f}%) | '
          f'轨迹碰撞案 {sum(1 for r in results if r["trajectory_collisions"]>0)} '
          f'(帧 {coll}) | 均值 {1000*np.mean(ts):.1f}ms 中位 {1000*np.median(ts):.1f}ms')
    return succ, coll


def main():
    cfg = load_config(CFG)
    gen = RobotIKAvoidEnvMJX(cfg=cfg, n_worlds=N_OBS_CASES + N_FREE_CASES)
    scenes_obs = gen.generate_scenes(N_OBS_CASES, 2026, with_obstacles=True)
    scenes_free = gen.generate_scenes(N_FREE_CASES, 2027, with_obstacles=False)
    scenes = scenes_obs + scenes_free

    s_jax = RLAvoidSolverMJX(cfg, model_path=MP)
    s_fast = FastRLAvoidSolverMJX(cfg, model_path=MP)

    # 预热 (两侧 JIT / numpy 无 JIT 但首案含杂项初始化)
    for sc in scenes[:2]:
        kw = dict(obstacle_specs=sc['obstacle_specs'],
                  start_q=np.asarray(sc['start_q'], np.float32))
        for s in (s_jax, s_fast):
            s.solve(np.asarray(sc['target_pos'], np.float32),
                    np.asarray(sc['target_quat'], np.float32), **kw)
    cases = scenes[2:]

    print('=== 纯 RL (n=%d, 有障碍 %d + 无障碍 %d) ==='
          % (len(cases), len(scenes_obs) - 2, N_FREE_CASES))
    r_j, t_j = run(s_jax, cases)
    r_f, t_f = run(s_fast, cases)

    # ① 碰撞帧数逐案一致性 (包络球粗筛不漏判的必要验证)
    mismatch = [(i, r_j[i]['trajectory_collisions'],
                 r_f[i]['trajectory_collisions'])
                for i in range(len(cases))
                if r_j[i]['trajectory_collisions'] != r_f[i]['trajectory_collisions']]
    print(f'碰撞帧数逐案一致: {"YES" if not mismatch else "NO " + str(mismatch)}')

    # ② 成功率统计等效
    sj, cj = summarize('jax ', r_j, t_j)
    sf, cf = summarize('fast', r_f, t_f)
    print(f'提速: {np.mean(t_j)/np.mean(t_f):.2f}x '
          f'({1000*np.mean(t_j):.0f}ms -> {1000*np.mean(t_f):.0f}ms)')

    # ③ hybrid (DLS 精调) 计时
    print('=== hybrid (DLS 精调, 有障碍前 20 案) ===')
    s_jax_h = RLAvoidSolverMJX(cfg, model_path=MP, use_dls_refine=True)
    s_fast_h = FastRLAvoidSolverMJX(cfg, model_path=MP, use_dls_refine=True)
    sub = scenes_obs[2:22]
    for s in (s_jax_h, s_fast_h):
        sc = sub[0]
        s.solve(np.asarray(sc['target_pos'], np.float32),
                np.asarray(sc['target_quat'], np.float32),
                obstacle_specs=sc['obstacle_specs'],
                start_q=np.asarray(sc['start_q'], np.float32))
    r_jh, t_jh = run(s_jax_h, sub)
    r_fh, t_fh = run(s_fast_h, sub)
    summarize('jax-hybrid ', r_jh, t_jh)
    summarize('fast-hybrid', r_fh, t_fh)
    print(f'hybrid 提速: {np.mean(t_jh)/np.mean(t_fh):.2f}x')


if __name__ == '__main__':
    main()
