"""Export an existing GpuPPO .pt checkpoint to inference-compatible zip."""
import argparse
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from ik_avoid.config import load_config  # noqa: E402
from ik_avoid.ppo_gpu import GpuPPO  # noqa: E402


def main(args):
    cfg = load_config(args.config)
    learner = GpuPPO(
        obs_dim=args.obs_dim,
        act_dim=7 + int(cfg.env.learned_step_scale),
        arch=tuple(cfg.train.net_arch),
        device=args.device,
        n_steps=1, n_envs=1, batch_size=1, n_epochs=1,
        constrained=cfg.train.constrained_ppo)
    learner.load(args.checkpoint)
    output = args.output or os.path.splitext(args.checkpoint)[0]
    learner.export_zip(output)
    if not output.endswith('.zip'):
        output += '.zip'
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='')
    parser.add_argument('--obs-dim', type=int, default=56)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    main(parser.parse_args())
