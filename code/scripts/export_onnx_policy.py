"""无 SB3 依赖地导出策略 ONNX FP32/INT8。

直接复用 ``ik_avoid.policy_io.load_policy_from_zip`` 解析历史兼容 zip，
因此旧 SB3 模型与自研 GpuPPO 导出的 zip 都可使用。

示例:
    python scripts/export_onnx_policy.py \
        --model models/panda_ppo_ik_avoid_v16_mjx_champion/champion_model
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ik_avoid.policy_io import load_policy_from_zip


class ExportPolicy(torch.nn.Module):
    def __init__(self, ops):
        super().__init__()
        layers = []
        for op in ops:
            if op[0] == 'lin':
                weight, bias = op[1], op[2]
                layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
                with torch.no_grad():
                    layer.weight.copy_(torch.from_numpy(weight))
                    layer.bias.copy_(torch.from_numpy(bias))
                layers.append(layer)
            elif op[0] == 'tanh':
                layers.append(torch.nn.Tanh())
            else:
                layers.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


def export(model_path, obs_dim, verify_n=256, seed=2026):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    ops, lo, hi, risk_head = load_policy_from_zip(model_path)
    if risk_head is not None:
        raise NotImplementedError(
            '独立风险步长头暂不支持 ONNX 导出，请使用 numpy 后端')
    model = ExportPolicy(ops).eval()
    stem = model_path[:-4] if model_path.lower().endswith('.zip') else model_path
    fp32_path = stem + '_policy_fp32.onnx'
    int8_path = stem + '_policy_int8.onnx'
    dummy = torch.zeros((1, obs_dim), dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, fp32_path,
            input_names=['obs'], output_names=['action'],
            dynamic_axes={'obs': {0: 'batch'}, 'action': {0: 'batch'}},
            opset_version=13, dynamo=False)
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)

    # 随机输入逐位验收; 完整闭环等效由 fast_policy_backend_ab.py 负责。
    from ik_avoid.policy_io import NumpyPolicy, ONNXPolicy
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(verify_n, obs_dim)).astype(np.float32)
    ref = NumpyPolicy(ops, lo, hi).forward_batch(obs)
    fp32 = ONNXPolicy(fp32_path, action_lo=lo,
                      action_hi=hi).forward_batch(obs)
    int8 = ONNXPolicy(int8_path, action_lo=lo,
                      action_hi=hi).forward_batch(obs)
    d32 = np.abs(fp32 - ref)
    d8 = np.abs(int8 - ref)
    print(f'FP32: {fp32_path}')
    print(f'INT8: {int8_path}')
    print(f'FP32 vs numpy: max={d32.max():.3e} mean={d32.mean():.3e}')
    print(f'INT8 vs numpy: max={d8.max():.3e} mean={d8.mean():.3e}')
    return fp32_path, int8_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='导出策略 ONNX FP32/INT8')
    parser.add_argument('--model', required=True, help='模型路径，可省略 .zip')
    parser.add_argument('--obs-dim', type=int, default=56)
    parser.add_argument('--verify-n', type=int, default=256)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    export(args.model, args.obs_dim, args.verify_n, args.seed)
