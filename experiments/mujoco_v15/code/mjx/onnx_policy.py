"""ONNX Runtime 策略包装 (SB3 兼容接口)

提供与 SB3 PPO 相同的 .predict(obs, deterministic=True) 接口, 使求解器 /
评测脚本无需改动即可切换推理后端 (SB3 <-> ONNX fp32 <-> ONNX INT8)。

用法:
    from mjx.onnx_policy import ONNXPolicy
    policy = ONNXPolicy('models/xxx_policy_int8.onnx')
    action, _ = policy.predict(obs, deterministic=True)
"""
import numpy as np
import onnxruntime as ort


class ONNXPolicy:
    """ONNX Runtime 推理后端, .predict 签名对齐 SB3

    注意: 与 SB3 model.predict 一致, 输出会被裁剪到动作空间边界
    (本 env 为 [-1,1])——策略网络输出的是无界高斯均值, 不裁剪会与
    SB3 推理路径不一致。env.step 内部也会再裁剪一次, 双重保险。
    """

    def __init__(self, onnx_path, intra_threads=1, inter_threads=1,
                 clip_low=-1.0, clip_high=1.0):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_threads
        opts.inter_op_num_threads = inter_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            onnx_path, sess_options=opts,
            providers=['CPUExecutionProvider'])
        self.in_name = self.sess.get_inputs()[0].name
        self.onnx_path = onnx_path
        self.clip_low = clip_low
        self.clip_high = clip_high

    def predict(self, obs, deterministic=True):
        obs = np.asarray(obs, dtype=np.float32)
        single = obs.ndim == 1
        obs_in = obs[None, :] if single else obs
        out = self.sess.run(None, {self.in_name: obs_in})[0]
        out = np.clip(out, self.clip_low, self.clip_high)
        return (out[0], None) if single else (out, None)
