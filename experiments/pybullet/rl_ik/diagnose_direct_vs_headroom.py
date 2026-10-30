"""Strict paired Direct-add vs Headroom-aware residual evaluation.

The script performs deterministic inference only.  It never calls ``learn``.
Direct-add is run first and closed; Headroom-aware is then run with the same
episode seeds and every requested reset-scene field is checked exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

# Required when SB3 deserializes the saved custom feature extractor.
from ik_6d.residual_pointnet_features import ResidualPointNetFeatureExtractor  # noqa: F401
from ik_6d.rl_env_6d_residual_headroom import HeadroomResidualObstacleEnv
from ik_6d.rl_env_6d_residual_obstacle import ResidualObstacleEnv


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "residual_obstacle" / "residual_final.zip"
OUTPUT_DIR = ROOT / "benchmarks" / "residual_obstacle"
SUMMARY_PATH = OUTPUT_DIR / "direct_vs_headroom_summary.json"
CASES_PATH = OUTPUT_DIR / "direct_vs_headroom_cases.json"
DEFAULT_EPISODES = 1_000
DEFAULT_BASE_SEED = 2_026_081_300
BOOTSTRAP_SAMPLES = 20_000
EPSILON = 1e-12


@dataclass(frozen=True)
class SceneSignature:
    state: np.ndarray
    target_position: np.ndarray
    target_orientation: np.ndarray
    obstacle_type: str
    obstacle_position: np.ndarray
    obstacle_size_key: str
    obstacle_size: np.ndarray
    point_cloud: np.ndarray


@dataclass(frozen=True)
class EpisodeResult:
    index: int
    seed: int
    final_position_mm: float
    final_orientation_deg: float
    had_collision: bool
    episode_min_distance_m: float
    steps: int

    @property
    def success_5(self) -> bool:
        return self.final_position_mm < 5.0 and self.final_orientation_deg < 10.0

    @property
    def success_10(self) -> bool:
        return self.final_position_mm < 10.0 and self.final_orientation_deg < 10.0

    @property
    def collision_free_5(self) -> bool:
        return self.success_5 and not self.had_collision

    @property
    def collision_free_10(self) -> bool:
        return self.success_10 and not self.had_collision


class ActionDiagnostics:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.raw: list[np.ndarray] = []
        self.clipped: list[np.ndarray] = []
        self.nominal: list[np.ndarray] = []
        self.headroom: list[np.ndarray] = []
        self.effective: list[np.ndarray] = []
        self.base: list[np.ndarray] = []
        self.preclip: list[np.ndarray] = []
        self.final: list[np.ndarray] = []

    def add(self, info: dict) -> None:
        base = np.asarray(info["base_action"], dtype=np.float32)
        final = np.asarray(info["final_action"], dtype=np.float32)
        raw = np.asarray(info["raw_residual_action"], dtype=np.float32)
        clipped = np.asarray(info["clipped_residual_action"], dtype=np.float32)
        if self.mode == "direct":
            nominal = np.asarray(info["scaled_residual"], dtype=np.float32)
            preclip = (base + nominal).astype(np.float32)
            effective = (final - base).astype(np.float32)
        else:
            nominal = np.asarray(info["nominal_scaled_residual"], dtype=np.float32)
            preclip = np.asarray(info["preclip_final_action"], dtype=np.float32)
            effective = np.asarray(info["effective_residual"], dtype=np.float32)
            self.headroom.append(np.asarray(info["headroom"], dtype=np.float32).copy())
        self.raw.append(raw.copy())
        self.clipped.append(clipped.copy())
        self.nominal.append(nominal.copy())
        self.effective.append(effective.copy())
        self.base.append(base.copy())
        self.preclip.append(preclip.copy())
        self.final.append(final.copy())

    @staticmethod
    def _distribution(values: np.ndarray) -> dict:
        if values.size == 0:
            return {"count": 0, "mean": None, "p05": None, "p50": None, "p95": None}
        return {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }

    def summary(self) -> dict:
        raw = np.asarray(self.raw, dtype=np.float64)
        clipped = np.asarray(self.clipped, dtype=np.float64)
        nominal = np.asarray(self.nominal, dtype=np.float64)
        effective = np.asarray(self.effective, dtype=np.float64)
        base = np.asarray(self.base, dtype=np.float64)
        preclip = np.asarray(self.preclip, dtype=np.float64)
        final = np.asarray(self.final, dtype=np.float64)
        numerical_clip = preclip != final
        nominal_sum = float(np.sum(np.abs(nominal)))
        effective_sum = float(np.sum(np.abs(effective)))
        result = {
            "steps": int(base.shape[0]),
            "actual_numerical_clipping_dimension_rate": float(np.mean(numerical_clip)),
            "actual_numerical_clipping_step_rate": float(np.mean(np.any(numerical_clip, axis=1))),
            "mean_abs_nominal_scaled_residual": float(np.mean(np.abs(nominal))),
            "mean_abs_effective_residual": float(np.mean(np.abs(effective))),
            "mean_effective_residual_l2": float(np.mean(np.linalg.norm(effective, axis=1))),
            "effective_abs_sum_over_nominal_abs_sum": (
                effective_sum / nominal_sum if nominal_sum > EPSILON else None
            ),
            "mean_abs_actual_final_minus_base": float(np.mean(np.abs(final - base))),
            "inward_effective_dimension_rate": float(np.mean(base * effective < 0.0)),
            "outward_effective_dimension_rate": float(np.mean(base * effective > 0.0)),
            "per_joint": [
                {
                    "joint": joint + 1,
                    "numerical_clipping_rate": float(np.mean(numerical_clip[:, joint])),
                    "mean_abs_nominal_scaled_residual": float(np.mean(np.abs(nominal[:, joint]))),
                    "mean_abs_effective_residual": float(np.mean(np.abs(effective[:, joint]))),
                    "mean_abs_actual_final_minus_base": float(np.mean(np.abs(final[:, joint] - base[:, joint]))),
                }
                for joint in range(base.shape[1])
            ],
        }
        if self.mode == "headroom":
            headroom = np.asarray(self.headroom, dtype=np.float64)
            positive = clipped >= 0.0
            negative = clipped < 0.0
            outward = base * clipped > 0.0
            inward = base * clipped < 0.0
            result["directional_headroom"] = {
                "overall": self._distribution(headroom.ravel()),
                "positive_residual": self._distribution(headroom[positive]),
                "negative_residual": self._distribution(headroom[negative]),
                "outward_relative_to_base": self._distribution(headroom[outward]),
                "inward_relative_to_base": self._distribution(headroom[inward]),
                "zero_headroom_dimension_rate": float(np.mean(headroom == 0.0)),
                "per_joint": [
                    {
                        "joint": joint + 1,
                        "mean": float(np.mean(headroom[:, joint])),
                        "positive_residual_mean": (
                            float(np.mean(headroom[positive[:, joint], joint]))
                            if np.any(positive[:, joint]) else None
                        ),
                        "negative_residual_mean": (
                            float(np.mean(headroom[negative[:, joint], joint]))
                            if np.any(negative[:, joint]) else None
                        ),
                        "zero_rate": float(np.mean(headroom[:, joint] == 0.0)),
                    }
                    for joint in range(headroom.shape[1])
                ],
            }
        return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_scene(env, observation: dict) -> SceneSignature:
    obstacle = env.obstacle_info
    if obstacle is None:
        raise RuntimeError("obstacle_info is missing after reset")
    obstacle_type = str(obstacle["type"])
    if obstacle_type == "sphere":
        size_key = "radius"
        size = np.asarray([obstacle["radius"]], dtype=np.float64)
    elif obstacle_type == "box":
        size_key = "half_extents"
        size = np.asarray(obstacle["half_extents"], dtype=np.float64)
    else:
        raise RuntimeError(f"Unknown obstacle type: {obstacle_type}")
    return SceneSignature(
        state=np.asarray(observation["state"], dtype=np.float32).copy(),
        target_position=np.asarray(env.target_pos, dtype=np.float32).copy(),
        target_orientation=np.asarray(env.target_quat, dtype=np.float32).copy(),
        obstacle_type=obstacle_type,
        obstacle_position=np.asarray(obstacle["position"], dtype=np.float64).copy(),
        obstacle_size_key=size_key,
        obstacle_size=size.copy(),
        point_cloud=np.asarray(observation["point_cloud"], dtype=np.float32).copy(),
    )


def assert_same_scene(expected: SceneSignature, actual: SceneSignature, index: int, seed: int) -> None:
    for field in ("obstacle_type", "obstacle_size_key"):
        if getattr(expected, field) != getattr(actual, field):
            raise RuntimeError(
                f"paired reset mismatch episode={index} seed={seed} field={field}"
            )
    for field in (
        "state", "target_position", "target_orientation", "obstacle_position",
        "obstacle_size", "point_cloud",
    ):
        left = getattr(expected, field)
        right = getattr(actual, field)
        if not np.array_equal(left, right):
            maximum = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
            raise RuntimeError(
                f"paired reset mismatch episode={index} seed={seed} "
                f"field={field} max_abs_difference={maximum}"
            )


def run_policy(mode: str, episodes: int, base_seed: int, expected_scenes=None):
    seed_everything(base_seed)
    env = (
        ResidualObstacleEnv(residual_scale=0.05, residual_penalty_weight=1e-4)
        if mode == "direct"
        else HeadroomResidualObstacleEnv(residual_scale=0.05, residual_penalty_weight=1e-4)
    )
    model = PPO.load(str(MODEL_PATH), device="cpu")
    model.policy.set_training_mode(False)
    scenes: list[SceneSignature] = []
    results: list[EpisodeResult] = []
    diagnostics = ActionDiagnostics(mode)
    try:
        for index in range(episodes):
            episode_seed = base_seed + index
            seed_everything(episode_seed)
            observation, _ = env.reset(seed=episode_seed)
            scene = capture_scene(env.unwrapped, observation)
            if expected_scenes is None:
                scenes.append(scene)
            else:
                assert_same_scene(expected_scenes[index], scene, index, episode_seed)

            episode_min = float(env.unwrapped._get_min_obstacle_distance())
            final_info = None
            steps = 0
            for _ in range(env.unwrapped.max_steps):
                with torch.inference_mode():
                    residual_action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(residual_action)
                diagnostics.add(info)
                final_info = info
                steps += 1
                episode_min = min(episode_min, float(info["min_obstacle_distance"]))
                if terminated or truncated:
                    break
            if final_info is None:
                raise RuntimeError(f"{mode} episode executed no steps")
            results.append(
                EpisodeResult(
                    index=index,
                    seed=episode_seed,
                    final_position_mm=float(final_info["pos_distance"]) * 1000.0,
                    final_orientation_deg=float(final_info["orient_distance_deg"]),
                    had_collision=bool(final_info.get("had_collision", False)),
                    episode_min_distance_m=episode_min,
                    steps=steps,
                )
            )
            if (index + 1) % 100 == 0:
                print(f"{mode}: {index + 1}/{episodes}", flush=True)
    finally:
        env.close()
    return scenes, results, diagnostics


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [center - half, center + half]


def paired_bootstrap_ci(headroom: np.ndarray, direct: np.ndarray, seed: int) -> list[float]:
    differences = headroom.astype(np.float64) - direct.astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_SAMPLES, 500):
        count = min(500, BOOTSTRAP_SAMPLES - start)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        means[start:start + count] = np.mean(differences[indices], axis=1)
    return [float(value) for value in np.percentile(means, [2.5, 97.5])]


def exact_mcnemar(a_to_b: int, b_to_a: int) -> dict:
    discordant = a_to_b + b_to_a
    if discordant == 0:
        return {"discordant_total": 0, "two_sided_exact_p": 1.0}
    smaller = min(a_to_b, b_to_a)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2.0 ** discordant)
    return {
        "discordant_total": discordant,
        "two_sided_exact_p": float(min(1.0, 2.0 * lower_tail)),
    }


def summarize(results: list[EpisodeResult]) -> dict:
    total = len(results)
    output: dict[str, object] = {"episodes": total}
    for name in ("success_5", "success_10", "had_collision", "collision_free_5", "collision_free_10"):
        values = np.asarray([bool(getattr(item, name)) for item in results])
        count = int(np.sum(values))
        output[name] = {
            "count": count,
            "rate": count / total,
            "wilson_95_ci": wilson_interval(count, total),
        }
    output.update(
        {
            "mean_final_position_error_mm": float(np.mean([x.final_position_mm for x in results])),
            "mean_final_orientation_error_deg": float(np.mean([x.final_orientation_deg for x in results])),
            "mean_episode_minimum_obstacle_distance_m": float(np.mean([x.episode_min_distance_m for x in results])),
            "mean_steps": float(np.mean([x.steps for x in results])),
        }
    )
    return output


def paired_analysis(direct: list[EpisodeResult], headroom: list[EpisodeResult]) -> dict:
    direct_collision = np.asarray([x.had_collision for x in direct], dtype=bool)
    headroom_collision = np.asarray([x.had_collision for x in headroom], dtype=bool)
    direct_cf10 = np.asarray([x.collision_free_10 for x in direct], dtype=bool)
    headroom_cf10 = np.asarray([x.collision_free_10 for x in headroom], dtype=bool)

    collision_rescued = int(np.sum(direct_collision & ~headroom_collision))
    collision_created = int(np.sum(~direct_collision & headroom_collision))
    cf10_gained = int(np.sum(~direct_cf10 & headroom_cf10))
    cf10_lost = int(np.sum(direct_cf10 & ~headroom_cf10))
    return {
        "collision": {
            "direct_collision_to_headroom_no_collision": collision_rescued,
            "direct_no_collision_to_headroom_collision": collision_created,
            "both_collision": int(np.sum(direct_collision & headroom_collision)),
            "both_no_collision": int(np.sum(~direct_collision & ~headroom_collision)),
            "difference_headroom_minus_direct": float(np.mean(headroom_collision) - np.mean(direct_collision)),
            "paired_bootstrap_95_ci": paired_bootstrap_ci(headroom_collision, direct_collision, 73_101),
            "exact_mcnemar": exact_mcnemar(collision_rescued, collision_created),
        },
        "collision_free_10mm": {
            "direct_fail_to_headroom_success": cf10_gained,
            "direct_success_to_headroom_fail": cf10_lost,
            "both_success": int(np.sum(direct_cf10 & headroom_cf10)),
            "both_fail": int(np.sum(~direct_cf10 & ~headroom_cf10)),
            "difference_headroom_minus_direct": float(np.mean(headroom_cf10) - np.mean(direct_cf10)),
            "paired_bootstrap_95_ci": paired_bootstrap_ci(headroom_cf10, direct_cf10, 73_102),
            "exact_mcnemar": exact_mcnemar(cf10_gained, cf10_lost),
        },
    }


def build_cases(direct: list[EpisodeResult], headroom: list[EpisodeResult]) -> dict:
    categories = {
        "direct_collision_headroom_no_collision": [],
        "direct_no_collision_headroom_collision": [],
        "direct_cf10_fail_headroom_success": [],
        "direct_cf10_success_headroom_fail": [],
    }
    for old, new in zip(direct, headroom):
        record = {"episode_index": old.index, "seed": old.seed, "direct": asdict(old), "headroom": asdict(new)}
        if old.had_collision and not new.had_collision:
            categories["direct_collision_headroom_no_collision"].append(record)
        if not old.had_collision and new.had_collision:
            categories["direct_no_collision_headroom_collision"].append(record)
        if not old.collision_free_10 and new.collision_free_10:
            categories["direct_cf10_fail_headroom_success"].append(record)
        if old.collision_free_10 and not new.collision_free_10:
            categories["direct_cf10_success_headroom_fail"].append(record)
    return {key: {"count": len(value), "first_20": value[:20]} for key, value in categories.items()}


def print_policy(name: str, item: dict) -> None:
    print(f"\n[{name}]")
    print(f"5mm+10deg IK:        {item['success_5']['rate'] * 100:.2f}%")
    print(f"10mm+10deg IK:       {item['success_10']['rate'] * 100:.2f}%")
    print(f"collision:           {item['had_collision']['rate'] * 100:.2f}%")
    print(f"collision-free 5mm:  {item['collision_free_5']['rate'] * 100:.2f}%")
    print(f"collision-free 10mm: {item['collision_free_10']['rate'] * 100:.2f}%")
    print(f"mean position:       {item['mean_final_position_error_mm']:.4f} mm")
    print(f"mean orientation:    {item['mean_final_orientation_error_deg']:.4f} deg")
    print(f"mean min distance:   {item['mean_episode_minimum_obstacle_distance_m']:.6f} m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    print("Phase 1/2: Direct-add", flush=True)
    scenes, direct, direct_actions = run_policy("direct", args.episodes, args.base_seed)
    print("Direct-add environment closed", flush=True)
    print("Phase 2/2: Headroom-aware with strict reset checks", flush=True)
    _, headroom, headroom_actions = run_policy("headroom", args.episodes, args.base_seed, scenes)
    print("All paired reset scenes matched exactly", flush=True)

    direct_action_summary = direct_actions.summary()
    headroom_action_summary = headroom_actions.summary()
    summary = {
        "episodes": args.episodes,
        "base_seed": args.base_seed,
        "residual_model": str(MODEL_PATH),
        "residual_scale": 0.05,
        "all_reset_scenes_exactly_equal": True,
        "scene_fields_compared": [
            "initial_state", "target_position", "target_orientation", "obstacle_type",
            "obstacle_position", "obstacle_size", "full_512x3_point_cloud",
        ],
        "direct_add": summarize(direct),
        "headroom_aware": summarize(headroom),
        "paired": paired_analysis(direct, headroom),
        "direct_action_diagnostics": direct_action_summary,
        "headroom_action_diagnostics": headroom_action_summary,
        "control_authority_comparison": {
            "mean_abs_effective_headroom_minus_direct": (
                headroom_action_summary["mean_abs_effective_residual"]
                - direct_action_summary["mean_abs_effective_residual"]
            ),
            "headroom_over_direct_mean_abs_effective": (
                headroom_action_summary["mean_abs_effective_residual"]
                / direct_action_summary["mean_abs_effective_residual"]
                if direct_action_summary["mean_abs_effective_residual"] > EPSILON else None
            ),
            "interpretation_note": (
                "Headroom intentionally remaps nominal residuals, so effective/nominal "
                "is descriptive and is not treated as the sole quality metric."
            ),
        },
        "statistics": {
            "paired_bootstrap_samples": BOOTSTRAP_SAMPLES,
            "mcnemar": "two-sided exact binomial test on discordant pairs",
            "binary_rate_intervals": "Wilson 95% intervals",
        },
        "no_training": True,
    }
    cases = build_cases(direct, headroom)

    print_policy("Direct-add", summary["direct_add"])
    print_policy("Headroom-aware", summary["headroom_aware"])
    print("\n[paired collision]")
    print(json.dumps(summary["paired"]["collision"], indent=2))
    print("\n[paired collision-free 10mm]")
    print(json.dumps(summary["paired"]["collision_free_10mm"], indent=2))
    print("\n[action diagnostic headline]")
    print(json.dumps({
        "direct": direct_action_summary,
        "headroom": headroom_action_summary,
        "comparison": summary["control_authority_comparison"],
    }, indent=2))

    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        CASES_PATH.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsummary saved: {SUMMARY_PATH}")
        print(f"cases saved: {CASES_PATH}")


if __name__ == "__main__":
    main()
