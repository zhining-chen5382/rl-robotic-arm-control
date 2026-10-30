"""Four fixed scenes and API contract checks; not a benchmark or training run."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))
import app as web
from deployment import METHODS
from ik_avoid.eval_stats import trajectory_safe_success


def run(output_path=None):
    client = web.app.test_client()
    assert client.get("/").status_code == 200
    info = client.get("/api/software_info").get_json()
    assert info["max_steps"] == 150 and info["observation_max_steps"] == 100
    assert not info["training_enabled"]
    assert client.post("/api/train_rl", json={}).status_code == 403
    assert client.post("/api/switch_robot", json={"robot_id": "ur5"}).status_code == 400
    cases = json.loads((REPO / "code/static/demo_avoid_cases.json").read_text(encoding="utf-8"))[:3]
    scenes = [dict(cases[0], obstacles=[])] + cases
    records = []
    for index, scene in enumerate(scenes):
        restored = client.post("/api/validate_scene", json=scene)
        assert restored.status_code == 200, restored.data
        np.testing.assert_allclose(restored.json["start_q"], scene["start_q"])
        for method in METHODS:
            response = client.post("/api/solve_ik_avoid", json=dict(scene, method=method))
            assert response.status_code == 200, response.data
            result = response.get_json()
            assert result["success"] == trajectory_safe_success(result)
            np.testing.assert_allclose(result["trajectory"][0], scene["start_q"], atol=1e-6)
            np.testing.assert_allclose(result["trajectory"][-1], result["joint_angles"], atol=1e-6)
            assert np.isfinite(result["solve_time_ms"]) and result["solve_time_ms"] >= 0
            assert np.isfinite(result["position_error"])
            assert np.isfinite(result["orientation_error_deg"])
            records.append({"scene_index": index, "method": method, "result": result})
            print(f"case={index} method={method} valid=True success={result['success']}", flush=True)
    invalid = [dict(scenes[0], target_pos=[1, 2]),
               dict(scenes[0], target_quat=[0, 0, 0, 0]),
               dict(scenes[0], method="missing"),
               dict(scenes[0], start_q=[100] * 7),
               dict(scenes[0], start_q=[float("nan")] * 7),
               dict(scenes[0], obstacles=[{}]),
               dict(scenes[0], obstacles=[{"shape": "sphere", "pos": [0, 0, 0], "size": {"radius": -1}}])]
    for scene in invalid:
        assert client.post("/api/solve_ik_avoid", json=scene).status_code == 400
        assert client.post("/api/validate_scene", json=scene).status_code == 400
    generated = client.post("/api/generate_obstacles", json={"count": 2, "target_pos": cases[0]["target_pos"], "start_q": cases[0]["start_q"]})
    assert generated.status_code == 200, generated.data
    np.testing.assert_allclose(generated.json["start_q"], cases[0]["start_q"])
    assert client.post("/api/generate_obstacles", json={"count": 2, "start_q": [100] * 7}).status_code == 400
    assert client.post("/api/generate_obstacles", json={"count": "invalid"}).status_code == 400

    def request_one(method):
        with web.app.test_client() as parallel_client:
            response = parallel_client.post("/api/solve_ik_avoid", json=dict(cases[0], method=method))
            assert response.status_code == 200
            return response.get_json()

    with ThreadPoolExecutor(2) as pool:
        concurrent = list(pool.map(request_one, ["dls_avoid", "nullspace"]))
    for method, result in zip(["dls_avoid", "nullspace"], concurrent):
        reference = next(row["result"] for row in records if row["scene_index"] == 1 and row["method"] == method)
        np.testing.assert_allclose(result["joint_angles"], reference["joint_angles"], atol=1e-7)
    report = {"scope": "4 fixed scenes x 4 methods; software integration checks, not a success-rate or speed benchmark",
              "passed": True, "deployment": info, "method_calls": len(records),
              "invalid_scene_checks": len(invalid), "concurrency_checked": True, "records": records}
    if output_path is not None:
        Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print("PASS: candidate API integration checks", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional report path; no implicit report file is created.")
    run(parser.parse_args().output)
