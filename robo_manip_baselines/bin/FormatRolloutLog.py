import argparse
import csv
import json
import os
from typing import List


def _load_rows(path: str):
    records = []
    max_obs = 0
    max_cmd = 0
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            step_idx = int(row.get("step_idx", 0))
            obs_values: List[float] = json.loads(row.get("obs", "[]"))
            cmd_values: List[float] = json.loads(row.get("direct_joint_command", "[]"))
            max_obs = max(max_obs, len(obs_values))
            max_cmd = max(max_cmd, len(cmd_values))
            records.append((step_idx, obs_values, cmd_values))
    return records, max_obs, max_cmd


def _write_rows(path: str, records, max_obs: int, max_cmd: int):
    header = ["step_idx"]
    header += [f"obs_{idx}" for idx in range(max_obs)]
    header += [f"direct_joint_command_{idx}" for idx in range(max_cmd)]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for step_idx, obs_values, cmd_values in records:
            row = [step_idx]
            row += obs_values + [""] * (max_obs - len(obs_values))
            row += cmd_values + [""] * (max_cmd - len(cmd_values))
            writer.writerow(row)


def convert_log(input_path: str, output_path: str):
    records, max_obs, max_cmd = _load_rows(input_path)
    if not records:
        raise ValueError(f"No rows found in {input_path}")
    _write_rows(output_path, records, max_obs, max_cmd)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Expand rollout_debug_log TSV into separate observation/action columns and save as CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join("robo_manip_baselines", "rollout_debug_log.tsv"),
        help="Path to the original TSV log produced during rollout.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("robo_manip_baselines", "rollout_debug_log_expanded.csv"),
        help="Path to save the expanded CSV with split columns.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    convert_log(args.input, args.output)


if __name__ == "__main__":
    main()
