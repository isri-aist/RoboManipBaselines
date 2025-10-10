import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _load_meta_info(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Meta info file not found: {path}")
    with path.open("rb") as f:
        meta = pickle.load(f)
    if not isinstance(meta, dict):
        raise TypeError(f"Unexpected meta info format: {type(meta)}, expected dict.")
    return meta


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _describe_array(name: str, array: np.ndarray) -> str:
    summary = f"{name}: shape={array.shape}, dtype={array.dtype}"
    if array.size == 0:
        return summary + ", empty"
    flattened = array.ravel()
    preview = ", ".join(f"{v:.4f}" if np.issubdtype(array.dtype, np.floating) else str(v) for v in flattened[:6])
    if flattened.size > 6:
        preview += ", ..."
    return f"{summary}, preview=[{preview}]"


def _print_section(title: str, data: Any, indent: int = 0) -> None:
    prefix = "  " * indent
    if isinstance(data, dict):
        print(f"{prefix}{title}:")
        for key, value in data.items():
            _print_section(str(key), value, indent + 1)
    elif isinstance(data, np.ndarray):
        print(f"{prefix}{_describe_array(title, data)}")
    elif isinstance(data, (list, tuple)):
        print(f"{prefix}{title}: list(len={len(data)})")
        for idx, item in enumerate(data[:5]):
            _print_section(f"[{idx}]", item, indent + 1)
        if len(data) > 5:
            print(f"{prefix}  ... ({len(data) - 5} more items)")
    else:
        print(f"{prefix}{title}: {data}")


def _summarize(meta: Dict[str, Any]) -> None:
    state = meta.get("state", {})
    action = meta.get("action", {})
    data = meta.get("data", {})
    image = meta.get("image", {})
    policy = meta.get("policy", {})

    print("=== Model Meta Info Summary ===")

    if state:
        print("\n[State]")
        print(f"  keys: {state.get('keys')}")
        example = state.get("example")
        if isinstance(example, np.ndarray):
            print(f"  example: shape={example.shape}, dtype={example.dtype}")
        mean = state.get("mean")
        if isinstance(mean, np.ndarray):
            print(f"  mean: shape={mean.shape}, dtype={mean.dtype}")
        std = state.get("std")
        if isinstance(std, np.ndarray):
            print(f"  std: shape={std.shape}, dtype={std.dtype}")

    if action:
        print("\n[Action]")
        print(f"  keys: {action.get('keys')}")
        example = action.get("example")
        if isinstance(example, np.ndarray):
            print(f"  example: shape={example.shape}, dtype={example.dtype}")
        mean = action.get("mean")
        if isinstance(mean, np.ndarray):
            print(f"  mean: shape={mean.shape}, dtype={mean.dtype}")
        std = action.get("std")
        if isinstance(std, np.ndarray):
            print(f"  std: shape={std.shape}, dtype={std.dtype}")

    if image:
        print("\n[Image]")
        for key, value in image.items():
            print(f"  {key}: {value}")

    if data:
        print("\n[Data]")
        for key, value in data.items():
            print(f"  {key}: {value}")

    if policy:
        print("\n[Policy]")
        for key, value in policy.items():
            print(f"  {key}: {value}")

    remaining = {k: v for k, v in meta.items() if k not in {"state", "action", "image", "data", "policy"}}
    if remaining:
        print("\n[Other]")
        for key, value in remaining.items():
            _print_section(str(key), value, indent=1)

    print("\n=== End Summary ===")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize contents of model_meta_info.pkl files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", type=Path, help="Path to model_meta_info.pkl")
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save the meta info as JSON with NumPy arrays converted to lists.",
    )
    parser.add_argument(
        "--full",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print full nested structure instead of summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = _load_meta_info(args.path)
    print(f"Loaded meta info from: {args.path.resolve()}")

    if args.full:
        _print_section("meta_info", meta)
    else:
        _summarize(meta)

    if args.save_json is not None:
        serializable = _to_serializable(meta)
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON representation to: {args.save_json.resolve()}")


if __name__ == "__main__":
    main()

