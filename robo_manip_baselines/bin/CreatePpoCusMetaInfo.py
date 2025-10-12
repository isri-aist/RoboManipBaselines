import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np


SECTION_ARRAY_FIELDS: Mapping[str, Iterable[str]] = {
    "state": ("example", "mean", "std", "min", "max", "range"),
    "action": ("example", "mean", "std", "min", "max", "range"),
}

SECTION_STRING_LIST_FIELDS: Mapping[str, Iterable[str]] = {
    "state": ("keys",),
    "action": ("keys",),
    "image": ("camera_names",),
}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a JSON object, got {type(data)}")
    return data


def _to_numpy(
    value: Any,
    *,
    field_name: str,
    section_name: str,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    try:
        return np.asarray(value, dtype=dtype)
    except Exception as exc:
        raise ValueError(
            f"Failed converting '{section_name}.{field_name}' to numpy array"
        ) from exc


def _convert_section(section_name: str, section_cfg: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    array_fields = set(SECTION_ARRAY_FIELDS.get(section_name, ()))
    string_list_fields = set(SECTION_STRING_LIST_FIELDS.get(section_name, ()))

    for key, value in section_cfg.items():
        if value is None:
            result[key] = None
            continue

        if key in string_list_fields:
            if not isinstance(value, list):
                raise TypeError(
                    f"'{section_name}.{key}' must be a list of strings, "
                    f"got {type(value)}"
                )
            result[key] = [str(v) for v in value]
            continue

        if key in array_fields:
            result[key] = _to_numpy(
                value, field_name=key, section_name=section_name
            )
            continue

        if isinstance(value, dict):
            nested_dict: Dict[str, Any] = {}
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, (list, tuple)):
                    nested_dict[nested_key] = _to_numpy(
                        nested_value,
                        field_name=f"{key}.{nested_key}",
                        section_name=section_name,
                    )
                else:
                    nested_dict[nested_key] = nested_value
            result[key] = nested_dict
            continue

        if isinstance(value, (list, tuple)):
            result[key] = _to_numpy(value, field_name=key, section_name=section_name)
            continue

        result[key] = value

    return result


def _validate_vector_lengths(section_name: str, section_cfg: Dict[str, Any]) -> None:
    lengths = {}
    for key, value in section_cfg.items():
        if isinstance(value, np.ndarray) and value.ndim == 1:
            lengths[key] = value.shape[0]

    if not lengths:
        return

    reference_length = next(iter(lengths.values()))
    mismatched = {
        key: length
        for key, length in lengths.items()
        if length != reference_length
    }
    if mismatched:
        mismatched_str = ", ".join(
            f"{key}={length}" for key, length in mismatched.items()
        )
        raise ValueError(
            f"Mismatched vector lengths in section '{section_name}'. "
            f"Expected {reference_length}, but got {mismatched_str}"
        )


def _build_meta_info(config: Dict[str, Any]) -> Dict[str, Any]:
    meta_info: Dict[str, Any] = {}
    for section_name, section_cfg in config.items():
        if not isinstance(section_cfg, dict):
            meta_info[section_name] = section_cfg
            continue

        converted_section = _convert_section(section_name, section_cfg)
        _validate_vector_lengths(section_name, converted_section)
        meta_info[section_name] = converted_section

    return meta_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create model_meta_info.pkl from a JSON config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON config file that describes meta info.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for the generated model_meta_info.pkl.",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = _load_json(args.config)
    meta_info = _build_meta_info(config)

    output_path = args.output
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Use --force to overwrite."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(meta_info, f)

    print(
        f"Saved model meta info to: {output_path.resolve()}\n"
        f"Sections: {', '.join(meta_info.keys())}"
    )


if __name__ == "__main__":
    main()
