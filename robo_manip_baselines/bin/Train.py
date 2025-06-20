import argparse
import importlib
import re
import sys

import wandb


def camel_to_snake(name):
    """Converts camelCase or PascalCase to snake_case (also converts the first letter to lowercase)"""
    name = re.sub(
        r"([a-z0-9])([A-Z])", r"\1_\2", name
    )  # Insert '_' between a lowercase/number and an uppercase letter
    name = re.sub(
        r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name
    )  # Insert '_' between consecutive uppercase letters followed by a lowercase letter
    name = name[0].lower() + name[1:]  # Convert the first letter to lowercase
    return name.lower()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This is a meta argument parser for the train switching between different policies and environments. The actual arguments are handled by another internal argument parser.",
        fromfile_prefix_chars="@",
        add_help=False,
    )
    parser.add_argument(
        "policy",
        type=str,
        nargs="?",
        default=None,
        choices=["Mlp", "Sarnn", "Act", "MtAct", "DiffusionPolicy"],
        help="policy",
    )
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show this help message and continue"
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Run sweep instead of normal training"
    )
    parser.add_argument(
        "--sweep_count", type=int, default=10, help="Number of sweep runs"
    )

    args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv
    if args.policy is None:
        parser.print_help()
        return
    elif args.help:
        parser.print_help()
        print("\n================================\n")
        sys.argv += ["--help"]

    policy_module = importlib.import_module(
        f"robo_manip_baselines.policy.{camel_to_snake(args.policy)}"
    )
    TrainPolicyClass = getattr(policy_module, f"Train{args.policy}")

    if args.sweep:
        print(f"[INFO] Running sweep for policy {args.policy}")
        sweep_config = TrainPolicyClass.get_sweep_config()
        sweep_train_fn = TrainPolicyClass.sweep_entrypoint()

        sweep_id = wandb.sweep(sweep_config, project="robomanip-act")
        wandb.agent(sweep_id, function=sweep_train_fn, count=args.sweep_count)

    else:
        print(f"[INFO] Running normal training for policy {args.policy}")
        train = TrainPolicyClass()
        train.run()
        train.close()


if __name__ == "__main__":
    main()
