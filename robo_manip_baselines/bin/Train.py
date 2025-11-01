import argparse
import importlib
import sys


class TrainMain:
    policy_parent_module_str = "robo_manip_baselines.policy"
    policy_choices = [
        "Mlp",
        "Sarnn",
        "Act",
        "MtAct",
        "DiffusionPolicy",
        "DiffusionPolicy3d",
        "FlowPolicy",
        "ManiFlowPolicy",
    ]

    def __init__(self):
        self.setup_args()

    def setup_args(self):
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
            choices=self.policy_choices,
            help="policy",
        )
        parser.add_argument(
            "-h",
            "--help",
            action="store_true",
            help="Show this help message and continue",
        )

        self.args, remaining_argv = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + remaining_argv
        if self.args.policy is None:
            parser.print_help()
            sys.exit(1)
        elif self.args.help:
            parser.print_help()
            print("\n================================\n")
            sys.argv += ["--help"]

    def run(self):
        from robo_manip_baselines.common import camel_to_snake

        policy_module = importlib.import_module(
            f"{self.policy_parent_module_str}.{camel_to_snake(self.args.policy)}"
        )
        TrainPolicyClass = getattr(policy_module, f"Train{self.args.policy}")

        train = TrainPolicyClass()
        train.run()
        train.close()


if __name__ == "__main__":
    main = TrainMain()
    main.run()
