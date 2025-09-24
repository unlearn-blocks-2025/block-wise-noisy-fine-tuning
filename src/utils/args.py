import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Block-wise Unlearning Experiments")

    # General
    parser.add_argument(
        "--experiment_type",
        type=str,
        default="test_run",
        help="Name of experiment, used for saving files",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "mnist"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="blocks",
        choices=["train_base", "retrain", "unlearning_base", "blocks"],
        help="Experiment mode",
    )
    parser.add_argument(
        "--modelnames",
        type=str,
        nargs="+",
        default=["block_type copy 3"],
        help="List of model names, used for saving",
    )
    parser.add_argument(
        "--exps",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="List of experiment indices",
    )

    # Block-specific
    parser.add_argument(
        "--blocks_mode",
        type=str,
        default="qr",
        choices=["qr", "identity", "perm_identity"],
        help="Block construction mode",
    )
    parser.add_argument(
        "--blocks_split_type",
        type=str,
        default="equal",
        choices=["equal", "layers", "head"],
        help="How to split blocks",
    )

    # Training options
    parser.add_argument("--training", action="store_true", help="Run training")
    parser.add_argument("--evaluation", action="store_true", help="Run evaluation")
    parser.add_argument(
        "--do_visualization", action="store_true", help="Run visualization"
    )
    parser.add_argument("--do_mia", action="store_true", help="Run MIA attacks")
    parser.add_argument("--tag", type=str, default=None, help="Optional tag")
    parser.add_argument("--one_color", action="store_true", help="Plot with one color")
    parser.add_argument(
        "--fix_budget", type=float, default=None, help="Fix budget for visualization"
    )
    parser.add_argument(
        "--T_max", type=int, default=1000, help="Max T for visualization"
    )

    # Paths
    parser.add_argument(
        "--trained_full_state_path",
        type=str,
        default="experiments_results/models/resnet_cifar10/basemodel.pt",
        help="Path to pretrained full model",
    )

    return parser.parse_args()
