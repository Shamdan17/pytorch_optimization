import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch Cifar10 Training")
    parser.add_argument("--batch_size", default=256, type=int, help="batch size")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument(
        "--epochs", default=100, type=int, help="number of training epochs"
    )
    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
    parser.add_argument("--save_freq", default=50, type=int, help="save frequency")
    parser.add_argument(
        "--save_dir", default="checkpoints", type=str, help="save directory"
    )

    # Add arguments for multi-gpu usage. Unused for single gpu
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--master-address", type=str, default="localhost")
    parser.add_argument(
        "--master-port",
        default="12355",
        help="Port used for distributed training",
    )

    return parser
