#necessary imports for run script
import argparse
import os
import subprocess
import sys

#names for models for flags you can pass
model_aliases = {
    "base": "vit_base_patch16_224",
    "small": "vit_small_patch16_224",
    "tiny": "vit_tiny_patch16_224",
    "large": "vit_large_patch16_224"
}
#parses arguments
def parse_args():
    #creates parser
    parser = argparse.ArgumentParser(
        description="Thin convenience wrapper around trainer.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    #adds argument for model, sets default to vit tiny for model usag
    parser.add_argument(
        "-m",
        "--model",
        default="tiny",
    )
    #adds argument for preparation type, either base or distort
    parser.add_argument(
        "-p",
        "--prep",
        default="base",
    )
    #which dataset trained on; k=1,2,3, or 5
    parser.add_argument(
        "-k",
        "--kshot",
        type=int,
        default=1,
    )
    #lambda value used in direction penalty
    parser.add_argument(
        "-L",
        "-lambda",
        "--lambda_penalty",
        dest="lambda_penalty",
        type=float,
        default=0.0,
    )
    #learning rate
    parser.add_argument(
        "-r",
        "-lr",
        "--lr",
        dest="lr",
        type=float,
        default=3e-4,
    )
    #batch size
    parser.add_argument(
        "-b",
        "--bs",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=64,
    )
    #epoch count
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=30,
    )
    #weight decay
    parser.add_argument(
        "--wd",
        "--weight_decay",
        dest="weight_decay",
        type=float,
        default=1e-4,
    )
    #number of workers or cpus used
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    #device trained on
    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )
    #seed used for training/shuffling/etc.
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    #output directory for checkpoints
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints.",
    )
    #potential extra arguments
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
    )
    #returns the arguments parsed
    return parser.parse_args()

#gets full proepr name for model from model aliases dict above
def resolve_model_name(model_arg: str) -> str:
    return model_aliases.get(model_arg, model_arg)


def main():
    #obtains args
    args = parse_args()
    #gets model name
    model_name = resolve_model_name(args.model)
    #inits this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    #collects trainer
    trainer_path = os.path.join(this_dir, "trainer.py")
    #sets up configuration based on args
    cmd = [
        sys.executable,
        trainer_path,
        "--model",
        model_name,
        "--prep_type",
        args.prep,
        "--k",
        str(args.kshot),
        "--lambda_penalty",
        str(args.lambda_penalty),
        "--lr",
        str(args.lr),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
    ]
    #extends configuration if the rest are applicable, e.g. weight decay
    cmd.extend(["--weight_decay", str(args.weight_decay)])
    #number of workers
    cmd.extend(["--num_workers", str(args.num_workers)])
    #which device
    if args.device is not None:
        cmd.extend(["--device", args.device])
    #which seed
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    #which output directory
    if args.output_dir is not None:
        cmd.extend(["--output_dir", args.output_dir])
    #add additional args
    if args.extra:
        cmd.append("--")
        cmd.extend(args.extra)

    #runs with configuration
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
