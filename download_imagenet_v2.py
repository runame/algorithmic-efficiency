"""Download ImageNetV2 dataset."""

from argparse import ArgumentParser

from datasets.dataset_setup import download_imagenet_v2

if __name__ == "__main__":
    parser = ArgumentParser(description="Download ImageNetV2 dataset")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/imagenet",
        help="Directory to save the dataset",
    )

    args = parser.parse_args()

    download_imagenet_v2(args.data_dir)
