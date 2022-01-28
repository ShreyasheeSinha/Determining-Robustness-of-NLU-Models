import argparse

from click import option
from paraphraser import Paraphraser

import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the dataset jsonl file", default="./data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl")
    parser.add_argument("--save_path", help="Path with file name where to save the paraphrased dataset", default="./multinli_1.0_dev_mismatched_paraphrased.jsonl") # TODO: Add proper path
    return parser.parse_args()

if __name__ == '__main__':
    # Set numpy, tensorflow and python seeds for reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)

    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    options = {}
    options['device'] = device
    options['data_path'] = args.data_path
    options['save_path'] = args.save_path
    print(options)

    paraphraser = Paraphraser(options)
    paraphraser.execute()