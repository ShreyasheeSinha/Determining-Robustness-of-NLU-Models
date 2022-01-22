import os
import pickle
import argparse

import torch
import numpy as np
from train import Trainer
from test import Tester
import util.model_utils as model_utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="Specify if you want to test the model", action="store_true", default=False)
    parser.add_argument("--model_type", help="The model type you wish to use", choices=["bilstm", "cbow"], default="bilstm")
    parser.add_argument("--save_path", help="Directory to save model and model checkpoints into", default="./saved_model")
    parser.add_argument("--train_path", help="Path to the train dataset jsonl file", default="./data/multinli_1.0/multinli_1.0_train.jsonl") # TODO: Add proper path
    parser.add_argument("--val_path", help="Path to the validation dataset jsonl file", default="./data/multinli_1.0/multinli_1.0_dev_matched.jsonl") # TODO: Add proper path
    parser.add_argument("--test_path", help="Path to the test dataset jsonl file", default="./data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl") # TODO: Add proper path
    parser.add_argument("--batch_size", help="Batch size", type=int, default=32)
    parser.add_argument("--emb_path", help="Path to the GloVe embeddings", default="./data/glove.840B.300d.txt") # TODO: Add proper path
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=15)
    parser.add_argument("--model_name", help="A custom name given to your model", required=True)
    parser.add_argument("--hidden_size", help="Hidden units in the LSTM", type=int, default=64)
    parser.add_argument("--stacked_layers", help="Number of stacked LSTM units", type=int, default=2)
    parser.add_argument("--seq_len", help="Maximum sequence length", type=int, default=50)
    return check_args(parser.parse_args())

def check_args(args):
    assert args.epochs >= 1
    assert args.batch_size >= 1
    if not args.test:
        save_path = f'{args.save_path}/'
        create_path(save_path)
    return args

def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print ("Created a path: %s"%(path))

if __name__ == '__main__':
    # Set numpy, tensorflow and python seeds for reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)

    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    options = {}
    save_path = f'{args.save_path}/'
    model_name = args.model_type + "_" + args.model_name

    if args.test:
        print("Testing model..")
        options = model_utils.load_model_config(save_path, model_name)
        options['device'] = device
        options['emb_path'] = args.emb_path
        options['test_path'] = args.test_path
        options['batch_size'] = args.batch_size
        print(options)
        tester = Tester(options)
        tester.execute()
    else:
        print("Training model..")
        options['model_type'] = args.model_type
        options['model_name'] = model_name
        options['save_path'] = save_path
        options['hidden_size'] = args.hidden_size
        options['stacked_layers'] = args.stacked_layers
        options['seq_len'] = args.seq_len
        options['device'] = device
        options['train_path'] = args.train_path
        options['val_path'] = args.val_path
        options['epochs'] = args.epochs
        options['emb_path'] = args.emb_path
        options['batch_size'] = args.batch_size
        options['learning_rate'] = 0.005 # TODO: Make this a CLI arg
        print(options)
        model_utils.save_model_config(save_path, model_name, options)
        trainer = Trainer(options)
        trainer.execute()