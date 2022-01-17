import os
import pickle
import argparse

def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", help="Directory to save model and model checkpoints into", default="./saved_model")
    parser.add_argument("--train_path", help="Path to the train dataset", default="./data") # TODO: Add proper path
    parser.add_argument("--val_path", help="Path to the validation dataset", default="./data") # TODO: Add proper path
    parser.add_argument("--test_path", help="Path to the test dataset", default="./data") # TODO: Add proper path
    parser.add_argument("--batch_size", help="Batch size", type=int, default=32)
    parser.add_argument("--val_batch", help="Validation batch size", type=int, default=1)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=15)
    parser.add_argument("--model_name", help="A custom name given to your model")
    parser.add_argument("--model_type", help="The model type you wish to use", choices=["bilstm", "cbow"], default="bilstm")
    args, _ = parser.parse_known_args()
    return args

def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print ("Created a path: %s"%(path))