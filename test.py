import util.load_utils as load_utils
import util.model_utils as model_utils
from util.dataset_loader import DataSetLoader
from models.bilstm import BiLSTM
from models.cbow import CBOW
import torch
import torch.nn as nn
import time
from tqdm import tqdm
import numpy as np
import argparse
import string

class Tester:

    def __init__(self, options):
        self.model_type = options['model_type']
        self.model_name = options['model_name']
        self.save_path = options['save_path'] # Path of the folder where everything will be saved
        self.device = options['device']
        self.test_path = options['test_path']
        self.emb_path = options['emb_path']
        self.batch_size = options['batch_size']
        self.seq_len = options['seq_len']
        self.hidden_size = options['hidden_size']
        self.stacked_layers = options['stacked_layers']
        self.num_classes = options['num_classes']
        self.vocab = model_utils.load_vocab(self.save_path, self.model_name)

    def labels_to_indices(self, labels):
        label_dict = {'entailment': 2, 'contradiction': 0, 'neutral': 1}
        label_indices = [label_dict[t] for t in labels]
        return label_indices

    def strip_punctuations(self, sentence):
        table = str.maketrans(dict.fromkeys(string.punctuation))
        new_s = sentence.translate(table) 
        return new_s

    def convert_to_indices(self, premises, hypotheses):
        print("Coverting sentences to indexes..")
        premise_indices = []
        premise_masks = []
        hypothesis_indices = []
        hypothesis_masks = []

        for premise, hypothesis in tqdm(zip(premises, hypotheses), total=len(premises)):
            indices = []
            masks = []
            premise_tokens = self.strip_punctuations(premise).split(' ')
            for i in range(self.seq_len):
                if i >= len(premise_tokens):
                    indices.append(0) # Append padding
                    masks.append(0)
                else:
                    w = premise_tokens[i]
                    if self.vocab.get_index(w):
                        indices.append(self.vocab.get_index(w))
                    else:
                        indices.append(1) # UNK token index
                    masks.append(1)
            premise_indices.append(indices)
            premise_masks.append(masks)
            
            indices = []
            masks = []
            hypothesis_tokens = self.strip_punctuations(hypothesis).split(' ')
            for i in range(self.seq_len):
                if i >= len(hypothesis_tokens):
                    indices.append(0) # Append padding
                    masks.append(0)
                else:
                    w = hypothesis_tokens[i]
                    if self.vocab.get_index(w):
                        indices.append(self.vocab.get_index(w))
                    else:
                        indices.append(1) # UNK token index
                    masks.append(1)
            hypothesis_indices.append(indices)
            hypothesis_masks.append(masks)
        
        return premise_indices, premise_masks, hypothesis_indices, hypothesis_masks

    def create_test_data(self):
        val_df = load_utils.load_data(self.test_path)
        val_df = val_df[val_df['gold_label'] != '-']
        premises = val_df['sentence1'].to_list()
        hypotheses = val_df['sentence2'].to_list()
        label_int = val_df['gold_label'].astype(int) # Convert boolean columns to int, True: 1 and False: 0
        label_indices = label_int.to_list()

        premise_indices, premise_masks, hypothesis_indices, hypothesis_masks = self.convert_to_indices(premises, hypotheses)
        # label_indices = self.labels_to_indices(labels)

        test_data = DataSetLoader(np.array(premise_indices), np.array(premise_masks), np.array(hypothesis_indices), np.array(hypothesis_masks), np.array(label_indices))

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size)

        return test_loader

    def create_model(self):
        embeddings_index = load_utils.load_embeddings(self.emb_path)
        embedding_matrix = model_utils.create_embedding_matrix(embeddings_index, 300, self.vocab)

        if self.model_type == 'bilstm':
            model = BiLSTM(hidden_size=self.hidden_size, stacked_layers=self.stacked_layers, weights_matrix=embedding_matrix, device=self.device, num_classes=self.num_classes)
        elif self.model_type == 'cbow':
            model = CBOW(weights_matrix=embedding_matrix, num_classes=self.num_classes)

        model.to(self.device)
        return model

    def multi_acc(self, predictions, labels):
        if self.num_classes == 3:
            predictions = torch.log_softmax(predictions, dim=1).argmax(dim=1)
            two_class_predictions = torch.where(predictions <= 1, 0, 1) # Collapse neutral and contradiction into a single class 0, entailment becomes class 1
            acc = (two_class_predictions == labels).sum().float() / float(labels.size(0))
        else:
            acc = (torch.log_softmax(predictions, dim=1).argmax(dim=1) == labels).sum().float() / float(labels.size(0))
        return acc

    def test(self, test_data, model, criterion):
        model.eval()
        total_test_acc  = 0
        total_test_loss = 0
        with torch.no_grad():
            for premises, premise_mask, hypotheses, hypothesis_mask, labels in tqdm(test_data):
                premises = premises.to(self.device)
                hypotheses = hypotheses.to(self.device)
                labels = labels.to(self.device)
                
                model.zero_grad()
                predictions = model(premises, premise_mask, hypotheses, hypothesis_mask)

                loss = criterion(predictions, labels)
                acc  = self.multi_acc(predictions, labels)

                total_test_loss += loss.item()
                total_test_acc  += acc.item()

        test_acc = total_test_acc/len(test_data)
        test_loss = total_test_loss/len(test_data)

        return test_acc, test_loss

    def execute(self):
        total_t0 = time.time()
        test_data = self.create_test_data()
        
        model = self.create_model()
        criterion = nn.CrossEntropyLoss()
        model_info = model_utils.load_model(self.save_path, self.model_name)

        model.load_state_dict(model_info['model_state_dict'])

        test_acc, test_loss = self.test(test_data, model, criterion)
        print(f'test_loss: {test_loss:.4f} test_acc: {test_acc:.4f}')

        print("Testing complete!")
        print("Total testing took {:} (h:mm:ss)".format(model_utils.format_time(time.time()-total_t0)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", help="The model type you wish to use", choices=["bilstm", "cbow"], default="bilstm")
    parser.add_argument("--save_path", help="Directory to save model and model checkpoints into", default="./saved_model")
    parser.add_argument("--test_path", help="Path to the test dataset jsonl file", default="./data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl")
    parser.add_argument("--batch_size", help="Batch size", type=int, default=32)
    parser.add_argument("--emb_path", help="Path to the GloVe embeddings", default="./data/glove.840B.300d.txt")
    parser.add_argument("--model_name", help="A custom name given to your model", required=True)
    return check_args(parser.parse_args())

def check_args(args):
    assert args.batch_size >= 1
    return args

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

    print("Testing model..")
    options = model_utils.load_model_config(save_path, model_name)
    options['device'] = device
    options['emb_path'] = args.emb_path
    options['test_path'] = args.test_path
    options['batch_size'] = args.batch_size
    print(options)
    tester = Tester(options)
    tester.execute()