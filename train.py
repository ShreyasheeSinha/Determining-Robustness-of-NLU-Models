import util.load_utils as load_utils
from util.vocab import Vocabulary
import util.model_utils as model_utils
from util.dataset_loader import DataSetLoader
from models.bilstm import BiLSTM
from models.cbow import CBOW
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
import numpy as np
import argparse
import os
import string
from collections import Counter

class Trainer:

    def __init__(self, options):
        self.model_type = options['model_type']
        self.model_name = options['model_name']
        self.save_path = options['save_path'] # Path of the folder where everything will be saved
        self.device = options['device']
        self.train_path = options['train_path']
        self.val_path = options['val_path']
        self.emb_path = options['emb_path']
        self.batch_size = options['batch_size']
        self.epochs = options['epochs']
        self.hidden_size = options['hidden_size']
        self.stacked_layers = options['stacked_layers']
        self.learning_rate = options['learning_rate']
        self.seq_len = options['seq_len']
        self.num_classes = options['num_classes']
        self.vocab = None
        self.vocab_size = options['vocab_size']

    def strip_punctuations(self, sentence):
        table = str.maketrans(dict.fromkeys(string.punctuation))
        new_s = sentence.translate(table) 
        return new_s

    def build_vocab(self, premises, hypotheses):
        self.vocab = Vocabulary(self.vocab_size)
        print("Building vocab..")
        words = []
        for premise, hypothesis in tqdm(zip(premises, hypotheses), total=len(premises)):
            for token in self.strip_punctuations(premise).lower().split(' '):
                words.append(token)
            for token in self.strip_punctuations(hypothesis).lower().split(' '):
                words.append(token)

        vocab_words = Counter(words).most_common(self.vocab_size - 1)
        for word, _ in vocab_words:
            self.vocab.add_word(word)
        print("Vocab size:", str(self.vocab.get_vocab_size()))
        print("Saving vocab..")
        model_utils.save_vocab(self.save_path, self.vocab, self.model_name)
        print("Vocab saved!")

    def labels_to_indices(self, labels):
        print("Coverting labels to indexes..")
        if self.num_classes == 2:
            label_dict = {'entailment': 1, 'non-entailment': 0}
        else:
            label_dict = {'entailment': 2, 'contradiction': 0, 'neutral': 1}
        label_indices = [label_dict[t] for t in tqdm(labels)]
        return label_indices

    def convert_to_indices(self, premises, hypotheses):
        print("Coverting sentences to indexes..")
        premise_indices = []
        premise_masks = []
        hypothesis_indices = []
        hypothesis_masks = []

        for premise, hypothesis in tqdm(zip(premises, hypotheses), total=len(premises)):
            indices = []
            masks = []
            premise_tokens = premise.split(' ')
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
            hypothesis_tokens = hypothesis.split(' ')
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

    def create_train_data(self):
        print("Creating training data..")
        train_df = load_utils.load_data(self.train_path)
        premises = train_df['sentence1'].to_list()
        hypotheses = train_df['sentence2'].to_list()
        if self.num_classes == 2:
            train_df['gold_label'] = train_df['gold_label'].replace('contradiction', 'non-entailment')
            train_df['gold_label'] = train_df['gold_label'].replace('neutral', 'non-entailment')
        labels = train_df['gold_label'].to_list()

        self.build_vocab(premises, hypotheses)

        premise_indices, premise_masks, hypothesis_indices, hypothesis_masks = self.convert_to_indices(premises, hypotheses)
        label_indices = self.labels_to_indices(labels)

        train_data = DataSetLoader(np.array(premise_indices), np.array(premise_masks), np.array(hypothesis_indices), np.array(hypothesis_masks), np.array(label_indices))

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        return train_loader

    def create_val_data(self):
        print("Creating validation data..")
        val_df = load_utils.load_data(self.val_path)
        val_df = val_df[val_df['gold_label'] != '-'] # The dataset has some entries with labels as '-'
        premises = val_df['sentence1'].to_list()
        hypotheses = val_df['sentence2'].to_list()
        label_int = val_df['gold_label'].astype(int) # Convert boolean columns to int, True: 1 and False: 0
        label_indices = label_int.to_list()

        premise_indices, premise_masks, hypothesis_indices, hypothesis_masks = self.convert_to_indices(premises, hypotheses)

        val_data = DataSetLoader(np.array(premise_indices), np.array(premise_masks), np.array(hypothesis_indices), np.array(hypothesis_masks), np.array(label_indices))

        val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

        return val_loader

    def create_model(self):
        embeddings_index = load_utils.load_embeddings(self.emb_path)
        embedding_matrix = model_utils.create_embedding_matrix(embeddings_index, 300, self.vocab)

        if self.model_type == 'bilstm':
            model = BiLSTM(hidden_size=self.hidden_size, stacked_layers=self.stacked_layers, weights_matrix=embedding_matrix, device=self.device, num_classes=self.num_classes)
        elif self.model_type == 'cbow':
            model = CBOW(weights_matrix=embedding_matrix, num_classes=self.num_classes)

        model.to(self.device)
        print(model)
        return model

    def multi_acc(self, predictions, labels, val=False):
        if val and self.num_classes == 3:
            predictions = torch.log_softmax(predictions, dim=1).argmax(dim=1)
            two_class_predictions = torch.where(predictions <= 1, 0, 1) # Collapse neutral and contradiction into a single class 0, entailment becomes class 1
            acc = (two_class_predictions == labels).sum().float() / float(labels.size(0))
        else:
            acc = (torch.log_softmax(predictions, dim=1).argmax(dim=1) == labels).sum().float() / float(labels.size(0))
        return acc

    def train(self, train_data, model, criterion, optimizer):
        model.train()
        total_train_loss = 0
        total_train_acc  = 0
        for premises, premise_mask, hypotheses, hypothesis_mask, labels in tqdm(train_data):
            premises = premises.to(self.device)
            hypotheses = hypotheses.to(self.device)
            labels = labels.to(self.device)
            
            model.zero_grad()
            predictions = model(premises, premise_mask, hypotheses, hypothesis_mask)

            loss = criterion(predictions, labels)
            acc  = self.multi_acc(predictions, labels)

            loss.backward()
            optimizer.step()
      
            total_train_loss += loss.item()
            total_train_acc  += acc.item()
        
        train_acc  = total_train_acc/len(train_data)
        train_loss = total_train_loss/len(train_data)

        return train_acc, train_loss

    def val(self, val_data, model, criterion):
        model.eval()
        total_val_acc  = 0
        total_val_loss = 0
        with torch.no_grad():
            for premises, premise_mask, hypotheses, hypothesis_mask, labels in tqdm(val_data):
                premises = premises.to(self.device)
                hypotheses = hypotheses.to(self.device)
                labels = labels.to(self.device)
                
                model.zero_grad()
                predictions = model(premises, premise_mask, hypotheses, hypothesis_mask)

                loss = criterion(predictions, labels)
                acc  = self.multi_acc(predictions, labels, val=True)

                total_val_loss += loss.item()
                total_val_acc  += acc.item()

        val_acc = total_val_acc/len(val_data)
        val_loss = total_val_loss/len(val_data)

        return val_acc, val_loss

    def execute(self):
        total_t0 = time.time()
        last_best = 0
        training_stats = []

        train_data = self.create_train_data()
        val_data = self.create_val_data()

        model = self.create_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        for epoch_i in range(0, self.epochs):
            train_acc, train_loss = self.train(train_data, model, criterion, optimizer)
            val_acc, val_loss = self.val(val_data, model, criterion)
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }
            )
            print(f'Epoch {epoch_i + 1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
            if val_acc > last_best:
                print("Saving model..")
                model_utils.save_model(model, optimizer, self.model_name, self.save_path, training_stats)
                last_best = val_acc
                print("Model saved.")

        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(model_utils.format_time(time.time()-total_t0)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", help="The model type you wish to use", choices=["bilstm", "cbow"], default="bilstm")
    parser.add_argument("--save_path", help="Directory to save model and model checkpoints into", default="./saved_model")
    parser.add_argument("--train_path", help="Path to the train dataset jsonl file", default="./data/multinli_1.0/multinli_1.0_train.jsonl")
    parser.add_argument("--val_path", help="Path to the validation dataset jsonl file", default="./data/multinli_1.0/multinli_1.0_dev_matched.jsonl")
    parser.add_argument("--batch_size", help="Batch size", type=int, default=32)
    parser.add_argument("--emb_path", help="Path to the GloVe embeddings", default="./data/glove.840B.300d.txt")
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=15)
    parser.add_argument("--model_name", help="A custom name given to your model", required=True)
    parser.add_argument("--hidden_size", help="Hidden units in the LSTM", type=int, default=64)
    parser.add_argument("--stacked_layers", help="Number of stacked LSTM units", type=int, default=2)
    parser.add_argument("--seq_len", help="Maximum sequence length", type=int, default=50)
    parser.add_argument("--vocab_size", help="The size of the vocabulary", type=int, default=50000)
    parser.add_argument("--num_classes", help="Number of output classes - RTE has 2, MNLI has 3", type=int, choices=[2, 3], default=2)
    return check_args(parser.parse_args())

def check_args(args):
    assert args.epochs >= 1
    assert args.batch_size >= 1
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
    options['num_classes'] = args.num_classes
    options['vocab_size'] = args.vocab_size
    options['learning_rate'] = 0.005 # TODO: Make this a CLI arg
    print(options)
    model_utils.save_model_config(save_path, model_name, options)
    trainer = Trainer(options)
    trainer.execute()