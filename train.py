import util.load_utils as load_utils
from util.vocab import Vocabulary
import util.model_utils as model_utils
from util.dataset_loader import DataSetLoader
from models.bilstm import BiLSTM
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
import numpy as np

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
        self.vocab = None

    def build_vocab(self, premises, hypotheses):
        self.vocab = Vocabulary()
        print("Building vocab..")
        for premise, hypothesis in tqdm(zip(premises, hypotheses), total=len(premises)):
            self.vocab.addSentence(premise.lower())
            self.vocab.addSentence(hypothesis.lower())

        print("Vocab size:", str(self.vocab.get_vocab_size()))
        print("Saving vocab..")
        model_utils.save_vocab(self.save_path, self.vocab, self.model_name)
        print("Vocab saved!")

    def labels_to_indices(self, labels):
        print("Coverting labels to indexes..")
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
            premise_tokens = word_tokenize(premise.lower())
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
            hypothesis_tokens = word_tokenize(hypothesis.lower())
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
        # self.vocab = model_utils.load_vocab(self.save_path, self.model_name)
        labels = val_df['gold_label'].to_list()

        premise_indices, premise_masks, hypothesis_indices, hypothesis_masks = self.convert_to_indices(premises, hypotheses)
        label_indices = self.labels_to_indices(labels)

        val_data = DataSetLoader(np.array(premise_indices), np.array(premise_masks), np.array(hypothesis_indices), np.array(hypothesis_masks), np.array(label_indices))

        val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

        return val_loader

    def create_model(self):
        embeddings_index = load_utils.load_embeddings(self.emb_path)
        embedding_matrix = model_utils.create_embedding_matrix(embeddings_index, 300, self.vocab)

        if self.model_type == 'bilstm':
            model = BiLSTM(hidden_size=self.hidden_size, stacked_layers=self.stacked_layers, weights_matrix=embedding_matrix, device=self.device)

        model.to(self.device)
        print(model)
        return model

    def multi_acc(self, predictions, labels):
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
                acc  = self.multi_acc(predictions, labels)

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