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
        self.vocab = None

    def build_vocab(self, premises, hypotheses):
        self.vocab = Vocabulary()

        for premise in premises:
            self.vocab.addSentence(premise)
        
        for hypothesis in hypotheses:
            self.vocab.addSentence(hypothesis)

        print("Vocab size: " + self.vocab.get_vocab_size())
        print("Saving vocab..")
        model_utils.save_vocab(self.save_path, self.vocab)
        print("Vocab saved!")

    def labels_to_indices(self, labels):
        label_dict = {'entailment': 2, 'contradiction': 0, 'neutral': 1}
        label_indices = [label_dict[t] for t in labels]
        return label_indices

    def convert_to_indices(self, premises, hypotheses):
        premise_indices = []
        hypothesis_indices = []

        for premise in premises:
            indices = [self.vocab.get_index(w) for w in word_tokenize(premise)]
            premise_indices.append(indices)

        for hypothesis in hypotheses:
            indices = [self.vocab.get_index(w) for w in word_tokenize(hypothesis)]
            hypothesis_indices.append(indices)
        
        return premise_indices, hypothesis_indices

    def create_train_data(self):
        train_df = load_utils.load_data(self.train_path)
        premises = train_df['sentence1'].to_list()
        hypotheses = train_df['sentence2'].to_list()
        labels = train_df['gold_label'].to_list()

        self.build_vocab(premises, hypotheses)

        premise_indices, hypothesis_indices = self.convert_to_indices(premises, hypotheses)
        label_indices = self.labels_to_indices(labels)

        train_data = DataSetLoader(premise_indices, hypothesis_indices, label_indices)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size = self.batch_size)

        return train_loader

    def create_val_data(self):
        val_df = load_utils.load_data(self.val_path)
        premises = val_df['sentence1'].to_list()
        hypotheses = val_df['sentence2'].to_list()
        labels = val_df['gold_label'].to_list()

        premise_indices, hypothesis_indices = self.convert_to_indices(premises, hypotheses)
        label_indices = self.labels_to_indices(labels)

        val_data = DataSetLoader(premise_indices, hypothesis_indices, label_indices)

        val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size)

        return val_loader

    def create_model(self):
        embeddings_index = load_utils.load_embeddings(self.emb_path)
        embedding_matrix = model_utils.create_embedding_matrix(embeddings_index, 300, self.vocab)

        if self.model_type == 'bilstm':
            model = BiLSTM(hidden_size=self.hidden_size, stacked_layers=self.stacked_layers, weights_matrix=embedding_matrix, device=self.device)

        model.to(self.device)
        return model

    def multi_acc(self, predictions, labels):
        acc = (torch.log_softmax(predictions, dim=1).argmax(dim=1) == labels).sum().float() / float(labels.size(0))
        return acc

    def train(self, train_data, model, criterion, optimizer):
        model.train()
        total_train_loss = 0
        total_train_acc  = 0
        for step, batch in enumerate(tqdm(train_data)):
            premises = batch[0].to(self.device)
            hypotheses = batch[1].to(self.device)
            labels = batch[2].to(self.device)
            
            model.zero_grad()
            predictions = model(premises, hypotheses)

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
            for step, batch in enumerate(tqdm(val_data)):
                premises = batch[0].to(self.device)
                hypotheses = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                model.zero_grad()
                predictions = model(premises, hypotheses)

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

        model = self.create_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        train_data = self.create_train_data()
        val_data = self.create_val_data()

        for epoch_i in range(0, self.epochs):
            train_acc, train_loss = self.train(train_data, model, criterion, optimizer)
            val_acc, val_loss = self.val(val_data, model, criterion)
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': train_loss,
                    'Training Accur.': train_acc,
                    'Valid. Loss': val_loss,
                    'Valid. Accur.': val_acc
                }
            )
            print(f'Epoch {epoch_i + 1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
            if val_acc > last_best:
                print("Saving model..")
                model_utils.save_model(model, optimizer, self.model_name, self.save_path, training_stats)
                print("Model saved.")

        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(model_utils.format_time(time.time()-total_t0)))