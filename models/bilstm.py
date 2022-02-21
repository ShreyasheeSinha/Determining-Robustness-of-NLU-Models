"""
PyTorch implementation of a BiLSTM model. The code was written 
using https://github.com/imran3180/pytorch-nli/blob/master/models/bilstm.py as a reference.
"""
import torch
import torch.nn as nn

class BiLSTM(nn.Module):

    def __init__(self, hidden_size, stacked_layers, weights_matrix, device, dropout=0.2, num_classes=2):
        super(BiLSTM, self).__init__()
        self.directions = 2
        self.num_layers = 2
        self.concat = 4
        self.device = device
        self.hidden_size = hidden_size
        self.stacked_layers = stacked_layers

        num_embeddings, embedding_dim = weights_matrix.shape[0], weights_matrix.shape[1]
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))

        self.lstm = nn.LSTM(embedding_dim, self.hidden_size, self.num_layers, bidirectional = True, batch_first = True, dropout = dropout)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout)

        self.lin1 = nn.Linear(self.hidden_size * self.concat, self.hidden_size)
        self.lin2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin3 = nn.Linear(self.hidden_size, num_classes)

        for lin in [self.lin1, self.lin2, self.lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        self.out = nn.Sequential(
            self.lin1,
            self.relu,
            self.dropout,
            self.lin2,
            self.relu,
            self.dropout,
            self.lin3
        )

    def forward_once(self, sequence, mask):
        batch_size = sequence.size(0)
        sequence_lens = mask.int().sum(1)
        h0 = torch.zeros(self.stacked_layers*2, batch_size, self.hidden_size).to(self.device) # 2 for bidirection 
        c0 = torch.zeros(self.stacked_layers*2, batch_size, self.hidden_size).to(self.device)
        embedded_sequence = self.embedding(sequence)
        packed_sequence = nn.utils.rnn.pack_padded_sequence(embedded_sequence, lengths=sequence_lens, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed_sequence, (h0, c0))
        return hidden

    def forward(self, premises, premise_mask, hypotheses, hypothesis_mask):
        premise = self.forward_once(premises, premise_mask)
        hypothesis = self.forward_once(hypotheses, hypothesis_mask)

        combined_outputs  = torch.cat((premise, hypothesis, torch.abs(premise - hypothesis), premise * hypothesis), dim=2)

        return self.out(combined_outputs[-1])
