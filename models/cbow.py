from turtle import forward
import torch
import torch.nn as nn

class CBOW(nn.Module):

    def __init__(self, weights_matrix, seq_len, dropout=0.2):
        super(CBOW, self).__init__()

        num_embeddings, embedding_dim = weights_matrix.shape[0], weights_matrix.shape[1]
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout)

        self.lin1 = nn.Linear(4 * seq_len * embedding_dim, 128) # As per the concat logic the linear layer needs to be 4 * seq_len * embedding_dim
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, 3)

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

    def forward_once(self, sequence):
        batch_size = sequence.size(0)
        embedded_sequence = self.embedding(sequence)
        return embedded_sequence.view(batch_size, -1)

    """
    The masks are added to maintain code consistency between
    the BiLSTM and CBOW model. These are not used in the CBOW
    implementation.
    """
    def forward(self, premises, premise_mask, hypotheses, hypothesis_mask):
        premise = self.forward_once(premises)
        hypothesis = self.forward_once(hypotheses)

        combined_outputs  = torch.cat((premise, hypothesis, torch.abs(premise - hypothesis), premise * hypothesis), dim=1)

        return self.out(combined_outputs)