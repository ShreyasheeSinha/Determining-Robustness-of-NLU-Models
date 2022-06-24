from turtle import forward
import torch
import torch.nn as nn

class CBOW(nn.Module):

    def __init__(self, weights_matrix, dropout=0.2, num_classes=2, is_hypothesis_only=False):
        super(CBOW, self).__init__()

        num_embeddings, embedding_dim = weights_matrix.shape[0], weights_matrix.shape[1]
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
        self.is_hypothesis_only = is_hypothesis_only

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout)

        if not self.is_hypothesis_only:
            self.lin1 = nn.Linear(4 * embedding_dim, embedding_dim) # As per the concat logic the linear layer needs to be 4 * embedding_dim
        else:
            self.lin1 = nn.Linear(embedding_dim, embedding_dim)
        self.lin2 = nn.Linear(embedding_dim, embedding_dim)
        self.lin3 = nn.Linear(embedding_dim, num_classes)

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
        embedded_sequence = self.embedding(sequence)
        avg_vector = embedded_sequence.mean(1)
        return avg_vector

    """
    The masks are added to maintain code consistency between
    the BiLSTM and CBOW model. These are not used in the CBOW
    implementation.
    """
    def forward(self, premises, premise_mask, hypotheses, hypothesis_mask):
        if not self.is_hypothesis_only:
            premise = self.forward_once(premises)
        hypothesis = self.forward_once(hypotheses)

        if not self.is_hypothesis_only:
            combined_outputs  = torch.cat((premise, hypothesis, torch.abs(premise - hypothesis), premise * hypothesis), dim=1)
        else:
            combined_outputs = hypothesis

        return self.out(combined_outputs)