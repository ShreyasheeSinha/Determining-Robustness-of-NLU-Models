from torch.utils.data import Dataset

class DataSetLoader(Dataset):
    def __init__(self, premises, premises_mask, hypotheses, hypotheses_mask, labels, is_hypothesis_only=False):
        self.premises = premises
        self.premises_mask = premises_mask
        self.hypotheses = hypotheses
        self.hypotheses_mask = hypotheses_mask
        self.labels = labels
        self.is_hypothesis_only = is_hypothesis_only

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if not self.is_hypothesis_only:
            return self.premises[index], self.premises_mask[index], self.hypotheses[index], self.hypotheses_mask[index], self.labels[index]
        return [], [], self.hypotheses[index], self.hypotheses_mask[index], self.labels[index]