from torch.utils.data import Dataset

class DataSetLoader(Dataset):
    def __init__(self, premises, hypotheses, labels):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels

    def __len__(self):
        return len(self.premises) + len(self.hypotheses)

    def __getitem__(self, index):
        return self.premises[index], self.hypotheses[index], self.labels[index]