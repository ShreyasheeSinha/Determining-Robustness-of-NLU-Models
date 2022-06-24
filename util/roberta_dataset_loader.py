from cProfile import label
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class RobertaDatasetLoader(Dataset):

    def __init__(self, data, tokenizer, label_dict=None, is_hypothesis_only=False):
        self.data = data
        self.tokenizer = tokenizer
        self.label_dict = label_dict
        self.is_hypothesis_only = is_hypothesis_only

    def process_data(self):
        print("Processing data..")
        token_ids = []
        mask_ids = []
        seg_ids = []
        y = []

        premise_list = self.data['sentence1'].to_list()
        hypothesis_list = self.data['sentence2'].to_list()
        label_list = self.data['gold_label'].to_list()

        for (premise, hypothesis, label) in tqdm(zip(premise_list, hypothesis_list, label_list), total=len(premise_list)):

            if premise != None and hypothesis != None:
                if not self.is_hypothesis_only:
                    encoded_values = self.tokenizer.encode_plus(premise, hypothesis, return_token_type_ids=True, return_attention_mask=True)
                else:
                    encoded_values = self.tokenizer.encode_plus(hypothesis, return_token_type_ids=True, return_attention_mask=True)
                token_ids.append(torch.tensor(encoded_values['input_ids']))
                seg_ids.append(torch.tensor(encoded_values['token_type_ids']))
                mask_ids.append(torch.tensor(encoded_values['attention_mask']))
                if self.label_dict != None: # In case of the RTE dataset, each column value is an integer itself. Hence, no label_dict is needed
                    y.append(self.label_dict[label])
                else:
                    y.append(label)
        
        token_ids = pad_sequence(token_ids, batch_first=True)
        mask_ids = pad_sequence(mask_ids, batch_first=True)
        seg_ids = pad_sequence(seg_ids, batch_first=True)
        y = torch.tensor(y)
        dataset = TensorDataset(token_ids, mask_ids, seg_ids, y)
        return dataset

    def get_data_loaders(self, batch_size=32, shuffle=True):
        processed_dataset = self.process_data()

        data_loader = DataLoader(
            processed_dataset,
            shuffle=shuffle,
            batch_size=batch_size
        )

        return data_loader