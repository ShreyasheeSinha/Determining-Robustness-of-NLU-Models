from util.roberta_dataset_loader import RobertaDatasetLoader
from models.transformer import Transformer
import numpy as np
import torch
from tqdm import tqdm
import time
import util.model_utils as model_utils
import argparse

class RobertaTest():

    def __init__(self, options):
        self.model_name = options['model_name']
        self.device = options['device']
        self.test_path = options['test_path']
        self.batch_size = options['batch_size']
        transformer = Transformer(self.model_name)
        self.model, self.tokenizer = transformer.get_model_and_tokenizer()
        self.model.to(self.device)

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def test(self, data_loader):
        self.model.eval()
        total_acc = 0
        total_loss = 0

        with torch.no_grad():
            for (pair_token_ids, mask_ids, seg_ids, y) in tqdm(data_loader):
                # optimizer.zero_grad()
                pair_token_ids = pair_token_ids.to(self.device)
                mask_ids = mask_ids.to(self.device)
                seg_ids = seg_ids.to(self.device)
                labels = y.to(self.device)

                # prediction = model(pair_token_ids, mask_ids, seg_ids)
                result = self.model(pair_token_ids, 
                                        token_type_ids=seg_ids, 
                                        attention_mask=mask_ids, 
                                        labels=labels,
                                        return_dict=True)
            
                loss = result.loss
                logits = result.logits
                # print(logits.shape)
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                    
                    # loss = criterion(prediction, labels)
                acc = self.flat_accuracy(logits, label_ids)
                # print(acc, loss)

                total_loss += loss.item()
                total_acc  += acc
        
        acc  = total_acc/len(data_loader)
        loss = total_loss/len(data_loader)

        return acc, loss
        
    def execute(self):
        total_t0 = time.time()

        dataset = RobertaDatasetLoader(self.test_path, self.tokenizer)
        data_loader = dataset.get_data_loaders(self.batch_size)

        test_acc, test_loss = self.test(data_loader)

        print(f'test_loss: {test_loss:.4f} test_acc: {test_acc:.4f}')

        print("Testing complete!")
        print("Total testing took {:} (h:mm:ss)".format(model_utils.format_time(time.time()-total_t0)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", help="Path to the test dataset jsonl file", default="./data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl") # TODO: Add proper path
    parser.add_argument("--batch_size", help="Batch size", type=int, default=32)
    parser.add_argument("--model_name", help="Name of the huggingface model or the path to the directoy containing a pre-trained transformer", default="roberta-large-mnli")
    return parser.parse_args()

if __name__ == '__main__':
    # Set numpy, tensorflow and python seeds for reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)

    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    options = {}
    options['batch_size'] = args.batch_size
    options['device'] = device
    options['test_path'] = args.test_path
    options['model_name'] = args.model_name
    print(options)

    roberta_tester = RobertaTest(options)
    roberta_tester.execute()