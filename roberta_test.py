from util.roberta_dataset_loader import RobertaDatasetLoader
from models.transformer import Transformer
import numpy as np
import torch
from tqdm import tqdm
import time
import util.model_utils as model_utils
import argparse
import util.load_utils as load_utils
from sklearn.metrics import precision_score, recall_score, f1_score

class RobertaTest():

    def __init__(self, options):
        self.model_name = options['model_name']
        self.device = options['device']
        self.test_path = options['test_path']
        self.batch_size = options['batch_size']
        self.is_hypothesis_only = options['is_hypothesis_only']
        transformer = Transformer(self.model_name)
        self.model, self.tokenizer = transformer.get_model_and_tokenizer()
        self.model.to(self.device)

    def flat_accuracy(self, preds, labels):
        output_shape = preds.shape[-1]
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        if output_shape == 3:
            pred_flat = np.where(pred_flat <= 1, 0, 1)
        acc = np.sum(pred_flat == labels_flat) / len(labels_flat)
        precision = precision_score(labels_flat, pred_flat, zero_division=0)
        recall = recall_score(labels_flat, pred_flat, zero_division=0)
        f1 = f1_score(labels_flat, pred_flat, zero_division=0)
        return acc, precision, recall, f1

    def test(self, data_loader):
        self.model.eval()
        total_acc = 0
        total_loss = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0

        with torch.no_grad():
            for (pair_token_ids, mask_ids, seg_ids, y) in tqdm(data_loader):
                # optimizer.zero_grad()
                pair_token_ids = pair_token_ids.to(self.device)
                mask_ids = mask_ids.to(self.device)
                seg_ids = seg_ids.to(self.device)
                labels = y.to(self.device)

                if "bart" not in self.model_name:
                    result = self.model(pair_token_ids,
                                            token_type_ids=seg_ids,
                                            attention_mask=mask_ids,
                                            labels=labels,
                                            return_dict=True)
                else:
                    result = self.model(pair_token_ids,
                                            decoder_input_ids=seg_ids,
                                            attention_mask=mask_ids,
                                            labels=labels,
                                            return_dict=True)
            
                loss = result.loss
                logits = result.logits
                # print(logits.shape)
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                    
                    # loss = criterion(prediction, labels)
                acc, precision, recall, f1 = self.flat_accuracy(logits, label_ids)
                # print(acc, loss)

                total_loss += loss.item()
                total_acc  += acc
                total_precision += precision
                total_recall += recall
                total_f1 += f1
        
        acc  = total_acc/len(data_loader)
        precision = total_precision/len(data_loader)
        recall = total_recall/len(data_loader)
        f1 = total_f1/len(data_loader)
        loss = total_loss/len(data_loader)

        return acc, precision, recall, f1, loss
        
    def execute(self):
        total_t0 = time.time()

        test_df = load_utils.load_data(self.test_path)
        test_df['gold_label'] = test_df['gold_label'].astype(int)
        dataset = RobertaDatasetLoader(test_df, self.tokenizer, is_hypothesis_only=self.is_hypothesis_only)
        data_loader = dataset.get_data_loaders(self.batch_size)

        test_acc, test_precision, test_recall, test_f1, test_loss = self.test(data_loader)

        print(f'test_acc: {test_acc:.4f} test_precision: {test_precision:.4f} test_recall: {test_recall:.4f} test_f1: {test_f1:.4f}')

        print("Testing complete!")
        print("Total testing took {:} (h:mm:ss)".format(model_utils.format_time(time.time()-total_t0)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", help="Path to the test dataset jsonl file", default="./data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl") # TODO: Add proper path
    parser.add_argument("--batch_size", help="Batch size", type=int, default=32)
    parser.add_argument("--model_name", help="Name of the huggingface model or the path to the directoy containing a pre-trained transformer", default="roberta-large-mnli")
    parser.add_argument("--is_hypothesis_only", action='store_true')
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
    options['is_hypothesis_only'] = args.is_hypothesis_only
    print(options)

    roberta_tester = RobertaTest(options)
    roberta_tester.execute()