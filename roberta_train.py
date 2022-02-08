from util.roberta_dataset_loader import RobertaDatasetLoader
from models.transformer import Transformer
import numpy as np
import torch
from tqdm import tqdm
import time
import util.model_utils as model_utils
import argparse
from transformers import AdamW, get_linear_schedule_with_warmup

class RobertaTrain():

    def __init__(self, options):
        self.model_name = options['model_name']
        self.device = options['device']
        self.train_path = options['train_path']
        self.val_path = options['val_path']
        self.batch_size = options['batch_size']
        self.epochs = options['epochs']
        self.save_path = options['save_path']
        transformer = Transformer(self.model_name, classification_head=True)
        self.model, self.tokenizer = transformer.get_model_and_tokenizer()
        self.model.to(self.device)
    
    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def train(self, data_loader, optimizer, scheduler):
        self.model.train()
        total_acc = 0
        total_loss = 0

        for (pair_token_ids, mask_ids, seg_ids, y) in tqdm(data_loader):
            # optimizer.zero_grad()
            pair_token_ids = pair_token_ids.to(self.device)
            mask_ids = mask_ids.to(self.device)
            seg_ids = seg_ids.to(self.device)
            labels = y.to(self.device)
            self.model.zero_grad()

            # prediction = model(pair_token_ids, mask_ids, seg_ids)
            result = self.model(pair_token_ids, 
                                        token_type_ids=seg_ids, 
                                        attention_mask=mask_ids, 
                                        labels=labels,
                                        return_dict=True)
            
            loss = result.loss
            logits = result.logits
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
                    
            acc = self.flat_accuracy(logits, label_ids)

            total_loss += loss.item()
            total_acc  += acc
        
        acc  = total_acc/len(data_loader)
        loss = total_loss/len(data_loader)

        return acc, loss

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
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
 
                acc = self.flat_accuracy(logits, label_ids)

                total_loss += loss.item()
                total_acc  += acc
        
        acc  = total_acc/len(data_loader)
        loss = total_loss/len(data_loader)

        return acc, loss

    def execute(self):
        total_t0 = time.time()
        last_best = 0
        print("Training model..")

        train_dataset = RobertaDatasetLoader(self.train_path, self.tokenizer)
        train_data_loader = train_dataset.get_data_loaders(self.batch_size)

        val_dataset = RobertaDatasetLoader(self.val_path, self.tokenizer)
        val_data_loader = val_dataset.get_data_loaders(self.batch_size)

        optimizer = AdamW(self.model.parameters(),
                lr = 3e-5,#lr = 4e-5, # args.learning_rate - default is 5e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
            )

        total_steps = len(train_data_loader) * self.epochs

        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 1, # Default value in run_glue.py
                                                    num_training_steps = total_steps)

        for epoch_i in range(0, self.epochs):
            train_acc, train_loss = self.train(train_data_loader, optimizer, scheduler)
            val_acc, val_loss = self.test(val_data_loader)

            print(f'Epoch {epoch_i + 1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
            if val_acc > last_best:
                print("Saving model..")
                model_utils.save_transformer(self.model, self.tokenizer, self.model_name, self.save_path)
                last_best = val_acc
                print("Model saved.")

        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(model_utils.format_time(time.time()-total_t0)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", help="Path to the train dataset jsonl file", default="./data/multinli_1.0/multinli_1.0_train.jsonl") # TODO: Add proper path
    parser.add_argument("--val_path", help="Path to the validation dataset jsonl file", default="./data/multinli_1.0/multinli_1.0_dev_matched.jsonl")
    parser.add_argument("--save_path", help="Directory to save the model", default="./saved_model")
    parser.add_argument("--batch_size", help="Batch size", type=int, default=4)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=5)
    parser.add_argument("--model_name", help="Name of the huggingface model or the path to the directory containing a pre-trained transformer", default="roberta-base")
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
    options['train_path'] = args.train_path
    options['val_path'] = args.val_path
    options['model_name'] = args.model_name
    options['save_path'] = args.save_path
    options['epochs'] = args.epochs
    print(options)

    roberta_trainer = RobertaTrain(options)
    roberta_trainer.execute()