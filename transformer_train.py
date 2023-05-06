from util.transformer_dataset_loader import TransformerDatasetLoader
from models.transformer import Transformer
import numpy as np
import torch
from tqdm import tqdm
import time
import util.model_utils as model_utils
import argparse
from transformers import AdamW, get_linear_schedule_with_warmup
import util.load_utils as load_utils
import os

class TransformerTrain():

    def __init__(self, options):
        self.model_name = options['model_name']
        self.device = options['device']
        self.train_path = options['train_path']
        self.val_path = options['val_path']
        self.batch_size = options['batch_size']
        self.epochs = options['epochs']
        self.save_path = options['save_path']
        self.num_classes = options['num_classes']
        self.gradient_accumulation = options['gradient_accumulation']
        self.is_hypothesis_only = options['is_hypothesis_only']
        transformer = Transformer(self.model_name, classification_head=True, num_classes=self.num_classes)
        self.model, self.tokenizer = transformer.get_model_and_tokenizer()
        self.model.to(self.device)
        self.train_data_loader = None
        self.val_data_loader = None
    
    def flat_accuracy(self, preds, labels, val=False):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        if val and self.num_classes == 3:
            pred_flat = np.where(pred_flat <= 1, 0, 1)
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def train(self, optimizer, scheduler):
        self.model.train()
        self.model.zero_grad()
        total_acc = 0
        total_loss = 0
        data_loader = self.train_data_loader

        for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(tqdm(data_loader)):
            pair_token_ids = pair_token_ids.to(self.device)
            mask_ids = mask_ids.to(self.device)
            seg_ids = seg_ids.to(self.device)
            labels = y.to(self.device)

            result = self.model(pair_token_ids, 
                                    decoder_input_ids=seg_ids, 
                                    attention_mask=mask_ids, 
                                    labels=labels,
                                    return_dict=True)
            
            loss = result.loss
            if self.gradient_accumulation > 0:
                loss = loss/self.gradient_accumulation
            logits = result.logits
            loss.backward()
            if self.gradient_accumulation == 0:
                optimizer.step()
                self.model.zero_grad()
                scheduler.step()
            elif ((batch_idx + 1) % self.gradient_accumulation == 0) or ((batch_idx + 1) == len(data_loader)):
                optimizer.step()
                self.model.zero_grad()
                scheduler.step()
            if (batch_idx + 1) % 10000 == 0:
                train_acc = total_acc/(batch_idx + 1)
                train_loss = total_loss/(batch_idx + 1)
                val_acc, val_loss = self.test()
                print(f'train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
                self.model.train()
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
                    
            acc = self.flat_accuracy(logits, label_ids)

            total_loss += loss.item()
            total_acc  += acc
        
        acc  = total_acc/len(data_loader)
        loss = total_loss/len(data_loader)

        return acc, loss

    def test(self):
        self.model.eval()
        total_acc = 0
        total_loss = 0
        data_loader = self.val_data_loader

        with torch.no_grad():
            for (pair_token_ids, mask_ids, seg_ids, y) in tqdm(data_loader):
                pair_token_ids = pair_token_ids.to(self.device)
                mask_ids = mask_ids.to(self.device)
                seg_ids = seg_ids.to(self.device)
                labels = y.to(self.device)

                result = self.model(pair_token_ids,
                                        decoder_input_ids=seg_ids,
                                        attention_mask=mask_ids,
                                        labels=labels,
                                        return_dict=True)
            
                loss = result.loss
                logits = result.logits
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
 
                acc = self.flat_accuracy(logits, label_ids, val=True)

                total_loss += loss.item()
                total_acc  += acc
        
        acc  = total_acc/len(data_loader)
        loss = total_loss/len(data_loader)

        return acc, loss

    def execute(self):
        total_t0 = time.time()
        last_best = 0
        print("Training model..")

        train_df = load_utils.load_data(self.train_path)
        if self.num_classes == 2:
            train_df['gold_label'] = train_df['gold_label'].replace('contradiction', 'non-entailment')
            train_df['gold_label'] = train_df['gold_label'].replace('neutral', 'non-entailment')
            label_dict = {'entailment': 1, 'non-entailment': 0}
        else:
            label_dict = {'entailment': 2, 'contradiction': 0, 'neutral': 1}
        train_dataset = TransformerDatasetLoader(train_df, self.tokenizer, label_dict=label_dict, is_hypothesis_only=self.is_hypothesis_only)
        self.train_data_loader = train_dataset.get_data_loaders(self.batch_size)

        val_df = load_utils.load_data(self.val_path)
        val_df['gold_label'] = val_df['gold_label'].astype(int)
        val_dataset = TransformerDatasetLoader(val_df, self.tokenizer, is_hypothesis_only=self.is_hypothesis_only) # Validation is on RTE, hence there are 2 classes
        self.val_data_loader = val_dataset.get_data_loaders(self.batch_size)

        optimizer = AdamW(self.model.parameters(),
                lr = 3e-6,#lr = 4e-5, # args.learning_rate - default is 5e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
            )

        total_steps = len(self.train_data_loader) * self.epochs

        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 1, # Default value in run_glue.py
                                                    num_training_steps = total_steps)

        for epoch_i in range(0, self.epochs):
            train_acc, train_loss = self.train(optimizer, scheduler)
            val_acc, val_loss = self.test()

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
    parser.add_argument("--gradient_accumulation", help="Number of batches to accumulate gradients", type=int, default=0)
    parser.add_argument("--model_name", help="Name of the huggingface model or the path to the directory containing a pre-trained transformer", default="roberta-base")
    parser.add_argument("--num_classes", help="Number of output classes - RTE has 2, MNLI has 3", type=int, choices=[2, 3], default=2)
    parser.add_argument("--is_hypothesis_only", action='store_true')
    return parser.parse_args()

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print ("Created a path: %s"%(path))

if __name__ == '__main__':
    # Set numpy, tensorflow and python seeds for reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)

    args = parse_args()
    assert args.gradient_accumulation >= 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    save_path = f'{args.save_path}/'
    create_path(save_path)
    
    options = {}
    options['batch_size'] = args.batch_size
    options['device'] = device
    options['train_path'] = args.train_path
    options['val_path'] = args.val_path
    options['model_name'] = args.model_name
    options['save_path'] = args.save_path
    options['epochs'] = args.epochs
    options['num_classes'] = args.num_classes
    options['gradient_accumulation'] = args.gradient_accumulation
    options['is_hypothesis_only'] = args.is_hypothesis_only
    print(options)

    trainer = TransformerTrain(options)
    trainer.execute()