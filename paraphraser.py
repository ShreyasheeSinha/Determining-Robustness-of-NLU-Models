from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import util.load_utils as load_utils
from tqdm import tqdm
tqdm.pandas()
import string
import torch
import numpy as np
import argparse

class Paraphraser:

    def __init__(self, options):
        self.device = options['device']
        self.data_path = options['data_path']
        self.save_path = options['save_path']
        self.jaccard_score = options['jaccard_score']
        self.model_name = 'Vamsi/T5_Paraphrase_Paws'
        self.model, self.tokenizer = self.load_paraphraser()

    def load_paraphraser(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)  
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        model.to(self.device)
        return model, tokenizer

    def load_data(self):
        return load_utils.load_data(self.data_path)

    def get_sentence_counts(self, data):
        datasets = data['dataset'].unique()
        print("Total length of the dataset: ", len(data))
        for dataset in datasets:
            print(dataset, len(data[data['dataset'] == dataset]))

    def get_paraphrases_counts(self, data):
        total_premise_paraphrases = 0
        total_hypothesis_paraphrases = 0
        datasets = data['dataset'].unique()
        for dataset in datasets:
            temp = data[data['dataset'] == dataset]
            premise_paraphrases = temp['sentence1dash'].str.len().sum()
            hypothesis_paraphrases = temp['sentence2dash'].str.len().sum()
            print(dataset, "Premise paraphrases:", premise_paraphrases, "Hypothesis paraphrases:", hypothesis_paraphrases)
            total_premise_paraphrases += premise_paraphrases
            total_hypothesis_paraphrases += hypothesis_paraphrases
        total_paraphrases = total_premise_paraphrases + total_hypothesis_paraphrases
        print("Total premise paraphrases:", total_premise_paraphrases, "Total hypothesis paraphrases:", total_hypothesis_paraphrases)
        print("Total paraphrases:", total_paraphrases)

    def jaccard_similarity(self, s1, s2): 
        s1 = self.preprocess(s1)
        s2 = self.preprocess(s2)
        
        s1 = set(s1.lower().split()) 
        s2 = set(s2.lower().split())
        intersection = s1.intersection(s2)

        union = s2.union(s1)
            
        # Calculate Jaccard similarity score 
        # using length of intersection set divided by length of union set
        return float(len(intersection)) / len(union)    
    
    def preprocess(self, a):
        table = str.maketrans(dict.fromkeys(string.punctuation))
        new_s = a.translate(table) 
        return new_s

    def best_paraphrase(self, sentence, paraphrases):
        # min_score = None
        best_paraphrases = []
        for paraphrase in paraphrases:
            if paraphrase.lower() == sentence.lower():
                continue
            jaccard_score = self.jaccard_similarity(sentence, paraphrase)
            if jaccard_score <= self.jaccard_score:
                details = {'paraphrase': paraphrase, 'jaccard_score': jaccard_score}
                best_paraphrases.append(details)
        return best_paraphrases

    def generate_paraphrase(self, sentence):
        # text = "paraphrase: " + sentence + " </s>"

        encoding = self.tokenizer.encode_plus(sentence, padding=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=256,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=3
        )

        paraphrases = []
        for output in outputs:
            line = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            paraphrases.append(line)

        return self.best_paraphrase(sentence, paraphrases)

    def execute(self):
        data = self.load_data()
        self.get_sentence_counts(data)
        data['sentence1dash'] = data['sentence1'].progress_apply(self.generate_paraphrase)
        data['sentence2dash'] = data['sentence2'].progress_apply(self.generate_paraphrase)
        self.get_paraphrases_counts(data)
        load_utils.save_data(data, self.save_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the dataset jsonl file", default="./RTE_dev.jsonl")
    parser.add_argument("--save_path", help="Path with file name where to save the paraphrased dataset", default="./RTE_dev_paraphrased.jsonl") # TODO: Add proper path
    parser.add_argument("--jaccard_score", help="Path with file name where to save the paraphrased dataset", default=0.75, type=float)
    return parser.parse_args()

if __name__ == '__main__':
    # Set numpy, tensorflow and python seeds for reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)

    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    options = {}
    options['jaccard_score'] = args.jaccard_score
    options['device'] = device
    options['data_path'] = args.data_path
    options['save_path'] = args.save_path
    print(options)

    paraphraser = Paraphraser(options)
    paraphraser.execute()