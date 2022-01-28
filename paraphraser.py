from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import util.load_utils as load_utils
from datasets import load_metric
from tqdm import tqdm
tqdm.pandas()
import string

class Paraphraser:

    def __init__(self, options):
        self.device = options['device']
        self.data_path = options['data_path']
        self.save_path = options['save_path']
        self.model_name = 'Vamsi/T5_Paraphrase_Paws'
        self.metric = load_metric("bertscore")
        self.model, self.tokenizer = self.load_paraphraser()

    def load_paraphraser(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)  
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        model.to(self.device)
        return model, tokenizer

    def load_data(self):
        return load_utils.load_data(self.data_path)

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
            if paraphrase == sentence:
                continue
            results = self.metric.compute(predictions=[paraphrase], references=[sentence], lang="en")
            bert_score = results["f1"][0]
            if bert_score >= 0.7:
                jaccard_score = self.jaccard_similarity(sentence, paraphrase)
                if jaccard_score <= 0.75:
                    # TODO: Decide whether to return single or multiple phrases
                    best_paraphrases.append(paraphrase)
                    # if min_score == None or jaccard_score < min_score:
                    #     min_score = jaccard_score
                    #     best_paraphrase = paraphrase
                    # else:
                    #     print("Jaccard above 0.75:", bert_score, paraphrase, jaccard_score)
            # else:
            #     print("BERT below 0.7", bert_score, paraphrase)
        # print("Best paraphrase:", best_paraphrase, min_score)
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
            num_return_sequences=10
        )

        paraphrases = []
        for output in outputs:
            line = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            paraphrases.append(line)

        return self.best_paraphrase(sentence, paraphrases)

    def execute(self):
        data = self.load_data()
        data['sentence_1_dash'] = data['sentence1'].progress_apply(self.generate_paraphrase)
        data['sentence_2_dash'] = data['sentence2'].progress_apply(self.generate_paraphrase)
        load_utils.save_data(data, self.save_path)
