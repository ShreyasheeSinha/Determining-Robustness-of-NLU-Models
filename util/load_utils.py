import numpy as np
import json
import pandas as pd

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)

def read_jsonl_file(path):
    with open(path, 'r') as json_file:
        json_list = list(json_file)

    results = []
    for json_str in json_list:
        results.append(json.loads(json_str))
    return results

def load_data(path):
    return pd.DataFrame(read_jsonl_file(path))

def save_data(df, path):
    json_data = df.to_json(orient='records', lines=True)
    with open(path, 'w') as f:
        json.dump(json_data, f)
