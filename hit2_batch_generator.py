import util.load_utils as load_utils
import pandas as pd
import numpy as np
import os

def generate_hit_data(paraphrased_data):
    hit_data = pd.DataFrame(columns=["sent1_1_para_id", "sent1_1", "sent1_2_para_id", "sent1_2",
    "sent2_1_para_id", "sent2_1", "sent2_2_para_id", "sent2_2", "sent3_1_para_id", "sent3_1", "sent3_2_para_id", "sent3_2"])
    for i, g in paraphrased_data.groupby(paraphrased_data.index // 3):
        values = {
            "sent1_1_para_id": None,
            "sent1_1": None,
            "sent1_2_para_id": None,
            "sent1_2": None,
            "sent2_1_para_id": None,
            "sent2_1": None,
            "sent2_2_para_id": None,
            "sent2_2": None,
            "sent3_1_para_id": None,
            "sent3_1": None,
            "sent3_2_para_id": None,
            "sent3_2": None
        }
        for id, (index, row) in enumerate(g.iterrows()):
            sent1_para_id = "sent" + str(id + 1) + "_1_para_id"
            sent2_para_id = "sent" + str(id + 1) + "_2_para_id"
            sent1 = "sent" + str(id + 1) + "_1"
            sent2 = "sent" + str(id + 1) + "_2"
            values[sent1_para_id] = row["s1_para_id"]
            values[sent2_para_id] = row["s2_para_id"]
            values[sent1] = row["sentence1"]
            values[sent2] = row["sentence2"]
        hit_data = hit_data.append(values, ignore_index=True)

    return hit_data

def save_batches(data):
    batch_num = 1
    for i in range(0, len(data), 500):
        name = "data_batch_" + str(batch_num) + ".csv"
        full_path = os.path.join("data/HIT2_batches", name)
        batch_num += 1
        print("Saving to:", full_path)
        data[i: i+500].to_csv(full_path, index=False)

test_data = load_utils.load_data("data/RTE_data/RTE_test_paraphrased.jsonl")
dev_data = load_utils.load_data("data/RTE_data/RTE_dev_paraphrased.jsonl")
test_data.fillna(value="", inplace=True)
dev_data.fillna(value="", inplace=True)
paraphrased_pairs_test = test_data[test_data["silver_label"] != ""]
paraphrased_pairs_dev = dev_data[dev_data["silver_label"] != ""]
paraphrased_data = pd.concat([paraphrased_pairs_dev, paraphrased_pairs_test], ignore_index=True)
paraphrased_data = paraphrased_data.sample(frac=1.0, random_state=42, ignore_index=True)
hit_data = generate_hit_data(paraphrased_data)
save_batches(hit_data)