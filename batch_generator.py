import pandas as pd
import util.load_utils as load_utils
import os
import argparse

def get_paraphrase(row, index):
    try:
        return row[index]['paraphrase']
    except IndexError:
        return None

def get_unique_paraphrases(para_list):
    unique_list = []
    unique_paras = set()
    for para in para_list:
        curr_para = para['paraphrase']
        curr_jaccard = para['jaccard_score']
        if curr_para.strip().lower() not in unique_paras:
            unique_paras.add(curr_para.strip().lower())
            curr_para_dict = {
                'paraphrase': curr_para,
                'jaccard_score': curr_jaccard
            }
            unique_list.append(curr_para_dict)
    if len(para_list) != len(unique_list):
        print("Before removing dups:", len(para_list))
        print("After removing dups:", len(unique_list))
    return unique_list

def remove_duplicate_paraphrases(data):
    data['sentence1dash'] = data['sentence1dash'].apply(lambda val: get_unique_paraphrases(val))
    data['sentence2dash'] = data['sentence2dash'].apply(lambda val: get_unique_paraphrases(val))
    return data

def get_data_frame_given_sentence_len(data, len):
    sentence1_data = data[data['sentence1_len'] == len]
    sentence2_data = data[data['sentence2_len'] == len]
    sentence1_data['origin_id'] = sentence1_data['dataset'] + "_s1_" + sentence1_data['corpus_sent_id'].astype(str)
    sentence2_data['origin_id'] = sentence2_data['dataset'] + "_s2_" + sentence2_data['corpus_sent_id'].astype(str)
    for i in range(3):
        column_name = 'paraphrase' + str(i)
        sentence1_data[column_name] = sentence1_data['sentence1dash'].apply(lambda t: get_paraphrase(t, i))
        sentence2_data[column_name] = sentence2_data['sentence2dash'].apply(lambda t: get_paraphrase(t, i))
    sentence1_data_modified = sentence1_data[['origin_id', 'sentence1', 'paraphrase0', 'paraphrase1', 'paraphrase2', 'task']]
    sentence2_data_modified = sentence2_data[['origin_id', 'sentence2', 'paraphrase0', 'paraphrase1', 'paraphrase2', 'task']]
    sentence1_data_modified = sentence1_data_modified.rename(columns={"sentence1": "sentence"})
    sentence2_data_modified = sentence2_data_modified.rename(columns={"sentence2": "sentence"})
    concat_sentence_data = pd.concat([sentence1_data_modified, sentence2_data_modified])
    return concat_sentence_data

def save_batches(data, prefix, path):
    batch_num = 1
    for i in range(0, len(data), 500):
        name = prefix + "_data_batch_" + str(batch_num) + ".csv"
        full_path = os.path.join(path, name)
        batch_num += 1
        print("Saving to:", full_path)
        data[i: i+500].to_csv(full_path, index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the dataset jsonl file", default="./data/paraphrased_data/jaccard_score_0.75/RTE_dev_paraphrased.jsonl")
    parser.add_argument("--save_folder_path", help="Path to the folder where the batches will be saved", default="./data/paraphrased_data/jaccard_score_0.75/batches/RTE_dev")
    return parser.parse_args()

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created a path: %s"%(path))

if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    save_folder_path = args.save_folder_path
    create_path(save_folder_path)

    data = load_utils.load_data(data_path)
    data = remove_duplicate_paraphrases(data)
    data['sentence1_len'] = data['sentence1dash'].str.len()
    data['sentence2_len'] = data['sentence2dash'].str.len()

    three_paraphrase_data = get_data_frame_given_sentence_len(data, 3).reset_index()
    two_paraphrase_data = get_data_frame_given_sentence_len(data, 2).reset_index()
    one_paraphrase_data = get_data_frame_given_sentence_len(data, 1).reset_index()
    
    full_path = os.path.join(save_folder_path, "three_paraphrase_data.csv")
    three_paraphrase_data.to_csv(full_path, index=False)
    full_path = os.path.join(save_folder_path, "two_paraphrase_data.csv")
    two_paraphrase_data.to_csv(full_path, index=False)
    full_path = os.path.join(save_folder_path, "one_paraphrase_data.csv")
    one_paraphrase_data.to_csv(full_path, index=False)

    three_paraphrase_data_len = len(three_paraphrase_data)
    two_paraphrase_data_len = len(two_paraphrase_data)
    one_paraphrase_data_len = len(one_paraphrase_data)

    one_paraphrase_data_to_merge = one_paraphrase_data[:three_paraphrase_data_len]
    three_paraphrase_data.rename(columns={"sentence": "sentence1", "origin_id": "sen1_origin_id", 
    "paraphrase0": "sen1_paraphrase0",
    "paraphrase1": "sen1_paraphrase1",
    "paraphrase2": "sen1_paraphrase2",
    "task": "sen1_task"}, inplace=True)
    one_paraphrase_data_to_merge.drop(columns=["paraphrase1", "paraphrase2"], inplace=True)
    one_paraphrase_data_to_merge.rename(columns={"sentence": "sentence2", "origin_id": "sen2_origin_id",
    "paraphrase0": "sen2_paraphrase0",
    "task": "sen2_task"}, inplace=True)
    three_one_merged_data = pd.concat([three_paraphrase_data, one_paraphrase_data_to_merge], axis=1)
    three_one_merged_data.drop(columns=['index'], inplace=True)

    one_paraphrase_data_leftover = one_paraphrase_data[three_paraphrase_data_len:]
    two_paraphrase_data.drop(columns=["paraphrase2"], inplace=True)
    one_paraphrase_data_leftover.drop(columns=['paraphrase1', 'paraphrase2'], inplace=True)
    two_two_merged_data = pd.DataFrame(columns=["sen1_origin_id", "sentence1", "sen1_paraphrase0", "sen1_paraphrase1", "sen1_task",
    "sen2_origin_id", "sentence2", "sen2_paraphrase0", "sen2_paraphrase1", "sen2_task"])
    one_merged_data = pd.DataFrame(columns=["sen1_origin_id", "sentence1", "sen1_paraphrase0", "sen1_task", 
    "sen2_origin_id", "sentence2", "sen2_paraphrase0", "sen2_task",
    "sen3_origin_id", "sentence3", "sen3_paraphrase0", "sen3_task",
    "sen4_origin_id", "sentence4", "sen4_paraphrase0", "sen4_task"])

    # Iterate every two rows
    for i, g in two_paraphrase_data.groupby(two_paraphrase_data.index // 2):
        values = {
            "sentence1": None, 
            "sen1_origin_id": None, 
            "sen1_paraphrase0": None, 
            "sen1_paraphrase1": None, 
            "sen1_task": None,
            "sentence2": None, 
            "sen2_origin_id": None, 
            "sen2_paraphrase0": None, 
            "sen2_paraphrase1": None, 
            "sen2_task": None
        }
        for id, (index, row) in enumerate(g.iterrows()):
            sentence = "sentence" + str(id + 1)
            origin_id = "sen" + str(id + 1) + "_origin_id"
            paraphrase0 = "sen" + str(id + 1) + "_paraphrase0"
            paraphrase1 = "sen" + str(id + 1) + "_paraphrase1"
            task = "sen" + str(id + 1) + "_task"
            values[sentence] = row["sentence"]
            values[origin_id] = row["origin_id"]
            values[paraphrase0] = row["paraphrase0"]
            values[paraphrase1] = row["paraphrase1"]
            values[task] = row["task"]
        two_two_merged_data = two_two_merged_data.append(values, ignore_index=True)

    # Iterate every four rows
    for i, g in one_paraphrase_data_leftover.groupby(one_paraphrase_data_leftover.index // 4):
        values = {
            "sentence1": None, 
            "sen1_origin_id": None, 
            "sen1_paraphrase0": None, 
            "sen1_task": None,
            "sentence2": None, 
            "sen2_origin_id": None, 
            "sen2_paraphrase0": None, 
            "sen2_task": None,
            "sentence3": None, 
            "sen3_origin_id": None, 
            "sen3_paraphrase0": None, 
            "sen3_task": None,
            "sentence4": None, 
            "sen4_origin_id": None, 
            "sen4_paraphrase0": None, 
            "sen4_task": None
        }
        for id, (index, row) in enumerate(g.iterrows()):
            sentence = "sentence" + str(id + 1)
            origin_id = "sen" + str(id + 1) + "_origin_id"
            paraphrase0 = "sen" + str(id + 1) + "_paraphrase0"
            task = "sen" + str(id + 1) + "_task"
            values[sentence] = row["sentence"]
            values[origin_id] = row["origin_id"]
            values[paraphrase0] = row["paraphrase0"]
            values[task] = row["task"]
        one_merged_data = one_merged_data.append(values, ignore_index=True)

    save_batches(three_one_merged_data, "3_1", save_folder_path)
    save_batches(two_two_merged_data, "2_2", save_folder_path)
    save_batches(one_merged_data, "1", save_folder_path)