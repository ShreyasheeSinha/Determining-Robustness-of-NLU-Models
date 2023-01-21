import pandas as pd
import seaborn as sns
import string
import matplotlib.pyplot as plt

df = pd.read_csv('main_data.csv')

# df = pd.read_csv('prte_test.csv')

# df['combined_original'] = df['original_pair_premise'] + ' ' + df['original_pair_hypothesis']
# df['combined_modified'] = df['paraphrased_pair_premise'] + ' ' + df['paraphrased_pair_hypothesis']

def preprocess(a):
    table = str.maketrans(dict.fromkeys(string.punctuation))
    new_s = a.translate(table) 
    return new_s

def unigram_overlap(s1, s2):

    s1 = preprocess(s1)
    s2 = preprocess(s2)

    s1 = set(s1.lower().split()) 
    s2 = set(s2.lower().split())
    intersection = s1.intersection(s2)

    union = s2.union(s1)
        
    # Calculate Jaccard similarity score 
    # using length of intersection set divided by length of union set
    return float(len(intersection)) / len(union)

# df['jaccard'] = df.apply(lambda x: unigram_overlap(x.combined_original, x.combined_modified), axis=1)

# df['jaccard_p'] = df.apply(lambda x: unigram_overlap(x.original_pair_premise, x.paraphrased_pair_premise), axis=1)
# df['jaccard_h'] = df.apply(lambda x: unigram_overlap(x.original_pair_hypothesis, x.paraphrased_pair_hypothesis), axis=1)

# gpt3 = pd.read_csv('gpt_3_rte_test_paraphrased_preds.csv')
# gpt3.drop("Unnamed: 0", axis=1, inplace=True)

def get_original_id_match_for_gpt3preds(row):

    return f"{row['dataset']}_{row['corpus_sent_id']}_{row['task']}"
    
def get_para_id_for_gpt3preds(row):
    ext = ''
    if not pd.isnull(row['s1_para_id']) and not pd.isnull(row['s2_para_id']):
        ext = '_ph'
    elif not pd.isnull(row['s1_para_id']):
        ext = f'_p'
    elif not pd.isnull(row['s2_para_id']):
        ext = f"_h"

    return f"{row['dataset']}_{row['corpus_sent_id']}_{row['task']}{ext}"

# gpt3['original_id'] = gpt3.apply(lambda row: get_original_id_match_for_gpt3preds(row), axis=1)
# gpt3['modified_id'] = gpt3.apply(lambda row: get_para_id_for_gpt3preds(row), axis=1)

# bert_orig = pd.read_csv('bert_large_3_class_rte_test_predictions.csv')
# bert_orig.drop("Unnamed: 0", axis=1, inplace=True)
# bert_orig['original_id'] = bert_orig.apply(lambda row: get_original_id_match_for_gpt3preds(row), axis=1)
# bert_orig['modified_id'] = bert_orig.apply(lambda row: get_original_id_match_for_gpt3preds(row), axis=1)

# bert_para = pd.read_csv('bert_large_3_class_rte_test_p_predictions.csv')
# bert_para.drop("Unnamed: 0", axis=1, inplace=True)
# bert_para['original_id'] = bert_para.apply(lambda row: get_para_id_for_gpt3preds(row), axis=1)
# bert_para['modified_id'] = bert_para.apply(lambda row: get_para_id_for_gpt3preds(row), axis=1)

# bilstm_orig = pd.read_csv('bilstm_3_class_rte_test_predictions.csv')
# bilstm_orig.drop("Unnamed: 0", axis=1, inplace=True)
# bilstm_orig['original_id'] = bilstm_orig.apply(lambda row: get_original_id_match_for_gpt3preds(row), axis=1)
# bilstm_orig['modified_id'] = bilstm_orig.apply(lambda row: get_original_id_match_for_gpt3preds(row), axis=1)

# bilstm_para = pd.read_csv('bilstm_3_class_rte_test_p_predictions.csv')
# bilstm_para.drop("Unnamed: 0", axis=1, inplace=True)
# bilstm_para['original_id'] = bilstm_para.apply(lambda row: get_para_id_for_gpt3preds(row), axis=1)
# bilstm_para['modified_id'] = bilstm_para.apply(lambda row: get_para_id_for_gpt3preds(row), axis=1)

# cbow_orig = pd.read_csv('cbow_3_class_rte_test_predictions.csv')
# cbow_orig.drop("Unnamed: 0", axis=1, inplace=True)
# cbow_orig['original_id'] = cbow_orig.apply(lambda row: get_original_id_match_for_gpt3preds(row), axis=1)
# cbow_orig['modified_id'] = cbow_orig.apply(lambda row: get_original_id_match_for_gpt3preds(row), axis=1)

# cbow_para = pd.read_csv('cbow_3_class_rte_test_p_predictions.csv')
# cbow_para.drop("Unnamed: 0", axis=1, inplace=True)
# cbow_para['original_id'] = cbow_para.apply(lambda row: get_para_id_for_gpt3preds(row), axis=1)
# cbow_para['modified_id'] = cbow_para.apply(lambda row: get_para_id_for_gpt3preds(row), axis=1)

# roberta_orig = pd.read_csv('roberta_large_3_class_rte_test_predictions.csv')
# roberta_orig.drop("Unnamed: 0", axis=1, inplace=True)
# roberta_orig['original_id'] = roberta_orig.apply(lambda row: get_original_id_match_for_gpt3preds(row), axis=1)
# roberta_orig['modified_id'] = roberta_orig.apply(lambda row: get_original_id_match_for_gpt3preds(row), axis=1)

# roberta_para = pd.read_csv('roberta_large_3_class_rte_test_p_predictions.csv')
# roberta_para.drop("Unnamed: 0", axis=1, inplace=True)
# roberta_para['original_id'] = roberta_para.apply(lambda row: get_para_id_for_gpt3preds(row), axis=1)
# roberta_para['modified_id'] = roberta_para.apply(lambda row: get_para_id_for_gpt3preds(row), axis=1)

# roberta_orig = roberta_orig[['sentence1', 'sentence2', 'original_id', 'modified_id', 'prediction', 'gold_label']]
# roberta_para = roberta_para[['sentence1', 'sentence2', 'original_id', 'modified_id', 'prediction', 'gold_label']]
# roberta = pd.concat([roberta_orig, roberta_para], ignore_index=True)

# bert_orig = bert_orig[['sentence1', 'sentence2', 'original_id', 'modified_id', 'prediction', 'gold_label']]
# bert_para = bert_para[['sentence1', 'sentence2', 'original_id', 'modified_id', 'prediction', 'gold_label']]
# bert = pd.concat([bert_orig, bert_para], ignore_index=True)

# bilstm_orig = bilstm_orig[['sentence1', 'sentence2', 'original_id', 'modified_id', 'prediction', 'gold_label']]
# bilstm_para = bilstm_para[['sentence1', 'sentence2', 'original_id', 'modified_id', 'prediction', 'gold_label']]
# bilstm = pd.concat([bilstm_orig, bilstm_para], ignore_index=True)

# cbow_orig = cbow_orig[['sentence1', 'sentence2', 'original_id', 'modified_id', 'prediction', 'gold_label']]
# cbow_para = cbow_para[['sentence1', 'sentence2', 'original_id', 'modified_id', 'prediction', 'gold_label']]
# cbow = pd.concat([cbow_orig, cbow_para], ignore_index=True)

# gpt3 = gpt3[['sentence1', 'sentence2', 'original_id', 'modified_id', 'prediction', 'gold_label']]

def keep_data(df, dataset):
    meta = list(dataset.paraphrased_pair_id) + list(dataset.original_pair_id)
    df = df[df.modified_id.isin(meta)]

    return df

# bert = keep_data(bert, df)
# gpt3 = keep_data(gpt3, df)
# cbow = keep_data(cbow, df)
# roberta = keep_data(roberta, df)
# bilstm = keep_data(bilstm, df)

def create_map_for_model_predictions(df):
    dic = {}
    for idx, row in df.iterrows():
        dic[row['modified_id']] = row['prediction']

    return dic

# bert_preds = create_map_for_model_predictions(bert)
# gpt3_preds = create_map_for_model_predictions(gpt3)
# cbow_preds = create_map_for_model_predictions(cbow)
# roberta_preds = create_map_for_model_predictions(roberta)
# bilstm_preds = create_map_for_model_predictions(bilstm)

# add_bert, add_roberta, add_cbow, add_bilstm, add_gpt = [], [], [], [], []
# for idx, row in df.iterrows():
#     add_bert.append(bert_preds[row.paraphrased_pair_id])
#     add_roberta.append(roberta_preds[row.paraphrased_pair_id])
#     add_cbow.append(cbow_preds[row.paraphrased_pair_id])
#     add_bilstm.append(bilstm_preds[row.paraphrased_pair_id])
#     add_gpt.append(gpt3_preds[row.paraphrased_pair_id])

# df['bert_para_pred'] = add_bert
# df['roberta_para_pred'] = add_roberta
# df['cbow_para_pred'] = add_cbow
# df['bilstm_para_pred'] = add_bilstm
# df['gpt_para_pred'] = add_gpt

# add_bert, add_roberta, add_cbow, add_bilstm, add_gpt = [], [], [], [], []
# for idx, row in df.iterrows():
#     add_bert.append(bert_preds[row.original_pair_id])
#     add_roberta.append(roberta_preds[row.original_pair_id])
#     add_cbow.append(cbow_preds[row.original_pair_id])
#     add_bilstm.append(bilstm_preds[row.original_pair_id])
#     add_gpt.append(gpt3_preds[row.original_pair_id])

# df['bert_orig_pred'] = add_bert
# df['roberta_orig_pred'] = add_roberta
# df['cbow_orig_pred'] = add_cbow
# df['bilstm_orig_pred'] = add_bilstm
# df['gpt_orig_pred'] = add_gpt

# df.to_csv('main_data.csv', index=False)

def pred_change(row, model):
    if int(row[f'{model}_orig_pred']) == (row[f'{model}_para_pred']):
        return 'No change'
    else:
        return 'Change'

df['bert_change'] = df.apply(lambda row: pred_change(row, 'bert'), axis=1)
df['roberta_change'] = df.apply(lambda row: pred_change(row, 'roberta'), axis=1)
df['gpt_change'] = df.apply(lambda row: pred_change(row, 'gpt'), axis=1)
df['bilstm_change'] = df.apply(lambda row: pred_change(row, 'bilstm'), axis=1)
df['cbow_change'] = df.apply(lambda row: pred_change(row, 'cbow'), axis=1)

def get_change(df, model):
    entail = df[df.gold_label == 'entailed']
    entail = entail.reset_index()
    ent_change = entail[entail[f"{model}_change"] == 'Change']
    print(f"{model} predictions change when gold is entailed: {ent_change.shape[0]}/{entail.shape[0]} = {ent_change.shape[0]/entail.shape[0]}")

    not_entail = df[df.gold_label == 'not_entailed']
    not_entail = not_entail.reset_index()
    not_ent_change = not_entail[not_entail[f"{model}_change"] == 'Change']
    print(f"{model} predictions change when gold is not_entailed: {not_ent_change.shape[0]}/{not_entail.shape[0]} = {not_ent_change.shape[0]/not_entail.shape[0]}")

get_change(df, 'bert')
get_change(df, 'roberta')
get_change(df, 'cbow')
get_change(df, 'bilstm')
get_change(df, 'gpt')



def get_para_change(df, model):
    h_para = df[df.paraphrased_pair_id.str.endswith('_h')]
    h_para = h_para.reset_index()
    h_para_change = h_para[h_para[f"{model}_change"] == 'Change']
    print(f"{model} predictions change when only hyp is paraphrased: {h_para_change.shape[0]}/{h_para.shape[0]} = {h_para_change.shape[0]/h_para.shape[0]}")

    p_para = df[df.paraphrased_pair_id.str.endswith('_p')]
    p_para = p_para.reset_index()
    p_para_change = p_para[p_para[f"{model}_change"] == 'Change']
    print(f"{model} predictions change when only premise is paraphrased: {p_para_change.shape[0]}/{p_para.shape[0]} = {p_para_change.shape[0]/p_para.shape[0]}")

    ph_para = df[df.paraphrased_pair_id.str.endswith('_ph')]
    ph_para = ph_para.reset_index()
    ph_para_change = ph_para[ph_para[f"{model}_change"] == 'Change']
    print(f"{model} predictions change when both are paraphrased: {ph_para_change.shape[0]}/{ph_para.shape[0]} = {ph_para_change.shape[0]/ph_para.shape[0]}")

get_para_change(df, 'bert')
get_para_change(df, 'roberta')
get_para_change(df, 'cbow')
get_para_change(df, 'bilstm')
get_para_change(df, 'gpt')

cbow_bilstm_both_inconsistent = df[(df.cbow_change == 'Change') & (df.bilstm_change == 'Change')]


# bert = sns.histplot(data=df, x="jaccard", hue="bert_change", multiple="stack", kde=True, stat='density', bins=30)
# fig = bert.get_figure()
# fig.savefig('berth.png')
# fig.clf()

# roberta = sns.histplot(data=df, x="jaccard", hue="roberta_change", multiple="stack", kde=True, stat='density', bins=30)
# fig = roberta.get_figure()
# fig.savefig('robertah.png')
# fig.clf()

# cbow = sns.histplot(data=df, x="jaccard", hue="cbow_change", multiple="stack", kde=True, stat='density', bins=30)
# fig = cbow.get_figure()
# fig.savefig('cbowh.png')
# fig.clf()

# bilstm = sns.histplot(data=df, x="jaccard", hue="bilstm_change", multiple="stack", kde=True, stat='density', bins=30)
# fig = bilstm.get_figure()
# fig.savefig('bilstmh.png')
# fig.clf()

# gpt = sns.histplot(data=df, x="jaccard", hue="gpt_change", multiple="stack", kde=True, stat='density', bins=30)

# fig = gpt.get_figure()
# fig.savefig('gpth.png')
# fig.clf()






# bert = sns.kdeplot(data=df, x="jaccard", hue="bert_change", multiple="stack", common_norm=False)
# fig = bert.get_figure()
# fig.savefig('bert.png')
# fig.clf()

# roberta = sns.kdeplot(data=df, x="jaccard", hue="roberta_change", multiple="stack", common_norm=False)
# fig = roberta.get_figure()
# fig.savefig('roberta.png')
# fig.clf()

# cbow = sns.kdeplot(data=df, x="jaccard", hue="cbow_change", multiple="stack", common_norm=False)
# fig = cbow.get_figure()
# fig.savefig('cbow.png')
# fig.clf()

# bilstm = sns.kdeplot(data=df, x="jaccard", hue="bilstm_change", multiple="stack", common_norm=False)
# fig = bilstm.get_figure()
# fig.savefig('bilstm.png')
# fig.clf()

# gpt = sns.kdeplot(data=df, x="jaccard", hue="gpt_change", multiple="stack", common_norm=False)
# fig = gpt.get_figure()
# fig.savefig('gpt.png')
# fig.clf()