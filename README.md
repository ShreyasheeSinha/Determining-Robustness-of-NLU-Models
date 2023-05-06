# Determining-Robustness-of-NLU-Models

This file is best viewed in a Markdown reader (eg. https://jbt.github.io/markdown-editor/)

## Overview

This repository contains the code and data we used to test how NLP models perform when subjected to meaning preserving yet lexically different sentences for the task of sentence entailment. Currently, the repository supports training and testing a BiLSTM, CBOW, BERT and RoBERTa models. It has support to run inference on GPT-3 using OpenAI's API. It also has support to generate paraphrases for a given entailment dataset.

## Setup

Follow these steps to setup your environment:

1. [Download and install Conda](http://https://conda.io/projects/conda/en/latest/user-guide/install/index.html "Download and install Conda")

2. Create a Conda environment with Python 3.6: `conda create -n <env_name> Python=3.6`

3. Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use this code: `conda activate <env_name>`

4. Install the requirements: `pip install -r requirements.txt`.


## Code Organisation


```
Determining-Robustness-of-NLU-Models
└── data/
└── models/
  ├── bilstm.py
  ├── cbow.py
  └── transformer.py
└── util/
  ├── dataset_loader.py
  ├── load_utils.py
  ├── model_utils.py
  ├── transformer_dataset_loader.py
  └── vocab.py
├── paraphraser.py
├── README.md
├── transformer_test.py
├── transformer_train.py
├── test.py
└── train.py
```

## Paraphraser

The `paraphraser.py` file is responsible for generating the paraphrased data. To do this, it uses the [HuggingFace's T5 paraphraser](https://huggingface.co/Vamsi/T5_Paraphrase_Paws). In order to ensure lexical diversity, we measure the Jaccard Similarity score between the paraphrased sentences and original sentence.

To run the paraphraser, the following options are available:

1. `--data_path`: The path to the `.jsonl` file containing the data you wish to paraphrase. It expects the data to contain two columns - `sentence1` and `sentence2` - which need to be paraphrased.
2. `--save_path`: The path where to save the paraphrased data.
3. `--jaccard_score`: The threshold Jaccard Score to be used. The default score is 0.75.

Sample command:

`python3 paraphraser.py --data_path RTE_test.jsonl --save_path RTE_test_paraphrased.jsonl`


## Training and testing - BiLSTM and CBOW

To train the BiLSTM and CBOW models, you would need to use the `train.py` file. The `models/bilstm.py` and `models/cbow.py` files contain the implementation of the BiLSTM and CBOW models respectively. The following options are available for training the models:

1. `--model_type`: The model type you wish to train. There are two options available - `bilstm` and `cbow`. The default model is `bilstm`.
2. `--save_path`: Path to the directory to save the model. The path is created if it does not exist. The default path is `./saved_model`.
3. `--train_path`: Path to the file which has the training data. The default path is `./data/multinli_1.0/multinli_1.0_train.jsonl`.
4. `--val_path`: Path to the file which has the validation data. The default path is `./data/multinli_1.0/multinli_1.0_dev_matched.jsonl`.
5. `--batch_size`: The batch size for the model. Default value is 32.
6. `--emb_path`: Path to the GloVe embeddings. Default path is `/data/glove.840B.300d.txt`.
7. `--epochs`: Number of epochs to train the model.
8. `--model_name`: The suffix given to your model. This will be appended to the `model_type`. __This parameter is required.__
9. `--hidden_size`: The number of hidden units in the LSTM.
10. `--stacked_layers`: Number of stacked LSTM units.
11. `--seq_len`: The maximum sequence length allowed. The default value is 50.
12. `--vocab_size`: The vocab size to be used for the model. The default value is 50,000.
13. `--num_classes`: The number of training classes. This repo involves training on the MNLI dataset with 3 classes - entailment, neutral and contradiction -  and validating/testing on the RTE dataset with two classes - entailment, non-entailment. This parameters specifies the number of training classes present in your training data. The default value is 2.
14. `--is_hypothesis_only`: Specifies if the model to be trained on the hypothesis only.

Sample training commands are:

`python3 train.py --save_path saved_model/cbow/ --val_path RTE_dev.jsonl --model_name 3_class --num_classes 3 --model_type cbow`

`python3 train.py --save_path saved_model/bilstm/ --val_path RTE_dev.jsonl --model_name 3_class --num_classes 3`

The model is saved based on the best validation accuracy achieved till the current epoch.

To test the BiLSTM and CBOW models, you would need to use the `test.py` file. The following options are available for testing the models:

1. `--model_type`: The model type you wish to test. There are two options available - `bilstm` and `cbow`. The default model is `bilstm`.
2. `--save_path`: Path to the directory where the trained model is saved. The default path is `./saved_model`.
3. `--test_path`: Path to the file which has the test data. The default path is `./data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl`.
4. `--batch_size`: The batch size for the model. Default value is 32.
5. `--emb_path`: Path to the GloVe embeddings. Default path is `/data/glove.840B.300d.txt`.
6. `--model_name`: The suffix given to your model. This will be appended to the `model_type` to help find the name of the saved model. __This parameter is required.__
7. `--predictions_save_path`: The file where the model predictions will be saved.

Sample testing commands are:

`python3 test.py --save_path saved_model/bilstm/ --test_path RTE_test.jsonl --model_name 3_class`

`python3 test.py --save_path saved_model/cbow/ --test_path RTE_test.jsonl --model_name 3_class --model_type cbow`

### Training and testing - transformers

To train transformer based, you would need to use the `transformer_train.py` file. The `models/transformer.py` contains the code to load the model you wish from HuggingFace. __Currently only `roberta` and `bert` family models are supported.__ The following options are available to fine-tune the models:

1. `--save_path`: Path to the directory to save the model. The path is created if it does not exist. The default path is `./saved_model`.
2. `--train_path`: Path to the file which has the training data. The default path is `./data/multinli_1.0/multinli_1.0_train.jsonl`.
3. `--val_path`: Path to the file which has the validation data. The default path is `./data/multinli_1.0/multinli_1.0_dev_matched.jsonl`.
4. `--batch_size`: The batch size for the model. Default value is 32.
5. `--epochs`: Number of epochs to train the model.
6. `--gradient_accumulation`: Number of batches to accumulate gradients. This was added to support simulation of large batches in the case the training GPU does not have a large memory. The default value is 0.
7. `--model_name`: The name or path to the directory of the model you wish to train. The default value is `roberta-base`.
8. `--num_classes`: The number of training classes. This repo involves training on the MNLI dataset with 3 classes - entailment, neutral and contradiction -  and validating/testing on the RTE dataset with two classes - entailment, non-entailment. This parameters specifies the number of training classes present in your training data. The default value is 2.
9. `--is_hypothesis_only`: Specifies if the model to be trained on the hypothesis only.

Sample training commands are:

`python3 transformer_train.py --batch_size 3 --val_path RTE_dev.jsonl --epochs 5 --num_classes 3 --save_path saved_model/roberta_3_class --model_name roberta-large --gradient_accumulation 8`

To test the trained model, you would need to use the `transformer_test.py` file. The following options are available to test the models:

1. `--test_path`: Path to the file which has the testing data. The default path is `./data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl`.
2. `--batch_size`: The batch size for the model. Default value is 32.
3. `--model_name`: The name or path to the directory of the model you wish to test. The default value is `roberta-large-mnli`.
4. `--is_hypothesis_only`: Specifies if the model to be tested on the hypothesis only.
5. `--predictions_save_path`: The file where the model predictions will be saved.

Sample testing commands are:

`python3 transformer_test.py --model_name saved_model/roberta-large`