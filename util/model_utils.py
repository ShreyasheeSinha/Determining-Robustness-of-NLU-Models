import datetime
import torch
import os
import pickle
import numpy as np

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def save_model(model, optimizer, model_name, path, training_stats):
    # save model state dict
    full_path = os.path.join(path, model_name)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_stats': training_stats
    }, full_path)

def load_model(path, model_name):
    # Load model state dict along with model data
    full_path = os.path.join(path, model_name)
    checkpoint = torch.load(full_path)
    return checkpoint

def save_vocab(path, vocab, model_name):
    name = model_name + "_vocab.obj"
    full_path = os.path.join(path, name)
    with open(full_path, 'wb') as handle:
        pickle.dump(vocab, handle)

def load_vocab(path, model_name):
    name = model_name + "_vocab.obj"
    full_path = os.path.join(path, name)
    with open(full_path, 'rb') as handle:
        vocab = pickle.load(handle)
    return vocab

def save_model_config(path, model_name, options):
    name = model_name + ".config"
    full_path = os.path.join(path, name)
    with open(full_path, 'wb') as handle:
        pickle.dump(options, handle)

def load_model_config(path, model_name):
    name = model_name + ".config"
    full_path = os.path.join(path, name)
    with open(full_path, 'rb') as handle:
        config = pickle.load(handle)
    return config

def create_embedding_matrix(embeddings_index, embedding_size, vocab):
    print("Creating embeddings..")
    vocab_size = vocab.get_vocab_size()
    embedding_matrix = 1 * np.random.randn(vocab_size + 1, embedding_size)
    embedded_count = 0
    for word, lang_word_index in vocab.get_word2index().items():
        if embeddings_index.get(word) is not None:
            embedding_matrix[lang_word_index] = embeddings_index.get(word)
            embedded_count += 1

    print("Embedded count:", embedded_count)
    return embedding_matrix
