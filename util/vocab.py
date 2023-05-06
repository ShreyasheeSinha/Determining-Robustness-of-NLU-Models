"""
This is a utility class
to create the vocabulary object for a given model.
This object must be saved in order to use
the model in the future.
"""

class Vocabulary:
    def __init__(self, vocab_size):
        UNK_TOKEN = "@UNK@"
        self.word2index = {UNK_TOKEN: 1}
        self.index2word = {1: UNK_TOKEN}
        self.n_words = 1
        self.vocab_size = vocab_size

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words + 1
            self.index2word[self.n_words + 1] = word
            self.n_words += 1
    
    def get_index(self, word):
        if word in self.word2index:
            return self.word2index[word]
        return self.word2index["@UNK@"]

    def get_word(self, index):
        return self.index2word[index]

    def get_word2index(self):
        return self.word2index

    def get_index2word(self):
        return self.index2word
    
    def get_vocab_size(self):
        return self.vocab_size