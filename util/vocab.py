"""
This is a utility class
to create the vocabulary object for a given model.
This object must be saved in order to use
the model in the future.
"""
from nltk.tokenize import word_tokenize

class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0

    def addSentence(self, sentence):
        for word in word_tokenize(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words + 1
            self.word2count[word] = 1
            self.index2word[self.n_words + 1] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def get_index(self, word):
        return self.word2index[word]

    def get_word(self, index):
        return self.index2word[index]

    def get_word_count(self, word):
        return self.word2count[word]

    def get_indices(self):
        return self.word2index

    def get_words(self):
        return self.index2word