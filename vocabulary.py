import torch
from collections import Counter
from nltk import word_tokenize
import pickle
import os
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import numpy as np

class Vocabulary:
    def __init__(self, train=True, coco = None, threshold = None):

        self.filename_windex = "word_to_index.pickle"
        self.filename_indexw = "index_to_word.pickle"
        self.word_to_index  = {}
        self.index_to_word  = {}
        if train:
            self.threshVocab = threshold
            self.coco        = coco
            self.ids         = list(coco.anns.keys())
            # let's check if vocabulary exist if not we make it
            # otherwise we read vocab
            if os.path.exists(self.filename_windex) and os.path.exists(self.filename_indexw):
                self.word_to_index = self.readvocabulary(self.filename_windex)
                self.index_to_word = self.readvocabulary(self.filename_indexw)
            else:
                self.makeVocab()
                self.savevocabulary(self.word_to_index, self.filename_windex)
                self.savevocabulary(self.index_to_word, self.filename_indexw)
        else:
            self.word_to_index = self.readvocabulary(self.filename_windex)
            self.index_to_word = self.readvocabulary(self.filename_indexw)


    def makeVocab(self):
        vocab = Counter()
        tokenizer = RegexpTokenizer(r'\w+')
        for i in tqdm(np.arange(len(self.ids))):
            caption =  self.coco.anns[self.ids[i]]["caption"]
            splitted_caption = word_tokenize(caption)
            vocab.update(splitted_caption)
        # now we can check and remove anything that is not greater than the threshold vocabulary
        vocab_copy = vocab.copy()
        for k, v in vocab_copy.items():
            if v < self.threshVocab:
                vocab.pop(k)
        self.get_word_to_index(vocab)
        self.get_index_to_word()


    def get_word_to_index(self, vocab):
        self.word_to_index["<start>"] = 0
        self.word_to_index["<end>"] = 1
        self.word_to_index["<unk>"] = 2
        count = 3
        for k, v in vocab.items():
            self.word_to_index[k] = count
            count +=1

    def get_index_to_word(self):
        for k, v in enumerate(self.word_to_index):
            self.index_to_word[k] = v

    def savevocabulary(self, vocab, filename):
        with open(filename,'wb') as handle:
            pickle.dump(vocab, handle)

    def readvocabulary(self, filename):
        return  pickle.load(open(filename, "rb"))




