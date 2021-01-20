import torch
import pandas as pd
import string
from sklearn.model_selection import train_test_split
import os
import re
import numpy as np
from collections import Counter
from nltk import word_tokenize
import pickle
from PIL import Image
from torchvision import transforms
import keras
from keras.preprocessing.sequence import pad_sequences

def read_data(fileName):
    df      = pd.read_csv(fileName, delimiter='|')
    df      = df.rename(columns={"image_name":"image_name"," comment_number":"comment_number"," comment":"comment"})
    return df

def preprocessText(data):
    sent=[]
    allSent = data["comment"].values
    for i in range(len(allSent)):
        try:
            sent_dat = allSent[i]
            sent_dat = re.sub('\d+','',sent_dat).strip()
            sent_dat = sent_dat.translate(str.maketrans('','',string.punctuation)).lower()
            data["comment"].iloc[i] = sent_dat
        except:
            continue
    return data

def savedPreprocessedFile(fileName):
    df = read_data(fileName)
    df = preprocessText(df)
    df.to_csv("preprocessedCSV.csv")

def buildVocab(fileName, vocabThreshold):
    df  =  pd.read_csv(fileName)
    df  =  df[df["comment"].notna()]
    allSent = "  ".join(df["comment"].values)
    # we will check if the vocab is less than threshold we don't consider it
    tokenizeWord = word_tokenize(allSent)
    vocab = Counter(tokenizeWord)
    word_to_index = {}
    word_to_index["END"] = 1
    word_to_index["UNK"] = 2
    word_to_index["START"] =3
    count = 4
    for key,value in vocab.items():
        if value >  vocabThreshold:
            word_to_index[key] = count
            count +=1

    index_to_word = {value: key for key,value in word_to_index.items()}
    # saving the pickle file
    with open("word_to_index.pickle", "wb") as handle:
        pickle.dump(word_to_index, handle)
    with open("index_to_index.pickle", "wb") as handle:
        pickle.dump(index_to_word, handle)

    print("Done...")

def collation_function(data):
    caption = []
    images  = []
    caption_len = []
    for i in range(len(data)):
        img, cap = data[i]
        caption.append(cap)
        images.append(img.cpu().numpy())
        caption_len.append(len(cap))
    # now we have caption length
    # let's use the max length to pad
    maxPad = max(caption_len)
    caption = pad_sequences(caption, maxlen= maxPad, padding='post')
    return torch.Tensor(np.array(images)), torch.Tensor(caption).long()



class dataset(torch.utils.data.Dataset):
    def __init__(self, dataFrame, total_len = None, transforms = None):
        super().__init__()
        # read the vocab
        self.word_to_index = pickle.load(open("word_to_index.pickle", "rb"))
        #read the file
        self.data = dataFrame
        self.transforms = transforms
        self.total_len = total_len

    def _preprocess(self, imageName, comment):
        img  = Image.open("../flickr30k_images/"+str(imageName)).convert('RGB')
        comment = "START "+str(comment).lower().strip()+" END"
        tokenizedWord = word_tokenize(comment)
        numericLabels = []
        for word in tokenizedWord:
            if self.word_to_index.get(word) != None:
                numericLabels.append(self.word_to_index[word])
            else:
                numericLabels.append(self.word_to_index["UNK"])
        return img, numericLabels

    def padSequence(self, sequence, padSize):
        if len(sequence) > padSize:
            return sequence[:padSize]
        else:
            for i in range(len(sequence), padSize):
                sequence.append(0)
        return sequence

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image, caption = row["image_name"], row["comment"]
        image, caption = self._preprocess(image, caption)
        if self.transforms != None:
            image = self.transforms(image)
        if self.total_len != None:
            if len(caption) != self.total_len:
                caption = self.padSequence(caption, self.total_len)
        return image, torch.Tensor(caption).long()
    def __len__(self):
        return len(self.data)
