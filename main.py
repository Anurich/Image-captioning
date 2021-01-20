import torch
import cv2
from model import encoder,decoder
import torchvision
import torch.nn as nn
import random
import numpy as np
from dataset import *
import math
import matplotlib.pyplot as plt
from torchvision import transforms
import sys
import pickle
from sklearn.model_selection import GroupShuffleSplit
import random


df = pd.read_csv("preprocessedCSV.csv")
test_size =  int(len(df)*0.2)
train_size = int(len(df)*0.8)
trainIndices, testIndices = torch.utils.data.random_split(df,[train_size, test_size])

def captionLength(train_df):
    caption_len = []
    for cap in train_df["comment"].values:
        try:
            caption_len.append(len(cap.lower().strip().split()))
        except:
            break
            continue
    return caption_len
def randomSelect(caption_len, batch_size):
    # here we randomly select the indices
    # than we only select those which has same length
    random_indices = random.choice(caption_len)
    # only hose index which has len
    selectedIndices = []
    for i in np.arange(len(caption_len)):
        if caption_len[i] == random_indices:
            selectedIndices.append(i)
    return list(np.random.choice(selectedIndices, batch_size)), random_indices

def check(train_df, indices):
    for d in train_df.iloc[indices]["comment"].values:
        print(len(word_tokenize(d)))

def read_data(batch_size, df, train=True):
    transform_train_test = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    # divide the data
    if train:

        train_df = df.iloc[trainIndices.indices]
        train_df = train_df[train_df['comment'].notna()]
        caption_len = captionLength(train_df)
        train_indices, random_indices = randomSelect(caption_len, batch_size)
        dsat = dataset(train_df, random_indices, transform_train_test)
        #check(train_df, train_indices)
        train_sampler  = torch.utils.data.SubsetRandomSampler(train_indices)
        batchSampler  = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=False)
        train_loader  = torch.utils.data.DataLoader(dsat, batch_sampler=batchSampler)
        return train_loader, train_df
    else:
        test_df = df.iloc[testIndices.indices]
        dsat_test = dataset(test_df, transforms = transform_train_test)
        caption_len = captionLength(test_df)
        test_indices,_ = randomSelect(caption_len, 1)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
        batchSampler = torch.utils.data.BatchSampler(test_sampler, batch_size=1, drop_last=False)
        test_loader = torch.utils.data.DataLoader(dsat_test, batch_sampler = batchSampler)
        return test_loader

def main(batch_size, train_df, trainLoader, embedding_dim, hidden_size, hidden_layer, index_to_word):
    vocab_size = len(index_to_word)+1
    enc = encoder(embedding_dim, batch_size)
    dec = decoder(vocab_size, embedding_dim, hidden_layer, hidden_size)
    iteration = 10
    #loss
    param = list(enc.dense.parameters()) + list(dec.parameters())
    criteria  = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(param, lr=0.001)
    total_steps = int(np.ceil(train_size/ batch_size))
    caption_len = captionLength(train_df)
    for epoch in range(iteration):
        total_loss = 0.0
        for step in range(total_steps):
            train_indices, _ = randomSelect(caption_len, batch_size)
            new_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
            trainLoader.batch_sampler.sampler = new_sampler
            data = next(iter(trainLoader))
            original_img, caption = data
            enc.zero_grad()
            dec.zero_grad()
            features = enc(original_img)
            prediction   = dec(features.long(), caption)
            #loss
            loss = criteria(prediction.view(caption.size(0)*caption.size(1),-1), caption.view(-1))

            loss.backward()
            optimizer.step()
            stats = "[%d/%d] Loss: %.4f, Perplexity: %5.4f "%(step, iteration, loss.item(), np.exp(loss.item()))
            print("\r" +stats, end="")
            sys.stdout.flush()
            total_loss += loss.item()
            if step % 100 ==0 and step != 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': enc.state_dict(),
                    'loss': total_loss/100,
                }, "loss_folder/encoder_"+str(epoch)+".pth")
                torch.save({
                    'model_state_dict':dec.state_dict()
                },"loss_folder/decoder_"+str(epoch)+".pth")
                total_loss = 0.0
                print("\r" + stats)

def test( batch_size, df, testLoader, index_to_word):

    enc = encoder(512, batch_size)
    enc.eval()
    dec = decoder(len(index_to_word)+1, 512, 1, 512)
    dec.eval()
    #load the model
    enc_weight = torch.load("loss_folder/encoder_2.pth")
    dec_weight = torch.load("loss_folder/decoder_2.pth")
    enc.load_state_dict(enc_weight["model_state_dict"])
    dec.load_state_dict(dec_weight["model_state_dict"])
    img, caption = next(iter(testLoader))
    print(img.shape)
    caption = caption[0]
    features = enc(img).unsqueeze(1)
    output   = dec.sample(features.float(), 27)
    sent  = ""

    for word in output:
        if index_to_word.get(word) !="START" and index_to_word.get(word)!="END" and word !=0:
            sent += index_to_word[word]+"  "
    print(sent)
    plt.imshow(img[0].permute(1,2,0).detach().numpy())
    plt.show()

if __name__  == "__main__":

    padSize =20
    batch_size = 32

    embedding_dim = 512
    hidden_size   = 512
    hidden_layer  = 1
    index_to_word = pickle.load(open("index_to_index.pickle","rb"))
    if len(os.listdir("loss_folder/")) == 0:
        train_loader, train_df  = read_data(batch_size, df, train=True)
        main(batch_size, train_df, train_loader,embedding_dim, hidden_size, hidden_layer, index_to_word)
    else:
        test_loader =  read_data(batch_size, df, train=False)
        test(batch_size, df, test_loader, index_to_word)
