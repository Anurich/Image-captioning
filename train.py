import torch
import numpy as np
from torchvision import transforms
from dataset import *
from model import encoder, decoder
import sys
def train():
    # few things that we have define
    batch_size = 32
    train = True
    transform_train = transforms.Compose([
        transforms.Resize(256),                          # smaller edge of image resized to 256
        transforms.RandomCrop(224),                      # get 224x224 crop from random location
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])
    iteration = 3
    vocabulary_threshold = 5
    embed_size = 512
    hidden_size = 512
    hidden_layer =1
    model_save = "model_storage/"
    # calling the dataloader
    train_dataLoader  = get_data_loader(vocabulary_threshold, train, batch_size, transform_train)
    enc = encoder(embed_size, batch_size)
    dec = decoder(len(train_dataLoader.dataset.vocab.word_to_index), embed_size, hidden_layer, hidden_size)
    params = list(enc.dense.parameters()) + list(dec.parameters())
    criteria = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params, lr=0.001,  betas=(0.9, 0.999), eps=1e-08)
    steps_per_epoch = int(np.math.ceil(len(train_dataLoader.dataset.caption_len)/batch_size))
    for epoch in range(iteration):
        for step in range(steps_per_epoch):
            index  = train_dataLoader.dataset.trainIndices(batch_size)
            sampler = torch.utils.data.SubsetRandomSampler(index)
            train_dataLoader.batch_sampler.sampler = sampler
            img, caption = next(iter(train_dataLoader))
            enc.zero_grad()
            dec.zero_grad()
            features  = enc(img)
            prediction = dec(features, caption)
            loss       = criteria(prediction.view(caption.size(0)*caption.size(1),-1), caption.view(-1))
            loss.backward()
            optimizer.step()
            stats = "[%d/%d] LOSS: %.4f, PERPLEXITY: %5.4f "%(step, iteration, loss.item(), np.exp(loss.item()))
            print("\r "+stats, end="")
            sys.stdout.flush()
            if step%1000 ==0 and step != 0:
                # here we save the weights
                torch.save({
                    "model_state":enc.state_dict()
                },model_save+"encoder_"+str(step)+".pth")
                torch.save({
                    "model_state":dec.state_dict()
                },model_save+"decoder_"+str(step)+".pth")
                print("\r"+stats)

if __name__ == "__main__":
    train()
