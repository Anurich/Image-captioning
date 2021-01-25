import torch
from dataset import *
import os
import matplotlib.pyplot as plt
from model import encoder, decoder
import pickle
from torchvision import transforms
def readVocab():
    file_name = "index_to_word.pickle"
    return pickle.load(open(file_name,"rb"))

def test():
    embed_size = 512
    hidden_size = 512
    weights = "model_storage/"
    weight_list = os.listdir(weights)
    selectedWeight = None
    index_to_word = readVocab()
    maxVal = 0
    transform_train = transforms.Compose([
        transforms.Resize(256),                          # smaller edge of image resized to 256
        transforms.RandomCrop(224),                      # get 224x224 crop from random location
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])
    for weight in weight_list:
        if "encoder" in weight:
            val  = int(weight.split(".")[0].split("_")[1])
            if val  > maxVal:
                selectedWeight  = weight
                maxVal = val

    encoder_weight = selectedWeight
    decoder_weight = selectedWeight.replace("encoder","decoder")
    enc_weight = torch.load(weights+encoder_weight)
    dec_weight = torch.load(weights+decoder_weight)
    enc = encoder(embed_size, batch_size=1)
    enc.eval()
    enc.load_state_dict(enc_weight["model_state"])
    dec = decoder(len(index_to_word), embed_size, 1, hidden_size)
    dec.eval()
    dec.load_state_dict(dec_weight["model_state"])
    test_loader  = get_data_loader(5,False,1,transform_train)
    img_test, original_img = next(iter(test_loader))
    features = enc(img_test)
    output = dec.sample(features.unsqueeze(1), 20)
    sentence = ""
    for val in output:
        if val!= 0 and val != 1 and val!=2:
            sentence += index_to_word[val]+"  "
    plt.imshow(np.uint8(original_img.squeeze(0).numpy()))
    plt.text(100,400,sentence, style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    plt.show()
if __name__ == "__main__":
    test()
