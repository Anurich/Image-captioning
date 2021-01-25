import torch
from pycocotools.coco import COCO
from tqdm import tqdm
import numpy as np
import nltk
import cv2
from vocabulary import Vocabulary
from nltk.tokenize import RegexpTokenizer
from PIL import Image
from torchvision import transforms

def getcaption_len(annotation_file, tokenizer,train=True):
    # here we call the training data
    coco = COCO(annotation_file)
    # now we can first get the ids
    ids  = list(coco.anns.keys())
    # we need to get the caption length
    caption_len = []
    for i in tqdm(np.arange(len(ids))):
        caption = coco.anns[ids[i]]["caption"].lower()
        caption_len.append(len(nltk.tokenize.word_tokenize(caption)))
    return caption_len, coco

class dataset(torch.utils.data.Dataset):
    def __init__(self, coco_ann_file, train, vocabThreshold=None, transforms = None):
        super().__init__()
        self.train = train
        self.ann_file = coco_ann_file
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.transform = transforms
        if train:
            self.caption_len, self._coco = getcaption_len(self.ann_file, self.tokenizer, train=True)
            self.vocab = Vocabulary(True,self._coco, vocabThreshold)
            self.ids   = list(self._coco.anns.keys())
        else:
            self._coco = COCO(coco_ann_file)
            self.ids  = list(self._coco.anns.keys())
            self.vocab = Vocabulary(train=False)

    def __getitem__(self, index):
        if self.train:
            anns_id  = self.ids[index]
            anns_dict = self._coco.anns[anns_id]
            caption = anns_dict["caption"].lower()
            imageName = anns_dict["image_id"]
            image_path = self._coco.loadImgs(imageName)[0]['file_name']
            img        =Image.open("../train2014/"+image_path).convert('RGB')
            if self.transform != None:
                img  = self.transform(img)
            # for captions
            caption =  nltk.tokenize.word_tokenize(caption)
            caption.insert(0, "<start>")
            caption.insert(len(caption), "<end>")
            # convert the caption to long tensor
            caption = [self.vocab.word_to_index[word] if self.vocab.word_to_index.get(word)!= None else self.vocab.word_to_index["<unk>"] for word in caption]
            caption = torch.Tensor(caption).long()
            return img, caption
        else:

            anns_id = self.ids[index]
            anns_dict = self._coco.anns[anns_id]
            original_image   = anns_dict["image_id"]
            image_path = self._coco.loadImgs(original_image)[0]['file_name']
            image_path = "../train2014/"+str(image_path)
            img  = Image.open(image_path).convert('RGB')
            orig_img =cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            if self.transform != None:
                img = self.transform(img)
            return img, torch.Tensor(orig_img)
    def trainIndices(self, batch_size):
        # randomly select the indices
        caption_length = np.random.choice(self.caption_len)
        #now we only  choose those caption that has length equal to caption_length
        caption_same_length_index = np.where([self.caption_len[i] == caption_length for i in np.arange(len(self.caption_len))])[0]
        #now we sample batch
        index = list(np.random.choice(caption_same_length_index, size=batch_size))
        return index
    def __len__(self):
        if self.train:
            return len(self.ids)
        else:
            return len(self.ids)

def get_data_loader(vocabulary_threshold, train, batch_size, transforms=None):
    if train:
        coco_annotation_file ="../annotations/captions_train2014.json"
        data = dataset(coco_annotation_file, train, vocabulary_threshold, transforms)
        # so the idea is to select the caption with set length
        # and pass through sampler
        index = data.trainIndices(batch_size)
        batch_sampler = torch.utils.data.SubsetRandomSampler(index)
        batchSampler = torch.utils.data.BatchSampler(batch_sampler, batch_size=batch_size, drop_last=False)
        train_loader = torch.utils.data.DataLoader(data, batch_sampler=batchSampler)
        return train_loader
    else:
        coco_annotation_file = "../annotations/captions_train2014.json"
        data = dataset(coco_annotation_file, False,transforms=transforms)
        test_loader  = torch.utils.data.DataLoader(data, batch_size=1, drop_last=False, shuffle=True)
        return test_loader

