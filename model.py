import torch
import torch.nn as nn
import torchvision

class encoder(nn.Module):
    def __init__(self,embed_size,batch_size):
        super(encoder,self).__init__()
        self.batch_size = batch_size
        resnet = torchvision.models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.dense     = nn.Linear(resnet.fc.in_features,embed_size)

    def forward(self,x):
        feature_conv = self.resnet(x)
        feature_conv = feature_conv.view(feature_conv.size(0),-1)
        feature_dense = self.dense(feature_conv)
        return feature_dense


class decoder(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_layer,hidden_size):
        super(decoder,self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim= embedding_dim
        self.hidden_layer = hidden_layer
        self.hidden_size  = hidden_size
        self.embed = nn.Embedding(vocab_size,embedding_dim)
        self.lstm  = nn.LSTM(embedding_dim,hidden_size,hidden_layer,batch_first=True, dropout=0.5)
        self.output = nn.Linear(hidden_size,vocab_size)

    def get_initial_state(self, batch_size):
        hidden_state = torch.zeros((self.hidden_layer, batch_size, self.hidden_size))
        cell_state = torch.zeros((self.hidden_layer, batch_size, self.hidden_size))
        return (hidden_state, cell_state)

    def forward(self,encoder_feature, caption):
        caption  =  caption[:,:-1]
        embed_output = self.embed(caption)
        # concatenate the data
        hiddenCell = self.get_initial_state(caption.size(0))
        embed_output =  torch.cat([encoder_feature.unsqueeze(1), embed_output],dim=1)
        # now pass it through lstm
        output, hiddenCell    = self.lstm(embed_output, hiddenCell)
        output       = self.output(output)
        return output

    def sample(self,inputs,maxlen):
        states = None
        predicted_sentence =[]
        for i in range(maxlen):
            outputs,states = self.lstm(inputs, states)
            outputs    = self.output(outputs).squeeze(1)
            targetIndex = outputs.argmax(dim=1)
            # Append result into predicted_sentence list
            predicted_sentence.append(targetIndex.item())
            # Update the input for next iteration
            inputs = self.embed(targetIndex).unsqueeze(1)
        return predicted_sentence
