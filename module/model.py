import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, hiddenSize, vocabSize):
        super(Encoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize

        self.embedding = nn.Embeddingi(self.vocabSize, self.vocabSize, padding_index=0)
        self.lstmCell = nn.LSTMCell(self.hidden, self.hidden)

    def forward(self):

    def clear(self):
        
