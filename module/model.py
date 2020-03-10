import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, hiddenSize, vocabSize, batchSize):
        super(Encoder, self).__init__()
        self.batchSize = batchSize
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize

        self.embedding = nn.Embedding(self.vocabSize, self.hiddenSize, padding_idx=0)

    def forward(self, source):
        self.clear()
        embedded = self.embedding(source)

    def clear(self):
       self.before_hidden = torch.zeros(self.batchSize, self.hiddenSize)


class Decoder(nn.Module):
    def __init__(self, hiddenSize, vocabSize, batchSize):
        super(Decoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize
        self.batchSize = batchSize

        self.embedding = nn.Embedding(self.vocabSize, self.hiddenSize, padding_idx=0)

    def forward(self, source, hidden, cell):
        embedded = self.embedding(source)
