class dictionary:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.vocab_num = 3

        self.pad = '<pad>'
        self.word2index[self.pad] = 0
        self.index2word[0] = self.pad

        self.eos = '<EOS>'
        self.word2index[self.eos] = 1
        self.index2word[1] = self.eos

        self.unk = '<UNK>'
        self.word2index[self.unk] = 2
        self.index2word[2] = self.unk

    def get_token(self, index):
        try:
            return self.index2word[index]
        except KeyError:
            return self.unk

    def get_index(self, token):
        try:
            return self.word2index[token]
        except KeyError:
            return self.word2index[self.unk]