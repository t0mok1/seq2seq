class Lang:
    def __init__(self, file_pass):
        self.word2index = {}
        self.index2word = {}
        self.vocab_num = 3
        self.word2count = {}

        self.pad = '<pad>'
        self.word2index[self.pad] = 0
        self.index2word[0] = self.pad

        self.eos = '<EOS>'
        self.word2index[self.eos] = 1
        self.index2word[1] = self.eos

        self.unk = '<UNK>'
        self.word2index[self.unk] = 2
        self.index2word[2] = self.unk

        self.file_pass = file_pass

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

    def addSentence(self,sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.vocab_num
            self.word2count[word] = 1
            self.index2word[self.vocab_num] = word
            self.vocab_num += 1
        else:
            self.word2count[word] += 1

    def makeDic(self):
        with open(self.file_pass, "r") as f:
            lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n','')
            self.addSentence(lines[i])
