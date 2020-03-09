from pathlib import Path
from collections import Counter
import torch
import torch.utils.data as data
import copy

class MyDataset(data.Dataset):
	def __init__(self, src_path, trg_path, src_vocab, trg_vocab):
		self.src_vocab = copy.copy(src_vocab)
		self.trg_vocab = copy.copy(trg_vocab)
		self.src_sentences = list()
		self.trg_sentences = list()

		self.max_length = 0
		
		max_len = 0
		with Path(src_path).open('r') as fs, Path(trg_path).open('r') as ft:
			for src, trg in zip(fs, ft):
				_src = src.strip().split()
				_trg = trg.strip().split()
				self.src_sentences.append(_src)
				self.trg_sentences.append(_trg)

				if len(_src) > max_len:
					max_len = len(_src)

				self.max_length = max_len + 1 #<eos>分追加

	def __getitem__(self, index):
		src_words = self.src_sentences[index]
		trg_words = self.trg_sentences[index]

		source = self.convert_words2ids(src_words, self.src_vocab, \
										self.src_vocab['<unk>'], eos=self.src_vocab['<eos>'])
		target = self.convert_words2ids(trg_words, self.trg_vocab, \
										self.trg_vocab['<unk>'], sos=self.trg_vocab['<sos>'])
		ground = self.convert_words2ids(trg_words, self.trg_vocab, \
										self.trg_vocab['<unk>'], eos=self.trg_vocab['<eos>'])

		source = torch.Tensor(source)
		target = torch.Tensor(target)
		ground = torch.Tensor(ground)

		return source, target, ground

	def __len__(self):
		return len(self.src_sentences)

	def convert_words2ids(self, words, vocab, unk, sos=None, eos=None):

		word_ids = [vocab[w] if w in vocab else unk for w in words]
		if sos is not None:
			word_ids.insert(0, sos)
		if eos is not None:
			word_ids.append(eos)

		return word_ids

def collate_fn(data):
	sources, targets, grounds = zip(*data)
	src_lengths = [len(src) for src in sources]
	trg_lengths = [len(trg) for trg in targets]
	_sources = torch.zeros(len(sources), max(src_lengths)).long()
	_targets = torch.zeros(len(targets), max(trg_lengths)).long()
	_tpairs = torch.zeros(len(grounds), max(trg_lengths)).long()

	for i, src in enumerate(sources):
		end = src_lengths[i]
		_sources[i, :end] = src
	
	for i, (trg, teach) in enumerate(zip(targets, grounds)):
		end = trg_lengths[i]
		_targets[i, :end] = trg
		_tpairs[i, :end] = teach

	return _sources, _targets, _tpairs
	
def get_vocab(src_path, trg_path, src_freq = 1, trg_freq = 1):
	src_file = Path(src_path)
	trg_file = Path(trg_path)

	src_vocab = {'<pad>':0, '<unk>':1, '<sos>':2, '<eos>':3}
	trg_vocab = {'<pad>':0, '<unk>':1, '<sos>':2, '<eos>':3}
	src_count = Counter()
	trg_count = Counter()

	with src_file.open('r') as f:
		for line in f:
			for word in line.strip().split():
				src_count[word] += 1

	for w, freq in src_count.most_common():
		if freq < src_freq:
			break
		if w not in src_vocab:
			src_vocab[w] = len(src_vocab)

	with trg_file.open('r') as f:
		for line in f:
			for word in line.strip().split():
				trg_count[word] += 1

	for w, freq in trg_count.most_common():
		if freq < trg_freq:
			break
		if w not in trg_vocab:
			trg_vocab[w] = len(trg_vocab)

	return src_vocab, trg_vocab
